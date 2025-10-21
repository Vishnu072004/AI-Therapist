import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    logger.info("Loading models...")
    
    try:
        # Load sentiment model
        sentiment_model_name = os.getenv("SENTIMENT_MODEL", "Coolstew07/fine-tuned-roberta")
        models['sentiment_tokenizer'] = RobertaTokenizer.from_pretrained(sentiment_model_name)
        models['sentiment_model'] = RobertaForSequenceClassification.from_pretrained(sentiment_model_name)
        models['sentiment_model'].eval()
        logger.info("Sentiment model loaded successfully")
        
        # Load generative model
        gen_model_name = os.getenv("GEN_MODEL", "google/gemma-2b-it")
        models['gen_tokenizer'] = AutoTokenizer.from_pretrained(gen_model_name)
        models['gen_model'] = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        models['gen_model'].eval()
        logger.info("Generative model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Cleaning up models...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="AI Therapist API",
    description="Mental health support chatbot with sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User's message")

class ChatResponse(BaseModel):
    sentiment: str
    response: str
    is_crisis: bool = False

# Persona definitions
PERSONAS = {
    "Anxiety": (
        "You are a calm, grounding AI therapist. Your goal is to validate the "
        "user's anxious feelings and gently help them feel more present."
    ),
    "Depression": (
        "You are an empathetic, patient, and warm AI therapist. Your goal is to "
        "acknowledge the user's pain, validate their feelings, and create a "
        "safe, non-judgmental space."
    ),
    "Suicidal": (
        "**CRITICAL SAFETY RESPONSE:** You are an AI focused on crisis intervention. "
        "Your ONLY goal is to respond with immediate, non-judgmental support "
        "and provide a crisis hotline."
    ),
    "Stress": (
        "You are a supportive and practical AI therapist. Your goal is to "
        "validate the user's feelings of being overwhelmed and gently help "
        "them identify the source of their stress."
    ),
    "Bipolar": (
        "You are a stable, non-judgmental AI therapist. Your goal is to "
        "listen and calmly reflect the user's feelings, without getting "
        "pulled into emotional highs or lows. Be a stable anchor."
    ),
    "Personality disorder": (
        "You are a consistent, clear, and boundaried AI therapist. "
        "Your goal is to offer validation for the user's intense emotions "
        "while maintaining a stable, supportive presence."
    ),
    "Normal": (
        "You are a friendly and positive AI therapist. The user seems to be "
        "in a good state. Your goal is to engage with them in a light, "
        "encouraging, and affirmative way."
    )
}

CRISIS_RESPONSE = (
    "It sounds like you are going through an incredibly painful time right now. "
    "Please know that your feelings are valid and you are not alone. "
    "If you are in immediate distress, please reach out for help. "
    "You can connect with people who can support you by calling or texting:\n\n"
    "🇺🇸 988 (US Suicide & Crisis Lifeline)\n"
    "🇬🇧 111 (UK)\n"
    "🇮🇳 91529 87821 (AASRA, India)\n\n"
    "These services are available 24/7 and completely confidential."
)

def get_sentiment(text: str) -> str:
    """Predict sentiment using the fine-tuned model"""
    try:
        tokenizer = models['sentiment_tokenizer']
        model = models['sentiment_model']
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
        sentiment = model.config.id2label[predicted_class_id]
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

def generate_response(sentiment: str, user_input: str) -> tuple[str, bool]:
    """Generate therapeutic response based on sentiment"""
    
    # Handle crisis situations
    if sentiment == "Suicidal":
        return CRISIS_RESPONSE, True
    
    try:
        persona = PERSONAS.get(sentiment, "You are a helpful and kind AI therapist.")
        
        prompt_template = [
            {"role": "user", "content": f"""
**Instructions:** Adopt this persona: '{persona}'. 
Read the user's statement, generate a short (2-3 sentence) empathetic response.
**CRITICAL:** Do NOT give medical advice or a diagnosis. 
End with one gentle, open-ended question.

**User's Statement:** "{user_input}"
"""},
        ]
        
        tokenizer = models['gen_tokenizer']
        model = models['gen_model']
        
        prompt = tokenizer.apply_chat_template(
            prompt_template, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        if "model\n" in response:
            response = response.split("model\n")[-1]
        elif prompt in response:
            response = response[len(prompt):].strip()
        
        return response.strip(), False
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        raise HTTPException(status_code=500, detail="Response generation failed")

@app.get("/")
async def root():
    return {"message": "AI Therapist API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all(key in models for key in ['sentiment_model', 'gen_model'])
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Detect sentiment
        sentiment = get_sentiment(request.message)
        logger.info(f"Detected sentiment: {sentiment}")
        
        # Generate response
        response, is_crisis = generate_response(sentiment, request.message)
        
        # Sanitize sentiment label for frontend
        display_sentiment = sentiment
        if sentiment == "Personality disorder":
            display_sentiment = "Emotional Intensity"
        
        return ChatResponse(
            sentiment=display_sentiment,
            response=response,
            is_crisis=is_crisis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)