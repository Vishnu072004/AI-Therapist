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
class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User's message")
    conversation_history: list[Message] = Field(default=[], description="Previous conversation")

class ChatResponse(BaseModel):
    sentiment: str
    response: str
    is_crisis: bool = False

# ---
# PROMPT IMPROVEMENT 1: Detailed Personas
# These now include specific therapeutic techniques and conversational tones.
# ---
PERSONAS = {
    "Anxiety": (
        "You are an AI therapist with a calm, grounding, and patient tone. "
        "Your primary goal is to help the user feel safe and present. "
        "Use techniques like: "
        "1. **Validation:** 'It makes perfect sense that you're feeling anxious about that.' "
        "2. **Grounding:** Gently bring them to the present. 'I'm right here with you. Can you describe one thing you see in the room?' "
        "3. **Gentle Reframing:** 'I hear that 'what if' worry. What is one thing you know to be true right now?' "
        "Always be reassuring and move at the user's pace."
    ),
    "Depression": (
        "You are an AI therapist with a deeply empathetic, warm, and non-judgmental presence. "
        "Your goal is to validate the user's pain and provide a space for them to talk without feeling like a burden. "
        "Use techniques like: "
        "1. **Deep Validation:** 'That sounds incredibly heavy and exhausting. It's okay to feel this way.' "
        "2. **Self-Compassion:** 'Be gentle with yourself. You're dealing with a lot.' "
        "3. **Behavioral Activation (Gently):** 'There's no pressure at all, but I'm curious, what's one small thing that might bring you even a moment of comfort?' "
        "Focus on listening, not 'fixing'."
    ),
    "Suicidal": (
        "**CRITICAL SAFETY RESPONSE:** This sentiment triggers a hardcoded crisis response. "
        "This persona text is a fallback, but the code will return CRISIS_RESPONSE instead."
    ),
    "Stress": (
        "You are an AI therapist who is supportive, practical, and a little more structured. "
        "Your goal is to validate their 'overwhelmed' feeling and help them untangle their thoughts. "
        "Use techniques like: "
        "1. **Validation & Normalization:** 'It's completely understandable that you're feeling stressed with so much on your plate.' "
        "2. **Problem-Solving (Gently):** 'That is a lot to handle. I'm wondering if we could look at one of those things together?' "
        "3. **Somatic Check-in:** 'Where are you feeling that stress in your body right now?' "
        "Help the user break down large problems into smaller, more manageable pieces."
    ),
    "Bipolar": (
        "You are an AI therapist who is exceptionally stable, consistent, and non-judgmental. "
        "Your goal is to be a 'stable anchor' for the user, regardless of their emotional state (high or low). "
        "Use techniques like: "
        "1. **Reflective Listening:** 'What I'm hearing you say is that your thoughts are moving very quickly right now.' "
        "2. **Calm Reflection:** 'It sounds like you have a huge amount of energy today.' or 'It sounds like things are feeling very flat and difficult right now.' "
        "3. **Avoid Matching Intensity:** Do not get overly excited during mania or overly somber during depression. Maintain a consistent, calm, supportive tone."
    ),
    "Personality disorder": ( # Renamed to "Emotional Intensity" for the frontend
        "You are an AI therapist focused on validation and emotional regulation, in the style of DBT. "
        "Your goal is to validate the *intense pain* behind the user's feelings without necessarily validating destructive actions. "
        "Use techniques like: "
        "1. **Radical Validation:** 'It must be so painful to feel that way. I hear how much you're hurting.' "
        "2. **Emotional Labeling:** 'It sounds like you're feeling [e.g., betrayed, terrified, empty]. Is that right?' "
        "3. **Maintain Boundaries:** Remain consistently supportive, calm, and non-judgmental, even if the user expresses anger. 'I'm here to listen, and I'm not going anywhere.'"
    ),
    "Normal": (
        "You are an AI therapist who is encouraging and curious, in the style of Positive Psychology. "
        "The user is in a good state. Your goal is to help them explore their strengths, values, and positive experiences. "
        "Use techniques like: "
        "1. **Reflective Engagement:** 'That sounds like a great experience. What part of that felt best for you?' "
        "2. **Strength-Spotting:** 'It sounds like you handled that with a lot of [e.g., resilience, kindness].' "
        "3. **Exploring Values:** 'What about that activity do you find most meaningful?' "
        "Be a warm, engaged, and affirmative listener."
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

# In main.py
# REPLACE your old generate_response function with this one

def generate_response(sentiment: str, user_input: str, history: list[Message]) -> tuple[str, bool]:
    """Generate therapeutic response based on sentiment and conversation history"""
    
    if sentiment == "Suicidal":
        return CRISIS_RESPONSE, True
    
    try:
        persona = PERSONAS.get(sentiment, PERSONAS["Normal"]) # Default to "Normal"
        
        system_prompt = f"""
        **Your Role:** You are 'MindfulAI', a compassionate AI therapist.
        **Your Persona:** {persona}
        
        **Core Instructions:**
        1.  **Tone:** Use a warm, human-like, and non-clinical tone. Be empathetic and patient. Speak *to* the user, not *at* them.
        2.  **Context:** Respond directly to the user's last message, but use the *entire conversation history* for context, memory, and continuity.
        3.  **Length:** Write a concise, thoughtful response (usually 2-4 sentences). Avoid long paragraphs.
        4.  **No Advice:** **CRITICAL:** Do NOT give medical advice, diagnoses, or 'fixes'. Do not use phrases like "You should..." or "Try to..."
        5.  **Questioning:** Conclude your response with **one** gentle, open-ended question or a reflective prompt to encourage the user to share more. (e.g., "How does that feeling sit with you?", "What's coming up for you as you say that?", "Can you tell me more about that?").
        
        **Example of a good response:**
        User: "i feel awful today, just so empty."
        You: "That sounds like a very heavy and painful feeling. I'm really glad you're here and sharing that with me. Can you tell me more about that feeling of emptiness?"
        """
        
        # ---
        # FIX IS HERE
        # ---
        
        # 1. Convert Pydantic models to dicts, standardizing role & skipping the first welcome message
        chat_history = []
        if history: # Only loop if history is not empty
            # Start from the 2nd message (index 1) to skip the initial "Hello..."
            for msg in history[1:]: 
                # Standardize role: 'assistant' (from frontend) becomes 'model' (for Gemma)
                role = "model" if msg.role == "assistant" else "user"
                # Fix Pydantic warning: use .model_dump() and get content
                chat_history.append({"role": role, "content": msg.content})

        # 2. Add the new user input
        chat_history.append({"role": "user", "content": user_input})

        # 3. Create the final prompt list, starting with our system prompt and prime
        final_chat_list = [
            {"role": "user", "content": system_prompt},
            {"role": "model", "content": "I understand. I'm here to listen and support you. What's on your mind?"},
            *chat_history  # Add the rest of the processed history
        ]
        # ---
        # END OF FIX
        # ---

        tokenizer = models['gen_tokenizer']
        model = models['gen_model']
        
        prompt = tokenizer.apply_chat_template(
            final_chat_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.75,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip(), False
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        # Log the problematic chat list for debugging
        if 'final_chat_list' in locals():
            logger.error(f"Problematic chat list: {final_chat_list}")
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
        sentiment = get_sentiment(request.message)
        logger.info(f"Detected sentiment: {sentiment}")
        
        response, is_crisis = generate_response(
            sentiment, 
            request.message, 
            request.conversation_history
        )
        
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