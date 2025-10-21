'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Brain, AlertCircle, Heart, Loader2 } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  sentiment?: string;
  isCrisis?: boolean;
  timestamp: Date;
}

const SENTIMENT_COLORS: Record<string, string> = {
  'Anxiety': 'bg-yellow-100 text-yellow-800 border-yellow-300',
  'Depression': 'bg-blue-100 text-blue-800 border-blue-300',
  'Suicidal': 'bg-red-100 text-red-800 border-red-300',
  'Stress': 'bg-orange-100 text-orange-800 border-orange-300',
  'Bipolar': 'bg-purple-100 text-purple-800 border-purple-300',
  'Emotional Intensity': 'bg-pink-100 text-pink-800 border-pink-300',
  'Normal': 'bg-green-100 text-green-800 border-green-300',
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "Hello, I'm here to listen and support you. How are you feeling today?",
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: data.response,
        sentiment: data.sentiment,
        isCrisis: data.is_crisis,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Unable to connect to the server. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-purple-100">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-xl">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              AI Therapist
            </h1>
            <p className="text-xs text-gray-500">Your mental wellness companion</p>
          </div>
        </div>
      </header>

      {/* Crisis Banner */}
      <div className="max-w-4xl mx-auto px-4 py-3">
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-red-800">
            <strong>Crisis Support:</strong> If you're in immediate danger, please call emergency services or contact a crisis helpline.
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="max-w-4xl mx-auto px-4 pb-4">
        <div className="bg-white rounded-2xl shadow-xl border border-purple-100 flex flex-col h-[calc(100vh-220px)]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                  {message.sentiment && (
                    <div className="mb-2 flex items-center gap-2">
                      <span className={`text-xs px-2 py-1 rounded-full border ${SENTIMENT_COLORS[message.sentiment] || 'bg-gray-100 text-gray-800'}`}>
                        {message.sentiment}
                      </span>
                      {message.isCrisis && (
                        <AlertCircle className="w-4 h-4 text-red-600" />
                      )}
                    </div>
                  )}
                  <div
                    className={`rounded-2xl px-4 py-3 ${
                      message.type === 'user'
                        ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
                        : message.isCrisis
                        ? 'bg-red-50 border-2 border-red-300 text-gray-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  </div>
                  <p className="text-xs text-gray-400 mt-1 px-2">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl px-4 py-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-purple-500" />
                  <span className="text-sm text-gray-600">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Error Message */}
          {error && (
            <div className="px-6 pb-2">
              <div className="bg-red-50 border border-red-200 rounded-lg p-2 text-sm text-red-700">
                {error}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4">
            <div className="flex gap-2 items-end">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Share what's on your mind..."
                className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm max-h-32"
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="bg-gradient-to-br from-purple-500 to-pink-500 text-white p-3 rounded-xl hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
              <Heart className="w-3 h-3" />
              <span>This is a support tool, not a replacement for professional help</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}