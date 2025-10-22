'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Brain, AlertCircle, Sparkles, MessageCircle, Trash2, Moon, Sun } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  sentiment?: string;
  isCrisis?: boolean;
  timestamp: Date;
}

const SENTIMENT_CONFIGS: Record<string, { color: string; emoji: string; gradient: string }> = {
  'Anxiety': { 
    color: 'bg-amber-100 text-amber-800 border-amber-300',
    emoji: '😰',
    gradient: 'from-amber-400 to-orange-500'
  },
  'Depression': { 
    color: 'bg-blue-100 text-blue-800 border-blue-300',
    emoji: '😔',
    gradient: 'from-blue-400 to-indigo-500'
  },
  'Suicidal': { 
    color: 'bg-red-100 text-red-800 border-red-300',
    emoji: '🆘',
    gradient: 'from-red-500 to-rose-600'
  },
  'Stress': { 
    color: 'bg-orange-100 text-orange-800 border-orange-300',
    emoji: '😫',
    gradient: 'from-orange-400 to-amber-500'
  },
  'Bipolar': { 
    color: 'bg-purple-100 text-purple-800 border-purple-300',
    emoji: '🎭',
    gradient: 'from-purple-400 to-pink-500'
  },
  'Emotional Intensity': { 
    color: 'bg-pink-100 text-pink-800 border-pink-300',
    emoji: '💗',
    gradient: 'from-pink-400 to-rose-500'
  },
  'Normal': { 
    color: 'bg-green-100 text-green-800 border-green-300',
    emoji: '😊',
    gradient: 'from-green-400 to-emerald-500'
  },
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "Hello, I'm here to listen and support you. This is a safe space where you can share what's on your mind. How are you feeling today?",
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const clearChat = () => {
    if (confirm('Are you sure you want to clear the conversation?')) {
      setMessages([{
        id: Date.now().toString(),
        type: 'bot',
        content: "Hello again. I'm here whenever you need to talk. What's on your mind?",
        timestamp: new Date(),
      }]);
    }
  };

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
    setIsTyping(true);
    setError('');

    // Prepare conversation history
    const history = messages.map(msg => ({
      role: msg.type === 'user' ? 'user' : 'assistant',
      content: msg.content
    }));

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: input,
          conversation_history: history 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // Simulate typing delay for more natural feel
      setTimeout(() => {
        setIsTyping(false);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.response,
          sentiment: data.sentiment,
          isCrisis: data.is_crisis,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, botMessage]);
      }, 800);

    } catch (err) {
      setIsTyping(false);
      setError('Unable to connect. Please check if the backend is running.');
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

  const bgClass = darkMode 
    ? 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900' 
    : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50';

  const cardBg = darkMode ? 'bg-slate-800/90' : 'bg-white/90';
  const textColor = darkMode ? 'text-white' : 'text-gray-800';

  return (
    <div className={`min-h-screen ${bgClass} transition-colors duration-300`}>
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-20 left-10 w-72 h-72 ${darkMode ? 'bg-purple-500' : 'bg-purple-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob`}></div>
        <div className={`absolute top-40 right-10 w-72 h-72 ${darkMode ? 'bg-pink-500' : 'bg-pink-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000`}></div>
        <div className={`absolute -bottom-8 left-1/2 w-72 h-72 ${darkMode ? 'bg-blue-500' : 'bg-blue-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000`}></div>
      </div>

      {/* Header */}
      <header className={`${cardBg} backdrop-blur-xl shadow-lg border-b ${darkMode ? 'border-purple-700' : 'border-purple-100'} relative z-10`}>
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-linear-to-r from-purple-500 to-pink-500 rounded-xl blur opacity-75 animate-pulse"></div>
              <div className="relative bg-linear-to-br from-purple-500 to-pink-500 p-2.5 rounded-xl">
                <Brain className="w-7 h-7 text-white" />
              </div>
            </div>
            <div>
              <h1 className={`text-xl font-bold bg-linear-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent ${darkMode && 'from-purple-400 to-pink-400'}`}>
                MindfulAI Therapist
              </h1>
              <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Your compassionate companion</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${darkMode ? 'bg-slate-700 text-yellow-400' : 'bg-purple-100 text-purple-600'} hover:scale-110 transition-transform`}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button
              onClick={clearChat}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg ${darkMode ? 'bg-slate-700 text-red-400 hover:bg-slate-600' : 'bg-red-50 text-red-600 hover:bg-red-100'} transition-colors text-sm`}
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          </div>
        </div>
      </header>

      {/* Crisis Banner */}
      <div className="max-w-5xl mx-auto px-4 py-3 relative z-10">
        <div className={`${darkMode ? 'bg-red-900/50 border-red-700' : 'bg-red-50 border-red-200'} border rounded-xl p-3 flex items-start gap-2 backdrop-blur-sm`}>
          <AlertCircle className={`w-5 h-5 ${darkMode ? 'text-red-400' : 'text-red-600'} shrink-0 mt-0.5`} />
          <div className={`text-sm ${darkMode ? 'text-red-300' : 'text-red-800'}`}>
            <strong>Crisis Support:</strong> If you're in immediate danger, please call emergency services or contact a crisis helpline immediately.
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="max-w-5xl mx-auto px-4 pb-4 relative z-10">
        <div className={`${cardBg} backdrop-blur-xl rounded-3xl shadow-2xl border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} flex flex-col h-[calc(100vh-240px)]`}>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((message, index) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className={`max-w-[85%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                  {message.sentiment && (
                    <div className="mb-2 flex items-center gap-2 animate-slideIn">
                      <span className={`text-xs px-3 py-1.5 rounded-full border backdrop-blur-sm ${SENTIMENT_CONFIGS[message.sentiment]?.color || 'bg-gray-100 text-gray-800'}`}>
                        <span className="mr-1">{SENTIMENT_CONFIGS[message.sentiment]?.emoji}</span>
                        {message.sentiment}
                      </span>
                      {message.isCrisis && (
                        <div className="flex items-center gap-1 text-red-600 animate-pulse">
                          <AlertCircle className="w-4 h-4" />
                          <span className="text-xs font-medium">Crisis Detected</span>
                        </div>
                      )}
                    </div>
                  )}
                  <div
                    className={`rounded-2xl px-5 py-3.5 shadow-lg transform transition-all hover:scale-[1.02] ${
                      message.type === 'user'
                        ? `bg-linear-to-br ${SENTIMENT_CONFIGS[message.sentiment || 'Normal']?.gradient || 'from-purple-500 to-pink-500'} text-white`
                        : message.isCrisis
                        ? `${darkMode ? 'bg-red-900/50 border-2 border-red-500' : 'bg-red-50 border-2 border-red-300'} ${darkMode ? 'text-red-200' : 'text-gray-800'}`
                        : `${darkMode ? 'bg-slate-700/80' : 'bg-linear-to-br from-gray-50 to-gray-100'} ${textColor}`
                    }`}
                  >
                    {message.type === 'bot' && !message.isCrisis && (
                      <div className="flex items-center gap-2 mb-2 opacity-70">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span className="text-xs font-medium">AI Therapist</span>
                      </div>
                    )}
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                  </div>
                  <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'} mt-1.5 px-2 flex items-center gap-1`}>
                    <MessageCircle className="w-3 h-3" />
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="flex justify-start animate-fadeIn">
                <div className={`${darkMode ? 'bg-slate-700/80' : 'bg-linear-to-br from-gray-50 to-gray-100'} rounded-2xl px-5 py-3.5 flex items-center gap-2 shadow-lg`}>
                  <div className="flex gap-1">
                    <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`}></div>
                    <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`} style={{ animationDelay: '0.2s' }}></div>
                    <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`} style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Listening...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Error Message */}
          {error && (
            <div className="px-6 pb-2">
              <div className={`${darkMode ? 'bg-red-900/50 border-red-700' : 'bg-red-50 border-red-200'} border rounded-xl p-3 text-sm ${darkMode ? 'text-red-300' : 'text-red-700'} animate-shake`}>
                {error}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className={`border-t ${darkMode ? 'border-purple-700/50' : 'border-gray-200'} p-4 backdrop-blur-sm`}>
            <div className="flex gap-3 items-end">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Share what's on your mind... (Press Enter to send)"
                className={`flex-1 resize-none rounded-2xl border ${darkMode ? 'border-purple-700/50 bg-slate-700/50 text-white placeholder-gray-500' : 'border-gray-300 bg-white'} px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm max-h-32 transition-all`}
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="bg-linear-to-br from-purple-500 to-pink-500 text-white p-4 rounded-2xl hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105 active:scale-95 disabled:hover:scale-100"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <div className={`flex items-center justify-between mt-3 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>Secure & Confidential</span>
              </div>
              <span>Not a replacement for professional help</span>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes blob {
          0%, 100% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
        }

        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }

        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }

        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }

        .animate-blob {
          animation: blob 7s infinite;
        }

        .animation-delay-2000 {
          animation-delay: 2s;
        }

        .animation-delay-4000 {
          animation-delay: 4s;
        }

        .animate-shake {
          animation: shake 0.4s ease-in-out;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: transparent;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #a855f7, #ec4899);
          border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #9333ea, #db2777);
        }
      `}</style>
    </div>
  );
}