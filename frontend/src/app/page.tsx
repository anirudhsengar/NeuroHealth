"use client";

import ReactMarkdown from 'react-markdown';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, User, BrainCircuit, Activity, HeartPulse, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';

type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning?: string[];
  isEmergency?: boolean;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I am NeuroHealth, your AI-powered Health Assistant. How can I help you today?',
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleReasoning = (id: string) => {
    setExpandedReasoning(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user1',
          message: userMessage.content,
          history: messages.map(m => ({ role: m.role, content: m.content }))
        })
      });

      const data = await response.json();

      const isEmergency = data.response.includes("EMERGENCY");

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        reasoning: data.reasoning_steps,
        isEmergency
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: 'There was an error connecting to the NeuroHealth engine. Please try again later.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex overflow-hidden">
      {/* Sidebar: User Profile & Context */}
      <aside className="w-80 border-r border-slate-200/60 bg-white/40 backdrop-blur-xl hidden md:flex flex-col p-6 m-4 rounded-3xl shadow-sm">
        <div className="flex items-center gap-3 mb-10 text-primary-700">
          <HeartPulse size={28} className="text-primary-500" />
          <h1 className="text-xl font-semibold tracking-tight">NeuroHealth</h1>
        </div>

        <div className="space-y-8 flex-1">
          <section>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <User size={14} /> Profile
            </h2>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-slate-100 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-slate-500">Age</span>
                <span className="font-medium text-slate-900">32</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">Gender</span>
                <span className="font-medium text-slate-900">Unknown</span>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <Activity size={14} /> Medical Constraints
            </h2>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1.5 bg-rose-50 text-rose-700 text-xs font-medium rounded-full border border-rose-100">
                Mild Knee Pain
              </span>
            </div>
          </section>

          <section>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <HeartPulse size={14} /> Preferences
            </h2>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1.5 bg-emerald-50 text-emerald-700 text-xs font-medium rounded-full border border-emerald-100">
                Vegetarian
              </span>
              <span className="px-3 py-1.5 bg-blue-50 text-blue-700 text-xs font-medium rounded-full border border-blue-100">
                Lose Weight
              </span>
            </div>
          </section>
        </div>

        <div className="p-4 bg-primary-50 rounded-2xl text-xs text-primary-800 border border-primary-100 leading-relaxed">
          The NeuroHealth Reasoning Engine integrates your profile with global medical guidelines to provide safe, personalized advice.
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-screen max-w-4xl mx-auto p-4 md:p-6 lg:p-8">
        <div className="flex-1 glass-panel rounded-[2rem] overflow-hidden flex flex-col relative">

          {/* Header */}
          <header className="h-16 border-b border-slate-200/50 flex items-center px-6 bg-white/50 z-10">
            <h2 className="font-medium text-slate-800">Consultation Session</h2>
            <div className="ml-auto flex items-center gap-2 text-xs font-medium px-3 py-1 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-100">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              Engine Active
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex flex-col max-w-[85%] ${message.role === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'}`}
                >
                  {/* Reasoning Block for Assistant */}
                  {message.role === 'assistant' && message.reasoning && message.reasoning.length > 0 && (
                    <div className="mb-2 w-full max-w-md">
                      <button
                        onClick={() => toggleReasoning(message.id)}
                        className="flex items-center gap-2 text-xs font-medium text-indigo-600 mb-1 hover:text-indigo-800 transition-colors"
                      >
                        <BrainCircuit size={14} />
                        {expandedReasoning[message.id] ? 'Hide Thought Process' : 'View Thought Process'}
                        {expandedReasoning[message.id] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      </button>

                      <AnimatePresence>
                        {expandedReasoning[message.id] && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="reasoning-panel p-3 rounded-xl text-xs text-indigo-900 space-y-1.5 shadow-sm">
                              {message.reasoning.map((step, idx) => (
                                <div key={idx} className="flex gap-2">
                                  <span className="opacity-50 mt-0.5">•</span>
                                  <span className="leading-relaxed">{step}</span>
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  {/* Message Bubble */}
                  <div
                    className={`p-4 rounded-2xl shadow-sm leading-relaxed ${message.role === 'user'
                      ? 'bg-primary-600 text-white rounded-tr-sm'
                      : message.isEmergency
                        ? 'bg-rose-50 border-2 border-rose-500 text-rose-900 rounded-tl-sm shadow-rose-100'
                        : message.role === 'system'
                          ? 'bg-amber-50 text-amber-900 border border-amber-200 rounded-tl-sm text-sm'
                          : 'bg-white text-slate-800 border border-slate-100 rounded-tl-sm prose prose-slate max-w-none prose-p:mb-2 prose-headings:mb-3 prose-ul:mb-2'
                      }`}
                  >
                    {message.isEmergency && (
                      <div className="flex items-center gap-2 mb-2 text-rose-600 font-bold uppercase tracking-wider text-xs">
                        <AlertTriangle size={16} /> Critical Alert
                      </div>
                    )}
                    {message.role === 'assistant' ? (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    ) : (
                      message.content
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-start gap-3 max-w-[85%]"
              >
                <div className="bg-white border border-slate-100 p-4 rounded-2xl rounded-tl-sm shadow-sm flex gap-1.5 items-center">
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 bg-white/50 border-t border-slate-200/50 backdrop-blur-md">
            <form onSubmit={handleSend} className="relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Describe your symptoms or ask a health question..."
                className="w-full pl-6 pr-14 py-4 rounded-full glass-input text-slate-800 placeholder:text-slate-400 shadow-sm text-[15px]"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="absolute right-2 p-2.5 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-300 disabled:text-slate-500 text-white rounded-full transition-colors shadow-md disabled:shadow-none"
              >
                <Send size={18} className="translate-x-[1px]" />
              </button>
            </form>
            <div className="text-center mt-3 text-[10px] text-slate-400 font-medium">
              NeuroHealth is an AI prototype. It does not provide medical diagnoses. In an emergency, dial 911.
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
