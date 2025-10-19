'use client'
import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

// --- TYPE DEFINITIONS ---
interface Message {
  role: 'user' | 'model';
  text: string;
}

// --- CONSTANTS ---
// BASE API URL
const API_BASE_URL = "http://localhost:8000";

// --- COMPONENT ---
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "model", text: "👋 Hi there! I'm SPARKY, your friendly Grade 3 peer tutor! What is your name?" }
  ]);
  const [input, setInput] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [isDarkMode]);

  useEffect(() => {
    const startNewChat = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/start_chat`, {
          method: "GET"
        });
        const data = await response.json();
        if (response.ok && data.session_id) {
          setSessionId(data.session_id);
          console.log("New session started:", data.session_id);
        } else {
          console.error("Failed to start a new chat session.");
        }
      } catch (error) {
        console.error("Error starting new chat session:", error);
      }
    };

    startNewChat();
  }, []); // Empty array ensures this runs only once on mount

  const downloadConversation = () => {
    const dataStr = JSON.stringify(messages, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sparky_conversation.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) {
      if (!sessionId) {
        console.error("No session ID. Cannot send message.");
      }
      return;
    }

    const newMessages: Message[] = [...messages, { role: "user", text: input }];
    setMessages(newMessages);
    const currentInput = input;
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/send_message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: currentInput,
          session_id: sessionId
        }),
      });

      const data = await response.json();
      if (response.ok && data.output) {
        typeBotMessage(data.output, newMessages);
      } else {
        const errorText = data.detail ? `⚠️ Error: ${data.detail}` : "No response received from SPARKY.";
        typeBotMessage(errorText, newMessages);
      }
    } catch (error) {
      console.error(error);
      typeBotMessage("Oops! Something went wrong.", newMessages);
    }
  };

  const typeBotMessage = (fullText: string = "", prevMessages: Message[]) => {
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index <= fullText.length) {
        const typingText = fullText.slice(0, index);
        setMessages([...prevMessages, { role: "model", text: typingText }]);
        index++;
      } else {
        clearInterval(typingInterval);
        setIsTyping(false);
      }
    }, 15);
  };

  return (
    <main className="chat-container">
      <div className="chat-header">
        <h2>🧠 SPARKY</h2>
        <button className="theme-toggle-btn" onClick={() => setIsDarkMode(!isDarkMode)}>
          {isDarkMode ? "☀️ Light" : "🌙 Dark"}
        </button>
      </div>

      <div className="message-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.role === 'user' ? 'user' : 'model'}`}>
            <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'model-bubble'}`}>
              <ReactMarkdown>
                {`${msg.role === "user" ? "👦 **You**" : "🤖 **SPARKY**"}: ${msg.text}`}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        {isTyping && <p className="typing-indicator"><em>SPARKY is thinking...</em></p>}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message here..."
        />
        <button onClick={sendMessage}>🚀 Send</button>
        <button onClick={downloadConversation}>💾 Download</button>
      </div>
    </main>
  );
}