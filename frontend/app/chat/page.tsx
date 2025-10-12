'use client'
import React, { useState } from "react";

function Chat() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "👋 Hi there! I'm SPARKY, your friendly Grade 3 peer tutor! What is your name?" }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input }),
      });

      const data = await response.json();

      if (response.ok && data.output) {
        typeBotMessage(data.output, newMessages);
      } else {
        const errorText = data.error
          ? `⚠️ Error: ${data.error}`
          : "No response received from SPARKY.";
        typeBotMessage(errorText, newMessages);
      }

    } catch (error) {
      console.error(error);
      typeBotMessage("Oops! Something went wrong.", newMessages);
    }
  };

  const typeBotMessage = (
    fullText: string = "",
    prevMessages: { role: string; text: string }[]
  ) => {
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index <= fullText.length) {
        const typingText = fullText.slice(0, index);
        setMessages([...prevMessages, { role: "bot", text: typingText }]);
        index++;
      } else {
        clearInterval(typingInterval);
        setIsTyping(false);
      }
    }, 20);
  };

  return (
    <main
      style={{
        maxWidth: "700px",
        margin: "40px auto",
        padding: "20px",
        borderRadius: "20px",
        background: "linear-gradient(135deg, #FFDDE1 0%, #BDE0FE 100%)", // softer gradient
        fontFamily: "'Comic Sans MS', 'Poppins', sans-serif",
        boxShadow: "0 4px 15px rgba(0,0,0,0.2)"
      }}
    >
      <h2
        style={{
          textAlign: "center",
          color: "#FF6B6B", // cheerful red
          fontSize: "28px",
          marginBottom: "10px",
          textShadow: "1px 1px #FFF8DC",
        }}
      >
        🧠 SPARKY
      </h2>

      <div
        style={{
          backgroundColor: "#FFF9F0", // light cream
          borderRadius: "15px",
          padding: "15px",
          height: "500px",
          overflowY: "auto",
          marginBottom: "15px",
          boxShadow: "inset 0 0 10px rgba(0,0,0,0.05)",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
              margin: "8px 0",
            }}
          >
            <div
              style={{
                backgroundColor: msg.role === "user" ? "#FFDAC1" : "#B5EAD7", // user = peach, bot = mint
                color: "#333",
                padding: "10px 14px",
                borderRadius: "18px",
                maxWidth: "75%",
                fontSize: "16px",
                lineHeight: "1.4",
                whiteSpace: "pre-wrap",
                boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
              }}
            >
              <strong>{msg.role === "user" ? "👦 You" : "🤖 SPARKY"}:</strong> {msg.text}
            </div>
          </div>
        ))}
        {isTyping && <p style={{ color: "#999" }}><em>SPARKY is thinking...</em></p>}
      </div>

      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message here..."
          style={{
            flex: 1,
            padding: "10px 15px",
            borderRadius: "20px",
            border: "2px solid #FFB3B3", // soft pink border
            outline: "none",
            fontSize: "16px",
          }}
        />
        <button
          onClick={sendMessage}
          style={{
            backgroundColor: "#FFB3B3", // soft pink button
            color: "#fff",
            border: "none",
            borderRadius: "20px",
            padding: "10px 20px",
            fontSize: "16px",
            cursor: "pointer",
            transition: "0.2s",
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#FF8C8C")}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#FFB3B3")}
        >
          🚀 Send
        </button>
      </div>
    </main>

  );
}

export default Chat;
