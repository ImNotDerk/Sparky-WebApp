'use client'
import React, { useState } from "react";

function Chat() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi! I am SPARKY, your friendly Grade 3 peer tutor." }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
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
            ? `Error: ${data.error}`
            : "No response received from the backend.";
          typeBotMessage(errorText, newMessages);
      }

    } catch (error) {
      console.error(error);
      typeBotMessage("Error generating response.", newMessages);
    }
  };

  // Typing animation
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
    }, 15);
  };


  return (
    <main style={{ maxWidth: "600px", margin: "auto", padding: "20px" }}>
      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px",
          height: "400px",
          overflowY: "auto",
          marginBottom: "10px",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              textAlign: msg.role === "user" ? "right" : "left",
              margin: "5px 0",
              whiteSpace: "pre-wrap",
            }}
          >
            <strong>{msg.role === "user" ? "You" : "SPARKY"}:</strong> {msg.text}
          </div>
        ))}
        {isTyping && <p><em>Bot is typing...</em></p>}
      </div>

      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type your message..."
          style={{ flex: 1, padding: "8px" }}
        />
        <button onClick={sendMessage} style={{ padding: "8px 16px" }}>
          Send
        </button>
      </div>
    </main>
  );
}

export default Chat;
