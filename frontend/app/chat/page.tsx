'use client'
import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

// --- TYPE DEFINITIONS ---
interface Message {
    role: 'user' | 'model';
    text: string;
    choices?: string[];
    selectedChoice?: string; // Stores the clicked topic
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
    const messageAreaRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

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
    }, []);

    useEffect(() => {
        if (messageAreaRef.current) {
            const messageArea = messageAreaRef.current;
            // Scroll to the very bottom
            messageArea.scrollTop = messageArea.scrollHeight;
        }
    }, [messages, isTyping]); // Triggers every time messages or typing status changes

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

    const sendMessage = async (messageText?: string) => {
        const textToSend = messageText !== undefined ? messageText : input.trim();

        if (!textToSend || !sessionId) {
            if (!sessionId) {
                console.error("No sessionID. Cannot send message.")
            }
            return
        }

        const newMessages: Message[] = [...messages, { role: "user", text: textToSend }];
        setMessages(newMessages);
        if (messageText === undefined) {
            setInput("");
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        }

        setIsTyping(true);

        try {
            const response = await fetch(`${API_BASE_URL}/send_message`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: textToSend,
                    session_id: sessionId
                }),
            });

            const data = await response.json();
            if (response.ok && data.output) {
                const botMessage: Message = {
                    role: 'model',
                    text: data.output,
                    choices: data.choices || []
                };
                typeBotMessage(botMessage, newMessages);
            } else {
                const errorText = data.detail ? `⚠️ Error: ${data.detail}` : "No response received from SPARKY.";
                typeBotMessage({ role: 'model', text: errorText }, newMessages);
            }
        } catch (error) {
            console.error(error);
            typeBotMessage({ role: 'model', text: "Oops! Something went wrong." }, newMessages);
        }
    };

    const typeBotMessage = (botMessage: Message, prevMessages: Message[]) => {
        let index = 0;
        const fullText = botMessage.text;

        const typingInterval = setInterval(() => {
            if (index <= fullText.length) {
                const typingText = fullText.slice(0, index);
                setMessages([...prevMessages, { role: "model", text: typingText, choices: [] }]);
                index++;
            } else {
                clearInterval(typingInterval);
                setIsTyping(false);
                setMessages([...prevMessages, botMessage]);
            }
        }, 15);
    };

    const handleChoiceClick = (choice: string, messageIndex: number) => {

        // Disable topics only for clicked message
        setMessages(prevMessages =>
            prevMessages.map((msg, idx) =>
                idx === messageIndex && msg.role === 'model'
                    ? { ...msg, selectedChoice: choice }
                    : msg
            )
        );

        sendMessage(choice);
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);

        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${e.target.scrollHeight}px`;
        }
    }

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        // Check if Enter is pressed *without* the Shift key
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            if (!isInputLocked) {
                sendMessage();
            }
        }
        // If Shift + Enter is pressed, it will just add a new line (default behavior)
    };

    // Get the last message, if it exists
    const lastMessage = messages[messages.length - 1];

    // Check if the topic selection is currently active
    const isChoiceSelectActive =
        lastMessage &&                   // 1. Is there a last message?
        lastMessage.role === 'model' &&  // 2. Is it from SPARKY?
        lastMessage.choices &&            // 3. Does it *have* topics?
        lastMessage.choices.length > 0 && // 4. Are there more than 0 topics?
        !lastMessage.selectedChoice;      // 5. Has a topic NOT been selected yet?

    // Combine both locking conditions
    const isInputLocked = isTyping || isChoiceSelectActive;

    let placeholderText = "Type your message here...";
    if (isTyping) {
        placeholderText = "SPARKY is typing...";
    } else if (isChoiceSelectActive) {
        placeholderText = "Please select an option above.";
    }

    return (
        <main className="chat-container">
            <div className="chat-header">
                <h2>🧠 SPARKY</h2>
                <button className="theme-toggle-btn" onClick={() => setIsDarkMode(!isDarkMode)}>
                    {isDarkMode ? "☀️ Light" : "🌙 Dark"}
                </button>
            </div>

            <div className="message-area" ref={messageAreaRef}>
                {messages.map((msg, idx) => {

                    const roleHeader = msg.role === "user" ? "👦 **You**" : "🤖 **SPARKY**";

                    return (
                        <div key={idx} className={`message-wrapper ${msg.role === 'user' ? 'user' : 'model'}`}>

                            <div className="message-content-wrapper">

                                <div className="message-role-header">
                                    <ReactMarkdown>{roleHeader}</ReactMarkdown>
                                </div>

                                <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'model-bubble'}`}>

                                    <ReactMarkdown>
                                        {msg.text}
                                    </ReactMarkdown>

                                    {msg.role === 'model' && msg.choices && msg.choices.length > 0 && (
                                        <div className="inline-button-container">
                                            {msg.choices.map((choice, choiceIdx) => {

                                                const isDisabled = isTyping || !!msg.selectedChoice || messages[messages.length - 1] !== msg;
                                                const isSelected = msg.selectedChoice === choice;

                                                return (
                                                    <button
                                                        key={choiceIdx}
                                                        className={`chat-choice-button ${isSelected ? 'active' : ''}`}
                                                        disabled={isDisabled}
                                                        onClick={() => handleChoiceClick(choice, idx)}
                                                    >
                                                        {choice}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div> {/* End of message-bubble */}
                            </div> {/* End of message-content-wrapper */}
                        </div> /* End of message-wrapper */
                    );
                })}
                {isTyping && <p className="typing-indicator"><em>SPARKY is thinking...</em></p>}
            </div>

            <div className="input-area">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholderText}
                    disabled={isInputLocked}
                    rows={1} // Start as 1 row
                />
                <button onClick={() => sendMessage()} disabled={isInputLocked}> {/* Lock the button */}
                    🚀 Send
                </button>
                <button onClick={downloadConversation}>💾 Download</button>
            </div>
        </main>
    );
}