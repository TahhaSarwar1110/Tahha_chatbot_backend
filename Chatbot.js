import { useState } from "react";
import axios from "axios";

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");

    try {
      const response = await axios.post(
        "https://tahha-chatbot-backend.onrender.com/chat",
        {
          message: input,
        }
      );
      const botMessage = { sender: "bot", text: response.data.response };
      setMessages([...messages, userMessage, botMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
    }
  };

  return (
    <div
      className="chat-container"
      style={{ maxWidth: "600px", margin: "auto", padding: "20px" }}
    >
      <div
        className="chat-box"
        style={{
          height: "400px",
          overflowY: "auto",
          border: "1px solid #ccc",
          padding: "10px",
        }}
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{ textAlign: msg.sender === "user" ? "right" : "left" }}
          >
            <p
              style={{
                display: "inline-block",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: msg.sender === "user" ? "#007bff" : "#e5e5ea",
                color: msg.sender === "user" ? "#fff" : "#000",
              }}
            >
              {msg.text}
            </p>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", marginTop: "10px" }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          style={{
            flexGrow: 1,
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "5px",
          }}
        />
        <button
          onClick={sendMessage}
          style={{
            marginLeft: "10px",
            padding: "10px 20px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
