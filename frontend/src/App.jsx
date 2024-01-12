import { useState } from "react";
import axios from "axios";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator,
} from "@chatscope/chat-ui-kit-react";

function App() {
  const [messages, setMessages] = useState([
    {
      message: "MTS AI at your service.",
      sender: "MTSAI",
    },
  ]);
  const [prompt, setPrompt] = useState("");
  const [typing, setTyping] = useState(false);

  const sendData = async (messagePrompt) => {
    const dataObj = {
      data: messagePrompt,
    };
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/createResponse",
        dataObj
      );
      // const incomingMessage  = [...messages, {message: response.data, sender: "MTSAI"}]
      // setMessages(incomingMessage)
      setTyping(false);
    } catch (err) {
      console.log(err);
    }
  };

  const handleSend = async (message) => {
    const addedMessage = {
      message: message,
      sender: "user",
      direction: "outgoing",
    };
    const appendingMessages = [...messages, addedMessage];
    setMessages(appendingMessages);
   
    setTyping(true);

    const dataObj = {
      data: message,
    };
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/createResponse",
        dataObj
      );
      const recievedMessages = { message: response.data, sender: "MTSAI" };
      setMessages(prevMessages => [...prevMessages, recievedMessages]);
      setTyping(false);
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="h-screen w-full bg-slate-500 flex flex-col items-center justify-center space-y-2">
      <div className="h-[80%] w-[80%]">
        <MainContainer>
          <ChatContainer>
            <MessageList
              typingIndicator={
                typing ? <TypingIndicator content="MTSAI is typing" /> : null
              }
            >
              {messages.map((message, i) => {
                return <Message key={i} model={message} />;
              })}
            </MessageList>
            <MessageInput placeholder="type message here" onSend={handleSend} />
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  );
}

export default App;
