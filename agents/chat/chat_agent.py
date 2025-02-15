import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

from agent import Agent
from chat.chat_history_manager import ChatHistoryManager

load_dotenv()

AGENT_CHAT = os.environ['AGENT_CHAT']


class ChatAgent(Agent):

    prompt: str = """
    You are a sales person working on an e-commerce platform. 
    You are a smart, talkative, and funny person. You can answer users' questions in a humorous and witty way based on some information such as:
    
    Chat history:
    {chat_history}
    User: {user_message}
    Reply:

    """

    model_name = AGENT_CHAT

    def __init__(self):
        super().__init__(agent_name="ChatAgent")
        self.chat_history_manager = ChatHistoryManager()

    def process_message(self, session_id: str, user_message: str):
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True)
            stored_messages = self.chat_history_manager.get_history(
                session_id=session_id)
            for msg in stored_messages:
                if msg["sender"] == "user":
                    memory.chat_memory.add_user_message(msg["message"])
                elif msg["sender"] == "agent":
                    memory.chat_memory.add_ai_message(msg["message"])

            chain = ConversationChain(
                llm=self.llm,
                memory=memory,
                prompt=self.prompt_template,
                input_key="user_message",
                verbose=True
            )

            result = chain.invoke({"user_message": user_message})
            print(result)
            response = result.get("response", "")

            self.chat_history_manager.append_message(session_id=session_id,
                                                     sender="user", message=user_message)
            self.chat_history_manager.append_message(session_id=session_id,
                                                     sender="agent", message=response)
            return response
        except Exception as e:
            raise e
