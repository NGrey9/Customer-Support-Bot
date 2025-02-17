import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

from agent import Agent

load_dotenv()

AGENT_CHAT = os.environ['AGENT_CHAT']


class ChatAgent(Agent):

    prompt: str = """
    You are NhanBot, a sales person working on an e-commerce platform. 
    You are intelligent, professional, articulate and have a quite sense of humor. You can answer user questions professionally based on information such as:
    
    Chat history:
    {chat_history}
    User: {user_message}
    Reply:

    """

    model_name = AGENT_CHAT

    def __init__(self, chat_history_manager):
        super().__init__(agent_name="ChatAgent")
        self.chat_history_manager = chat_history_manager

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
            response = result.get("response", "")

            message = self.postprocess_message(response)

            return message
        except Exception as e:
            raise e
