import os
import logging
import json
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from product_description import ProductDescriptionAgent
from chat import ChatAgent
from chat_history_manager import ChatHistoryManager
from sql_database_api import DatabaseAPI
from utils import load_commands


load_dotenv()
MODEL_NAME = os.environ['AGENT_MANAGER']
LOGS_DIR = os.environ['LOGS_DIR']


class Requirement(BaseModel):
    required_action: Optional[str] = Field(
        default=None, description="The action that user want you to do")
    additional_action_info: Optional[str] = Field(
        default=None, description="The additional information for action")
    product_name: Optional[str] = Field(
        default=None, description="The product that user mentioned")


class Data(BaseModel):
    requirements: List[Requirement] = Field(
        default=None, description="List of user requirements in the message"
    )


class AgentManager:
    def __init__(self, model_name=MODEL_NAME):
        self.llm = OllamaFunctions(model=model_name)
        self.sql_database_api = DatabaseAPI()
        self.product_description_agent = ProductDescriptionAgent(
            self.sql_database_api)
        self.product_description_commands = load_commands(
            "product_description")
        self.chat_history_manager = ChatHistoryManager()
        self.chat_agent = ChatAgent(self.chat_history_manager)

        self.set_prompt_template()
        self.create_chain()

    def set_prompt_template(self):
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a salesperson, you are tactful and smart in understanding customer requirements. Your role is to extract customer request information based on the message provided by the customer."
                    "Based on the other person's messages, you can analyze what the other person wants you to do or what information you need him/her/them to provide."
                    "Return the required_action's value the verb or verb phrase mentioned by customer in the sentence that represents the action the user wants you to do."
                    "Return additional information be used for giving more action informations for additional_action_info's value else null"
                    "Return product_name's value if you know else null",
                ),
                ("human", "{user_message}"),
            ]
        )

    def create_chain(self):
        try:
            # Tạo chain mới với cú pháp đúng
            structured_llm = self.llm.with_structured_output(Data)

            self.chain = self.prompt_template | structured_llm
        except Exception as e:
            raise e

    def extract_requirement(self, user_message: str):

        list_requirements = []
        try:
            data_object = self.chain.invoke({"user_message": user_message})
            for requirement in data_object:
                list_requirements.append([requirement[1][0].required_action.lower(),
                                         requirement[1][0].additional_action_info,
                                         requirement[1][0].product_name])
        except Exception as e:
            print(e)
            list_requirements.append(["chat", "", ""])

        return (list_requirements, user_message)

    def update_chat_history(self, session_id: str,
                            user_id: str,
                            user_message: str,
                            agent_message: str):
        self.chat_history_manager.append_message(session_id=session_id,
                                                 user_id=user_id,
                                                 sender="user",
                                                 message=user_message)
        self.chat_history_manager.append_message(session_id=session_id,
                                                 user_id=user_id,
                                                 sender="agent",
                                                 message=agent_message)

    def dispatch(self, session_id: str, user_id: str, user_message: str):

        try:
            list_requirements, user_message = self.extract_requirement(
                user_message=user_message)
            print(list_requirements)

            for requirement in list_requirements:
                required_action = requirement[0]
                additional_action_info = requirement[1]
                product_name = requirement[2]

                if required_action in self.product_description_commands and product_name not in ["null", "", None]:
                    agent_message = self.product_description_agent.describe_product(
                        product_name=product_name)
                else:
                    agent_message = self.chat_agent.process_message(session_id=session_id,
                                                                    user_message=user_message)

                self.update_chat_history(session_id=session_id,
                                         user_id=user_id,
                                         user_message=user_message,
                                         agent_message=agent_message)
            print(agent_message)
        except Exception as e:
            raise e


if __name__ == "__main__":
    agent_manager = AgentManager()
    user_message = "Ok, Thank you NhanBot. So do you remember my name?"
    agent_manager.dispatch(session_id="00002",
                           user_id="0001",
                           user_message=user_message)
