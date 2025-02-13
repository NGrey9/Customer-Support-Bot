import os
import logging
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from agent import Agent


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
    is_chat: Optional[str] = Field(
        default=None, description="Return True if user does not require any specific action")


class Data(BaseModel):
    requirements: List[Requirement] = Field(
        default=None, description="List of user requirements in the message"
    )


class AgentManager(Agent):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__(logs_dir=LOGS_DIR, agent_name='AgentManager')
        self.llm = OllamaFunctions(model=model_name)

    def create_chain(self):
        self.logger.debug("Create processing chain")
        try:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a salesperson, you are tactful and smart in understanding customer requirements. Your role is to extract customer request information based on the message provided by the customer."
                        "Based on the other person's messages, you can analyze what the other person wants you to do or what information you need him/her/them to provide."
                        "Return the required_action's value the verb or verb phrase mentioned by customer in the sentence that represents the action the user wants you to do."
                        "Return additional information be used for giving more action informations for additional_action_info's value else null"
                        "Return product_name's value if you know else null"
                        "Return the is_chat's value is 'yes' if the other person does not require any specific action, else 'no'.",
                    ),
                    ("human", "{user_message}"),
                ]
            )

            self.logger.debug("Created prompt template")

            # Tạo chain mới với cú pháp đúng
            structured_llm = self.llm.with_structured_output(Data)

            chain = prompt_template | structured_llm
            self.logger.debug("Successfully created processing chain")
            return chain

        except Exception as e:
            self.logger.error(f"Error creating chain: {str(e)}")
            raise

    def extract_requirement(self, user_message: str):
        self.logger.info(f"Extracting user requirements: {user_message}")
        self.query_logger.info(f"User query: {user_message}")
        try:
            chain = self.create_chain()
            self.logger.debug("Starting chain execution")

            data_object = chain.invoke({"user_message": user_message})
            self.logger.debug("Successfully extracted user requirements")
            self.logger.debug(
                f"Extracted actions: {str(([req.required_action for req in data_object.requirements]))}")
            self.query_logger.info(
                f"Extracted actions: {str(([req.required_action for req in data_object.requirements]))}")
            return data_object
        except Exception as e:
            self.logger.error(f"Error extracting requirement: {e}")
            pass


if __name__ == "__main__":
    agent_manager = AgentManager()
    user_message = "How many 'Halu' are left?"
    print(agent_manager.extract_requirement(user_message))
    # print(agent_manager.extract_requirement(user_message=user_message))
