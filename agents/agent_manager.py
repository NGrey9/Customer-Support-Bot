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


load_dotenv()
MODEL_NAME = os.environ['AGENT_MANAGER']
LOGS_DIR = os.environ['LOGS_DIR']

# Product Description
PRODUCT_DESCRIPTION_COMMANDS = os.environ['PRODUCT_DESCRIPTION_COMMANDS']
with open(PRODUCT_DESCRIPTION_COMMANDS, 'r') as f:
    product_description_commands = json.load(f)


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
        # self.registry = {
        #     "product_description": {
        #         "agent": ProductDescriptionAgent(),
        #         "commands": product_description_commands["commands"]
        #     },
        #     "chat": {
        #         "agent": ChatAgent()
        #     }
        # }
        self.product_description_agent = ProductDescriptionAgent()
        self.product_description_commands = product_description_commands["commands"]
        self.chat_agent = ChatAgent()
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
                    # "Return the is_chat's value is 'yes' if the other person does not require any specific action, else 'no'.",
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
        try:
            data_object = self.chain.invoke({"user_message": user_message})
            return data_object, user_message
        except Exception as e:
            return ("chat", user_message)

    def dispatch(self, user_message):
        def handle_chat():
            return self.chat_agent.process_message(session_id="0001",
                                                   user_message=user_message)

        try:
            requirements, user_message = self.extract_requirement(
                user_message=user_message)
            if requirements == "chat":
                print(handle_chat())
            else:
                for requirement in requirements:
                    print(type(requirement))
                    print(requirement)
                    print(requirement[0])
                    print(requirement[1])
                    print(type(requirement[1]))
                    # print(requirement[1][0].required_action)
                    # print(requirement[1][0]['required_action'])
                    if requirement[1][0].required_action:
                        required_action = requirement[1][0].required_action.lower(
                        )
                        if required_action in self.product_description_commands:
                            response = self.product_description_agent.describe_product(
                                requirement[1][0].product_name)
                            print(response)
                        else:
                            print(handle_chat())

                    else:
                        print(handle_chat())
        except Exception as e:
            raise e


if __name__ == "__main__":
    agent_manager = AgentManager()
    user_message = "I'm good. Haha!! Can you repeat my name?"
    agent_manager.dispatch(user_message=user_message)
    # print(agent_manager.extract_requirement(user_message=user_message))
