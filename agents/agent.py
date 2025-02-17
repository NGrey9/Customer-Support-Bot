import os
import logging
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from logs import SetupLogging


from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = os.environ['LOGS_DIR']


class Agent:

    prompt: str = """
    <prompt>
    """

    model_name: str

    def __init__(self, agent_name: str = None):
        self.agent_name = agent_name
        self.setup_logging = SetupLogging(self, self.agent_name)
        self.setup_logging.setup_logging()
        self.create_prompt_template()
        self.llm = OllamaLLM(model=self.model_name)
        self.embedding = OllamaEmbeddings(model=self.model_name)

    def create_prompt_template(self):
        self.prompt_template = ChatPromptTemplate.from_template(self.prompt)


    def postprocess_message(self, agent_message: str):
        try:
            agent_message = agent_message.split('</think>')[-1]
            return agent_message
        except:
            return agent_message