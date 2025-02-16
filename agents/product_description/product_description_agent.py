import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


from agent import Agent
load_dotenv()

AGENT_PRODUCT_DESCRIPTION = os.environ['AGENT_PRODUCT_DESCRIPTION']


class ProductDescriptionAgent(Agent):

    prompt = """
        Based on the following product information, please provide a detailed description:

        Context: {context}
        Product query: {query}

        Describe the product with specific specifications and materials used to make the product. And from the specifications, analyze the advantages of the product to advise users to buy it.
        If the product is not in database, you can apologize to the customer and ask them to re-state their request and the exact product name.
        """
    model_name = AGENT_PRODUCT_DESCRIPTION

    def __init__(self, sql_database_api):
        super().__init__(agent_name="ProductDescriptionAgent")
        self.database_api = sql_database_api
        self.vector_store = self.database_api.vector_store
        self.create_chain()

    def create_chain(self):
        try:
            self.chain = RunnableParallel(
                query=RunnablePassthrough(),
                context=lambda x: self.database_api.query_product_name_sql(x)
            ) | self.prompt_template | self.llm | StrOutputParser()
        except Exception as e:
            raise e

    def describe_product(self, product_name: str) -> str:
        try:
            # if self.vector_store is None:
            #     raise ValueError(
            #         "Please load data first using load_data method")
            if self.chain is None:
                raise ValueError(
                    "Chain is not created")
            response = self.chain.invoke(product_name)
            return response
        except Exception as e:
            raise e
