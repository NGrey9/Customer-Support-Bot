import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


from faiss_vector_database import VectorDBAPI

load_dotenv()
MODEL_NAME = os.environ['AGENT_PRODUCT_DESCRIPTION']


class ProductDescriptionAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_db_api = VectorDBAPI()
        self.vector_store = self.vector_db_api.vector_store
        self.create_chain()

    def get_context(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(query=query)

    def create_chain(self):
        try:
            template = """Based on the following product information, please provide a detailed description:

            Context: {context}
            Product query: {query}

            Describe the product with specific specifications and materials used to make the product. And from the specifications, analyze the advantages of the product to advise users to buy it.
            """
            prompt = ChatPromptTemplate.from_template(template=template)

            def _get_context(query: str) -> str:
                docs = self.get_context(query=query)
                print("\n".join(doc.page_content for doc in docs))
                return "\n".join(doc.page_content for doc in docs)

            self.chain = RunnableParallel(
                query=RunnablePassthrough(),
                context=lambda x: _get_context(x)
            ) | prompt | self.llm | StrOutputParser()
        except Exception as e:
            raise e

    def describe_product(self, product_name: str) -> str:
        try:
            if self.vector_store is None:
                raise ValueError(
                    "Please load data first using load_data method")
            if self.chain is None:
                raise ValueError(
                    "Chain is not created")

            response = self.chain.invoke(product_name)
            return response
        except Exception as e:
            raise e
