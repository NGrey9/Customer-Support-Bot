import os
import logging
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
load_dotenv()


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ProductDescriptionAgent:
    def __init__(self, model_name: str = "llama3.1:8b", logs_dir: str = "logs"):

        self.setup_logging(logs_dir=logs_dir)
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self.products_df = None

    def setup_logging(self, logs_dir: str):
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        os.remove('logs/*')

        self.logger = logging.getLogger('ProductDescriptionAgent')
        self.logger.setLevel(logging.DEBUG)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        detailed_log = os.path.join(
            logs_dir, f'detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(detailed_log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CustomFormatter(fmt))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter(fmt))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.query_logger = logging.getLogger('UserQueries')
        self.query_logger.setLevel(logging.INFO)
        query_log = os.path.join(
            logs_dir, f'queries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        query_handler = logging.FileHandler(query_log)
        query_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s'))
        self.query_logger.addHandler(query_handler)

    def load_data(self, csv_path: str):

        self.logger.info(f"Loading data from {csv_path}")

        try:
            self.products_df = pd.read_csv(csv_path)
            self.logger.info(
                f"Successfully loaded {len(self.products_df)} products from {csv_path}")

            documents = []
            for i, row in self.products_df.iterrows():
                doc = Document(
                    page_content=f"Product: {row['product_name']}\nDescription: {row['product_description']}",
                    metadata={"product_name": row['product_name']}
                )
                documents.append(doc)
            self.logger.info(
                f"Successfully created {len(documents)} documents")

            tex_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200
            )

            splits = tex_splitter.split_documents(documents=documents)
            self.logger.debug(
                f"Successfully split documents into {len(splits)} chunks")

            self.vector_store = FAISS.from_documents(documents=splits,
                                                     embedding=self.embeddings)
            self.logger.info(f"Successfully created FAISS vector store")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def get_context(self, query: str) -> List[Document]:
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Please load data first.")

        self.logger.debug(f"Retrieving context for query: {query}")
        return self.vector_store.similarity_search(query)

    def create_chain(self):
        self.logger.debug("Creating processing chain")
        try:
            template = """Based on the following product information, please provide a detailed description:

            Context: {context}
            Product query: {query}

            Please describe the product in a natural, conversational way. Include key features and benefits.
            """

            prompt = ChatPromptTemplate.from_template(template)
            self.logger.debug("Created prompt template")

            def _get_context(query: str) -> str:
                docs = self.get_context(query)
                return "\n".join(doc.page_content for doc in docs)

            # Tạo chain mới với cú pháp đúng
            chain = RunnableParallel(
                query=RunnablePassthrough(),
                context=lambda x: _get_context(x)
            ) | prompt | self.llm | StrOutputParser()

            self.logger.debug("Successfully created processing chain")
            return chain

        except Exception as e:
            self.logger.error(f"Error creating chain: {str(e)}")
            raise

    def describe_product(self, product_name: str) -> str:
        self.logger.info(f"Describing product: {product_name}")
        self.query_logger.info(f"User query: {product_name}")

        try:
            if self.vector_store is None:
                raise ValueError(
                    "Please load data first using load_data method")

            chain = self.create_chain()
            self.logger.debug("Starting chain execution")

            response = chain.invoke(product_name)
            self.logger.debug("Successfully generated product description")
            self.logger.debug(f"Product description: {response}")

            self.query_logger.info(
                f"Response for '{product_name}': {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error describing product: {e}")
            raise


if __name__ == "__main__":
    try:
        agent = ProductDescriptionAgent(logs_dir="logs")
        agent.logger.info("Agent initialized")

        agent.load_data('data/new_product_names.csv')
        product_name = "BEHR Premium Textured DECKOVER"
        response = agent.describe_product(product_name=product_name)
    except Exception as e:
        print(e)
