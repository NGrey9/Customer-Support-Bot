import os
import sys
import json
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from pydantic import BaseModel
import faiss
from sqlalchemy import URL
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

DB_DRIVER = os.environ['DB_DRIVER']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
DB_PORT = os.environ['DB_PORT']
DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_URL = URL.create(drivername=DB_DRIVER,
                    username=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT,
                    database=DB_NAME)

INDEX_PATH = os.environ['PRODUCT_DESCRIPTION_FAISS_INDEX']
VECTOR_DIM = os.environ['VECTOR_DIM']
AGENT_PRODUCT_DESCRIPTION = os.environ['AGENT_PRODUCT_DESCRIPTION']


class ProductUpdateRequest(BaseModel):
    id: int
    name: str
    description: str


class DatabaseAPI:
    def __init__(self, database_url: str = DB_URL,
                 index_path: str = INDEX_PATH,
                 embedding_model: str = AGENT_PRODUCT_DESCRIPTION):

        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)

        self.index_path = index_path
        self.text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200)

        self.embedding_model = OllamaEmbeddings(model=embedding_model)
        if not os.path.exists(index_path):
            self.create_vector_db()
        self.vector_store = FAISS.load_local(
            index_path, self.embedding_model, allow_dangerous_deserialization=True)

    def create_vector_db(self):
        documents = []
        with self.engine.connect() as conn:
            results = conn.execute(
                text(f"SELECT id, name, description FROM products")
            )
            data = [{"id": row[0], "name": row[1], "description": row[2]}
                    for row in results]
            for row in data:
                doc = Document(
                    page_content=f"Product: {row['name']}\nDescription: {row['description']}"
                )
                documents.append(doc)

            splits = self.text_splitter.split_documents(documents=documents)
            self.vector_store = FAISS.from_documents(documents=splits,
                                                     embedding=self.embedding_model)
            self.vector_store.save_local(self.index_path)

    def update_vector_db(self, payload: ProductUpdateRequest):
        product_id = payload.id
        product_name = payload.name
        product_description = payload.description
        new_doc = Document(
            page_content=f"Product: {product_name}\nDescription: {product_description}",
            metadata={"id": product_id, "product_name": product_name}
        )

        new_splits = self.text_splitter.split_documents([new_doc])

        self.vector_store.add_documents(new_splits)

        self.vector_store.save_local(self.index_path)

        return {"message": "Vector database updated successfully", "product_id": product_id}

    def query_product_name_sql(self, product_name: str):
        with self.engine.connect() as conn:
            try:
                results = conn.execute(
                    text(
                        f"SELECT id, name, description FROM products WHERE name = \"{product_name}\"".strip("'"))
                )
                data = [{"id": row[0], "name": row[1], "description": row[2]}
                        for row in results]
                return f"Product: {product_name}\nDescription: {data[0]['description']}"
            except Exception as e:
                print(e)
                return f"Product: {product_name} does not exist in the database."

    