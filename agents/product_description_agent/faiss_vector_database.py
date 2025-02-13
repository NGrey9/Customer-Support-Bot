import os
import sys
import json
from dotenv import load_dotenv

import numpy as np
from pydantic import BaseModel
import faiss
from sqlalchemy import URL
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_ollama import OllamaEmbeddings

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
PRODUCT_DESCRIPTION_JSON = os.environ['PRODUCT_DESCRIPTION_JSON']
AGENT_PRODUCT_DESCRIPTION = os.environ['AGENT_PRODUCT_DESCRIPTION']


class ProductUpdateRequest(BaseModel):
    id: int
    name: str
    description: str


class VectorDBAPI:
    def __init__(self, database_url: str = DB_URL,
                 index_path: str = INDEX_PATH,
                 vector_dim: int = VECTOR_DIM,
                 json_path: str = PRODUCT_DESCRIPTION_JSON,
                 embedding_model: str = AGENT_PRODUCT_DESCRIPTION):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)

        self.vector_dim = int(vector_dim)
        self.index_path = index_path
        self.json_path = json_path
        self.index = self._load_faiss_index()
        self.name_to_description = self._load_name_to_description()

        self.embedding_model = OllamaEmbeddings(model=embedding_model)

    def _load_faiss_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        return faiss.IndexFlatL2(self.vector_dim)

    def _load_name_to_description(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.name_to_description = json.load(f)
        else:
            self.name_to_description = {}

    def get_data_from_mysql(self):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, name, description FROM products"))
            data = [{"id": row[0], "name": row[1], "description": row[2]}
                    for row in result]
        return data

    def create_vector_db(self, batch_size: int = 10000):

        # data = self.get_data_from_mysql()
        with self.engine.connect() as conn:
            total_records = conn.execute(
                text("SELECT COUNT(*) FROM products")).scalar()

            offset = 0
            ids = []
            vectors = []
            name_to_description = {}

            while offset < total_records:
                result = conn.execute(
                    text(
                        f"SELECT id, name, description FROM products LIMIT {batch_size} OFFSET {offset}")
                )
                data = [{"id": row[0], "name": row[1], "description": row[2]}
                        for row in result]

                batch_vectors = []
                batch_ids = []

                for row in data:
                    product_text = f"{row['name']} - {row['description']}"
                    vector = self.embedding_model.embed_query(product_text)

                    batch_vectors.append(vector)
                    batch_ids.append(row['id'])
                    name_to_description[row['id']] = row['description']

                batch_vectors = np.array(batch_vectors, dtype=np.float32)
                assert batch_vectors.shape[
                    1] == self.vector_dim, f"Expected dimension {self.vector_dim}, got {batch_vectors.shape[1]}"
                self.index.add(batch_vectors)
                ids.extend(batch_ids)

                offset += batch_size

            faiss.write_index(self.index, self.index_path)

            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(name_to_description, f)

        return {"message": "Vector database created successfully", "total_vectors": len(ids)}

    def update_vector_db(self, payload: ProductUpdateRequest):
        product_id = payload.id
        product_name = payload.name
        product_description = payload.description
        product_text = f"{product_name} - {product_description}"

        vector = self.embedding_model.embed_query(product_text)
        vector_array = np.array(vector, dtype=np.float32)

        self.index.add_with_ids(
            vector_array, np.array([product_id], dtype=np.int64))
        faiss.write_index(self.index, self.index_path)

        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                name_to_description = json.load(f)
        else:
            name_to_description = {}

        name_to_description[str(product_id)] = product_description

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(name_to_description, f)

        return {"message": "Vector database updated successfully", "product_id": product_id}

    def query_vector_db(self, query: str, top_k: int = 1):

        if not os.path.exists(self.index_path):
            return {"error": "FAISS index is not created"}

        if not os.path.exists(self.json_path):
            return {"error": "Json file does not exist"}

        self.index = faiss.read_index(self.index_path)

        with open(self.json_path, 'r', encoding='utf-8') as f:
            name_to_description = json.load(f)

        query_vector = [self.embedding_model.embed_query(query)]
        query_array = np.array(query_vector, dtype=np.float32)
        distances, indices = self.index.search(query_array, top_k)

        results = []
        for idx in indices[0]:
            if idx in name_to_description:
                results.append(
                    {"product_id": idx, "description": name_to_description[str(idx)]})

        return {"query":  query, "results": results}
