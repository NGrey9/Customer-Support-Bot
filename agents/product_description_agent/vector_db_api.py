import os
from dotenv import load_dotenv

from fastapi import FastAPI
import uvicorn
from faiss_vector_database import VectorDBAPI

app = FastAPI()
vector_db_api = VectorDBAPI()


@app.post("/create_vector_db")
async def create_vector_db():
    return vector_db_api.create_vector_db()


@app.post("/update_vector_db")
async def update_vector_db():
    return vector_db_api.update_vector_db()


@app.get("/query")
async def query_vector_db(query: dict):
    name = query['name']
    return vector_db_api.query_vector_db(name)

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8088)
