import os
from dotenv import load_dotenv

from fastapi import FastAPI
import uvicorn

from agents.product_description.database_api import DatabaseAPI, ProductUpdateRequest
from product_description_agent import ProductDescriptionAgent

app = FastAPI()
database_api = DatabaseAPI()
product_description_agent = ProductDescriptionAgent()


@app.post("/create_vector_db")
async def create_vector_db():
    return database_api.create_vector_db()


@app.post("/update_vector_db")
async def update_vector_db(payload: ProductUpdateRequest):
    return database_api.update_vector_db(payload=payload)


@app.get("/query")
async def query_vector_db(query: dict):
    name = query['name']
    response = product_description_agent.describe_product(name)
    return {"description": response}

if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8088)
