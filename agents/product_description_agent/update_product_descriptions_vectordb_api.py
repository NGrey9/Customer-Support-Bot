import os
import json
from dotenv import load_dotenv

import numpy as np
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import faiss
from langchain_community.embeddings import OllamaEmbeddings
load_dotenv()

DATABASE_URL = os.environ['DATABASE_URL']
MODEL_NAME = os.environ['AGENT_MANAGER']
VECTOR_DIM = 512
INDEX_PATH = os.environ['']


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()




if __name__ == "__main__":
    pass
