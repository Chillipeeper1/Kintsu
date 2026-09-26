import os
from fastapi import FastAPI
import torch
from sentence_transformers import SentenceTransformer
from model_embeddings import load_model, load_embeddings, client_creation, search_object

app = FastAPI()

@app.get("/")
def home():
    # first time calling
    load_model()
    load_embeddings()
    client_creation()
    return {"message":"Services Running"}

@app.get("/search-query")
def search_query():
    results = search_object("black pants")
    return {"result":results}




