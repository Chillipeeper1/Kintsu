"""
Multimodal embedding model for image search engine

reference = https://www.youtube.com/watch?v=6chRtu94NTY

The clothing images dataset is available on https://huggingface.co/datasets/ashraq/fashion-product-images-small

"""
import os
import random
import uuid
import warnings

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


device = "cuda" if torch.cuda.is_available() else "cpu"
images = [os.path.join("", f) for f in os.listdir("clothing_images")]

model = None
embeddings = None
client = None

# initialize model
def load_model():
    global model
    if model is None:
        model = SentenceTransformer(
            "jinaai/jina-clip-v2",
            trust_remote_code=True,
            truncate_dim=1024,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False}
        )
    else:
        warnings.warn("Model already loaded", UserWarning)

#initialize embeddings
def load_embeddings():
    global embeddings
    if embeddings is None:

        embeddings = [model.encode(img, normalize_embeddings=True, batch_size=32, device=device) for img in images]
    else:
        warnings.warn("Embeddings already loaded", UserWarning)

#initializes qdrant client and collections
def client_creation():
    global client
    if not os.path.exists("image_store") and client is None:

        client = QdrantClient(host="localhost", port=6333)

        client.create_collection(
            collection_name="images",
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )
        client.upload_points(
            collection_name="images",
            points=[
                PointStruct(id=uuid.uuid4(), vector=embeddings[i], payload={"path": images[i]})
                for i in range(len(images))
            ],
            batch_size=100
        )
    else:
        warnings.warn("Qdrant client already up", UserWarning)



def search_object(search_query):
    query_embeddings = model.encode(search_query, normalize_embeddings=True, device=device)
    results = client.query_points(collection_name="images", query=query_embeddings, limit=5).points
    return results





