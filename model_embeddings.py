"""
Multimodal embedding model for image search engine

reference = https://www.youtube.com/watch?v=6chRtu94NTY

The clothing images dataset is available on https://huggingface.co/datasets/ashraq/fashion-product-images-small

"""
import os
import random
import uuid
import warnings

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "clothing_images")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
# sorted so the row order of the cached embeddings always matches the image order
images = [os.path.join(IMAGES_DIR, f) for f in sorted(os.listdir(IMAGES_DIR))]

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

#initialize embeddings (computed once, then cached on disk)
def load_embeddings():
    global embeddings
    if embeddings is not None:
        warnings.warn("Embeddings already loaded", UserWarning)
        return

    if os.path.exists(EMBEDDINGS_FILE):
        cached = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        # the cache is only valid if it was built from the same set of images
        if cached.shape[0] == len(images):
            embeddings = cached
            return
        warnings.warn("Embeddings cache is stale, recomputing", UserWarning)

    warnings.warn("Embeddings creation started", UserWarning)
    # If you have a stronger cpu try using a higher batch, for my rtx 3050 ti 8 batches were the limit
    embeddings = model.encode(images, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    warnings.warn("Embeddings finisehd")

#initializes qdrant client and collections
def client_creation():
    global client
    if client is None:
        client = QdrantClient(host="localhost", port=6333)

    # Qdrant already persists the vectors, so only build the collection once
    if not client.collection_exists("images"):
        load_embeddings()

        client.create_collection(
            collection_name="images",
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )
        client.upload_points(
            collection_name="images",
            points=[
                PointStruct(id=uuid.uuid4(), vector=embeddings[i], payload={"path": os.path.basename(images[i])})
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





