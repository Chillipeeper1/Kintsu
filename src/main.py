"""
Multimodal embedding model for image search engine

reference = https://www.youtube.com/watch?v=6chRtu94NTY

The clothing images dataset is available on https://huggingface.co/datasets/ashraq/fashion-product-images-small

"""
import os
import random
import uuid

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
    truncate_dim=1024,
    device=device
)

if not os.path.exists("image_store"):
    client = QdrantClient(host="localhost", port=6333)

    images = [os.path.join("", f) for f in os.listdir("clothing_images")] #clothing images dataset directory

    # Here we are about to process 5k images, I hope the 32 batching will not destroy my pc
    # I know I could reduce the number of images to a much more tiny number but for experimental proposes I'm going to scan the whole thing

    embeddings = [model.encode(img, normalize_embeddings=True, batch_size=32, device=device) for img in images]

    client.create_collection(
        collection_name="images",
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE )
    )
    client.upload_points(
        collection_name="images",
        points=[
            PointStruct(id=uuid.uuid4(), vector=embeddings[i], payload={"path":images[i]})
            for i in range(len(images))
        ] ,
        batch_size=100
    )
else:
    client = QdrantClient(host="localhost", port=6333)
print(embeddings[0])
print("DONE ur pc survived")
search_query = input("Enter query: ")
query_embeddings = model.encode(search_query, normalize_embeddings=True,device=device)

results = client.query_points(collection_name="images", query=query_embeddings, limit=5).points

print([r.payload["path"] for r in results])





