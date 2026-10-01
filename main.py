
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model_embeddings import load_model, client_creation, search_object


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    client_creation()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory="clothing_images"), name="images")


@app.get("/")
def home():
    file_path = "static/index.html"
    return FileResponse(file_path)


@app.get("/search-query")
def search_query(q: str):
    results = search_object(q)
    return {"result": results}
