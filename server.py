from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import services


app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
async def health():
    return {"message": "Server is running"}

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    text = services.extract_text_from_file(file)
    chunks = services.chunk_text(text)
    embeddings = services.get_embeddings(chunks)
    return {"message": "File uploaded successfully", "chunks": chunks, "embeddings": embeddings}



@app.post("/api/v1/query")
async def make_query(request: QueryRequest):
    return {"query": request.query, "top_k": request.top_k}