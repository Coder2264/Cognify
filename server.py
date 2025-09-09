from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import llm
import services
import uuid
from chromaDB import collection, reset_collection

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
async def health():
    return {"message": "Server is running"}

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    # Extract text
    text = services.extract_text_from_file(file)
    # Chunk text
    chunks = services.chunk_text(text)
    # Get embeddings
    embeddings = services.get_embeddings(chunks)

    # Generate a unique file_id for this file upload
    file_id = str(uuid.uuid4())

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "file_id": file_id,          # unique file_id
            "file_name": file.filename,  # original filename (optional, for display)
            "chunk_index": i
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "message": "File uploaded successfully",
        "file_name": file.filename,
        "file_id": file_id,
        "total_chunks": len(chunks)
    }



@app.post("/api/v1/query")
async def make_query(request: QueryRequest):
    # Get embedding for the query
    query_embedding = services.get_embeddings([request.query])[0]

    #Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k,
    )

    #Call LLM with context
    response= llm.call_llm(request.query, results["documents"][0])

    #Save query and response to Redis
    services.save_to_redis(request.query, "user")
    services.save_to_redis(response, "assistant")

    return {"query": request.query, "response": response}

@app.get("/api/v1/files")
async def list_files():
    files = collection.get(include=["metadatas"])
    all_metadatas = files["metadatas"] if files["metadatas"] else []
    unique_files = {}
    for metadata in all_metadatas:
        if metadata and "file_id" in metadata:
            unique_files[metadata["file_id"]] = metadata.get("file_name", "unknown")

    return {"files": [{"file_id": fid, "file_name": fname} for fid, fname in unique_files.items()]}


@app.delete("/api/v1/files/{file_id}")
async def delete_file(file_id: str):
    files = collection.get(include=["metadatas"])
    all_ids = files["ids"] if files["ids"] else []
    all_metadatas = files["metadatas"] if files["metadatas"] else []
    
    ids_to_delete = [
        idx for idx, metadata in zip(all_ids, all_metadatas)
        if metadata and metadata.get("file_id") == file_id
    ]

    if not ids_to_delete:
        return {"message": f"No chunks found for file_id {file_id}"}

    collection.delete(ids=ids_to_delete)

    return {"message": f"File with file_id {file_id} deleted successfully"}


@app.get("/api/v1/new-session")
async def new_session():
    services.clear_redis()
    global collection
    collection = reset_collection()
    return {"message": "New session started, all data cleared"}

@app.get("/api/v1/all-chats")
async def get_all_chats():
    history = services.get_chat_history()
    return {"chats": history}