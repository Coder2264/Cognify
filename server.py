from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import llm
import services
import uuid
from chromaDB import collection

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
async def health():
    return {"message": "Server is running"}

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    #Extract text
    text = services.extract_text_from_file(file)
    #Chunk text
    chunks = services.chunk_text(text)
    #Get embeddings
    embeddings = services.get_embeddings(chunks)

    ids =[]
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "file_name": file.filename,
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

    return {"query": request.query, "response": response}