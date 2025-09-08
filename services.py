import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile
import requests
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL")

def extract_text_from_file(file: UploadFile) -> str:
    text=""
    if file.content_type == "application/pdf":
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or " "
    else:
        text = file.file.read().decode("utf-8")
    return text


def chunk_text(text: str, chunk_size: int =1000, overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Get embeddings for text chunks using Voyage API.
    Uses batching for efficiency.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        data = {"input": batch}
        r = requests.post(EMBEDDING_SERVER_URL, json=data)

        if r.status_code != 200:
            raise Exception(f"Embedding API error: {r.text}")

        batch_embeddings = [item["embedding"] for item in r.json()["data"]]
        embeddings.extend(batch_embeddings)

    return embeddings

