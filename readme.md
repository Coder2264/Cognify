# Cognify – RAG-Based AI Chatbot

Cognify is an advanced AI-powered chatbot that leverages **Retrieval-Augmented Generation (RAG)** to answer questions using the content of uploaded documents. It combines semantic search, embeddings, and large language models to deliver accurate and context-aware responses.

---

## 🚀 Features

### **1. Document Ingestion**

* **Upload PDFs**: Easily upload documents of any length.
* **Automatic Text Extraction**: Extracts text from uploaded files efficiently.
* **Intelligent Chunking**: Splits text into meaningful segments for better semantic retrieval.
* **Vector Storage**: Stores embeddings in a vector database for fast similarity search.

### **2. Semantic Search & RAG**

* **Context-Aware Retrieval**: Retrieves relevant chunks from the knowledge base based on user queries.
* **RAG Pipeline**: Combines retrieved context with prompts to the LLM for generating accurate and informative answers.
* **Enhanced Accuracy**: Ensures responses are grounded in the uploaded documents.

### **3. Interactive Chat Interface**

* **User-Friendly UI**: Built with Streamlit for seamless interactions.
* **Dynamic Responses**: Displays answers with references from the PDFs.
* **Session Management**: Maintains conversational context using temporary storage for multi-turn interactions.

---

## 🛠 Technologies Used

| Layer/Feature              | Technology                                  | Purpose                                              |
| -------------------------- | ------------------------------------------- | ---------------------------------------------------- |
| Backend                    | **FastAPI**                                 | Handles API endpoints and document processing        |
| Frontend                   | **Streamlit**                               | Provides a clean, interactive chat interface         |
| Temporary Storage          | **Redis**                                   | Stores previous conversations for context-aware chat |
| Vector Database            | **ChromaDB**                                | Stores and retrieves embeddings efficiently          |
| Embedding Model            | **sentence\_transformers/all-MiniLM-L6-v2** | Converts text chunks into embeddings                 |
| Large Language Model (LLM) | **deepseek/deepseek-chat-v3-0324\:free**    | Generates contextual and informative responses       |

---

## ⚡ How It Works

1. User uploads a PDF document.
2. The system extracts and chunks the text, creating embeddings stored in ChromaDB.
3. When the user asks a question, the chatbot performs a semantic search on the embeddings.
4. Retrieved chunks are combined with the user query and passed to the LLM.
5. The chatbot generates an accurate, context-aware response displayed in the Streamlit interface.

---

## 📈 Benefits

* Quickly obtain answers from large documents without manual reading.
* Maintain conversational context for more natural interactions.
* Scalable architecture for handling multiple users and documents.

