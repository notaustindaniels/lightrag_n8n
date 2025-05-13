import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# Environment variables with safe defaults
WORKING_DIR = os.getenv("WORKING_DIR", "/app/data/rag_storage")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))

# Validate required environment variables
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable is not set")

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # naive, local, global, hybrid
    doc_id: Optional[str] = None  # For document filtering
    
class InsertRequest(BaseModel):
    text: str
    doc_id: str  # Document identifier

class ConfigRequest(BaseModel):
    doc_id: str
    config: Dict[str, Any]

# Store multiple LightRAG instances for different documents
rag_instances: Dict[str, LightRAG] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting LightRAG API Server...")
    yield
    # Shutdown logic
    print("Shutting down LightRAG API Server...")

app = FastAPI(lifespan=lifespan)

def get_rag_instance(doc_id: str) -> LightRAG:
    """Get or create a LightRAG instance for a specific document."""
    if doc_id not in rag_instances:
        # Create directory for this document
        doc_dir = os.path.join(WORKING_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Initialize LightRAG instance
        rag = LightRAG(
            working_dir=doc_dir,
            llm_model_func=lambda query, **kwargs: openai_complete_if_cache(
                query,
                model=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs
            ),
            embedding_func=EmbeddingFunc(
                func=lambda texts: openai_embedding(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL
                ),
                max_token_size=EMBEDDING_MAX_TOKEN_SIZE
            )
        )
        rag_instances[doc_id] = rag
    
    return rag_instances[doc_id]

@app.post("/insert")
async def insert_text(request: InsertRequest):
    """Insert text into a specific document's RAG instance."""
    try:
        rag = get_rag_instance(request.doc_id)
        await rag.ainsert(request.text)
        return {"message": "Text inserted successfully", "doc_id": request.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert_batch")
async def insert_batch(doc_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Insert multiple files into a specific document's RAG instance."""
    try:
        rag = get_rag_instance(doc_id)
        inserted_files = []
        
        for file in files:
            content = await file.read()
            text = content.decode('utf-8')
            await rag.ainsert(text)
            inserted_files.append(file.filename)
        
        return {
            "message": "Files inserted successfully",
            "doc_id": doc_id,
            "files": inserted_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Query a specific document's RAG instance."""
    try:
        rag = get_rag_instance(request.doc_id)
        result = await rag.aquery(
            request.query,
            param=QueryParam(mode=request.mode)
        )
        return {"result": result, "doc_id": request.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route_query")
async def route_query(query: str = Body(...)):
    """Route query to determine which document(s) to search."""
    # This is a simple implementation - you could make this more sophisticated
    # using embeddings or another classifier
    
    # For now, let's return all available document IDs
    # In production, you'd want to implement smart routing logic
    available_docs = list(rag_instances.keys())
    
    # You could implement logic to select relevant documents based on query
    # For example, looking for keywords or using a classifier
    
    return {
        "query": query,
        "suggested_docs": available_docs,
        "message": "Use /query endpoint with specific doc_id"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "instances": list(rag_instances.keys())}

@app.post("/clear/{doc_id}")
async def clear_document(doc_id: str):
    """Clear a specific document's RAG instance."""
    try:
        if doc_id in rag_instances:
            # Clear the storage directory
            doc_dir = os.path.join(WORKING_DIR, doc_id)
            if os.path.exists(doc_dir):
                import shutil
                shutil.rmtree(doc_dir)
            
            # Remove from instances
            del rag_instances[doc_id]
            
            return {"message": f"Document {doc_id} cleared successfully"}
        else:
            return {"message": f"Document {doc_id} not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    uvicorn.run(app, host=host, port=port)