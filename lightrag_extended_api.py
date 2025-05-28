#!/usr/bin/env python3
"""
Extended LightRAG API Server with enhanced functionality and web UI support
"""
import os
import sys
import asyncio
import hashlib
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class TextInsertRequest(BaseModel):
    text: str
    description: Optional[str] = None
    file_path: Optional[str] = None

class BatchTextInsertRequest(BaseModel):
    texts: List[str]
    description: Optional[str] = None

class EnhancedTextInsertRequest(BaseModel):
    text: str
    description: Optional[str] = None
    source_url: Optional[str] = None
    sitemap_url: Optional[str] = None
    doc_index: Optional[int] = None
    total_docs: Optional[int] = None

class DeleteByIdRequest(BaseModel):
    doc_ids: List[str]

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    stream: bool = False

# Global variables
rag_instance = None
metadata_store = {}
metadata_file_path = None

def compute_doc_id(content: str) -> str:
    """Compute document ID using MD5 hash of content"""
    return f"doc-{hashlib.md5(content.strip().encode()).hexdigest()}"

def load_metadata():
    """Load metadata from file"""
    global metadata_store, metadata_file_path
    
    working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
    metadata_file_path = os.path.join(working_dir, "document_metadata.json")
    
    if os.path.exists(metadata_file_path):
        try:
            with open(metadata_file_path, 'r') as f:
                metadata_store = json.load(f)
                logger.info(f"Loaded {len(metadata_store)} metadata entries")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            metadata_store = {}

def save_metadata():
    """Save metadata to file"""
    global metadata_store, metadata_file_path
    
    if metadata_file_path:
        try:
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata_store, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global rag_instance
    
    # Startup
    logger.info("Starting Enhanced LightRAG API Server...")
    
    # Load metadata
    load_metadata()
    
    # Initialize LightRAG
    working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
    os.makedirs(working_dir, exist_ok=True)
    
    rag_instance = LightRAG(
        working_dir=working_dir,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embed
        ),
        llm_model_func=gpt_4o_mini_complete,
    )
    
    await rag_instance.initialize_storages()
    await initialize_pipeline_status()
    
    logger.info("LightRAG initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    save_metadata()

# Create FastAPI app
app = FastAPI(
    title="Enhanced LightRAG API",
    description="LightRAG API with enhanced document management and web UI",
    version="1.0.0",
    lifespan=lifespan
)

# Try to mount static files for web UI
webui_path = None
for possible_path in [
    "/usr/local/lib/python3.11/site-packages/lightrag/api/webui",
    "/app/lightrag/api/webui",
    "./lightrag/api/webui",
    "/usr/local/lib/python3.11/site-packages/lightrag_api/webui",
]:
    if os.path.exists(possible_path):
        webui_path = possible_path
        break

if webui_path:
    app.mount("/webui", StaticFiles(directory=webui_path, html=True), name="webui")
    logger.info(f"Web UI mounted from: {webui_path}")
else:
    logger.warning("Web UI files not found, web interface will not be available")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "enhanced-lightrag"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Enhanced LightRAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "webui": "/webui" if webui_path else None,
            "documents": "/documents",
            "query": "/query"
        }
    }

# Redirect /webui to /webui/ if needed
@app.get("/webui")
async def webui_redirect():
    if webui_path:
        return HTMLResponse(content='<meta http-equiv="refresh" content="0; url=/webui/" />')
    else:
        raise HTTPException(status_code=404, detail="Web UI not available")

@app.post("/documents/text")
async def insert_text(request: TextInsertRequest):
    """Standard text insertion with file_path support"""
    try:
        # Compute document ID
        doc_id = compute_doc_id(request.text)
        
        # Use provided file_path or create one
        file_path = request.file_path if request.file_path else f"text/{doc_id}.txt"
        
        # Store metadata
        metadata_store[doc_id] = {
            "id": doc_id,
            "file_path": file_path,
            "description": request.description,
            "indexed_at": datetime.utcnow().isoformat(),
            "content_summary": request.text[:200] + "..." if len(request.text) > 200 else request.text
        }
        
        # Save metadata
        save_metadata()
        
        # Insert into LightRAG with file path
        await rag_instance.ainsert(request.text, file_paths=[file_path])
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/batch")
async def insert_batch(request: BatchTextInsertRequest):
    """Batch text insertion"""
    try:
        results = []
        file_paths = []
        
        for text in request.texts:
            doc_id = compute_doc_id(text)
            file_path = f"text/{doc_id}.txt"
            file_paths.append(file_path)
            
            # Store metadata
            metadata_store[doc_id] = {
                "id": doc_id,
                "file_path": file_path,
                "description": request.description,
                "indexed_at": datetime.utcnow().isoformat(),
                "content_summary": text[:200] + "..." if len(text) > 200 else text
            }
            
            results.append({
                "doc_id": doc_id,
                "file_path": file_path
            })
        
        # Save metadata
        save_metadata()
        
        # Insert into LightRAG
        await rag_instance.ainsert(request.texts, file_paths=file_paths)
        
        return {
            "status": "success",
            "message": f"Successfully inserted {len(request.texts)} documents",
            "documents": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/text/enhanced")
async def insert_text_enhanced(request: EnhancedTextInsertRequest):
    """Enhanced text insertion with full metadata support"""
    try:
        # Prepare metadata header
        metadata_parts = []
        if request.source_url:
            metadata_parts.append(f"[SOURCE_URL: {request.source_url}]")
        if request.sitemap_url:
            metadata_parts.append(f"[SITEMAP: {request.sitemap_url}]")
        metadata_parts.append(f"[INDEXED: {datetime.utcnow().isoformat()}]")
        if request.doc_index and request.total_docs:
            metadata_parts.append(f"[DOC_INDEX: {request.doc_index} of {request.total_docs}]")
        
        # Add metadata to content
        if metadata_parts:
            enriched_content = "\n".join(metadata_parts) + "\n\n" + request.text
        else:
            enriched_content = request.text
        
        # Compute document ID
        doc_id = compute_doc_id(enriched_content)
        
        # Create file path based on source URL or use a default
        file_path = request.source_url if request.source_url else f"text/{doc_id}.txt"
        
        # Store metadata
        metadata_store[doc_id] = {
            "id": doc_id,
            "file_path": file_path,
            "source_url": request.source_url,
            "description": request.description,
            "indexed_at": datetime.utcnow().isoformat(),
            "sitemap_url": request.sitemap_url,
            "doc_index": request.doc_index,
            "total_docs": request.total_docs,
            "content_summary": enriched_content[:200] + "..." if len(enriched_content) > 200 else enriched_content
        }
        
        # Save metadata
        save_metadata()
        
        # Insert into LightRAG with file path
        await rag_instance.ainsert(enriched_content, file_paths=[file_path])
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get all documents with proper file_path handling"""
    try:
        documents = []
        
        # Get documents from metadata store
        for doc_id, metadata in metadata_store.items():
            documents.append({
                "id": doc_id,
                "file_path": metadata.get('file_path', f"text/{doc_id}.txt"),
                "metadata": metadata,
                "status": "processed"
            })
        
        return {
            "statuses": {
                "processed": documents,
                "pending": [],
                "failed": []
            },
            "total": len(documents)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/by-sitemap/{sitemap_url:path}")
async def get_documents_by_sitemap(sitemap_url: str):
    """Get all documents for a specific sitemap URL"""
    try:
        matching_docs = []
        
        for doc_id, metadata in metadata_store.items():
            # Check both sitemap_url and sitemap_identifier (for legacy support)
            if (metadata.get('sitemap_url') == sitemap_url or 
                metadata.get('sitemap_identifier') == f"[SITEMAP: {sitemap_url}]"):
                matching_docs.append({
                    "doc_id": doc_id,
                    "source_url": metadata.get('source_url'),
                    "file_path": metadata.get('file_path'),
                    "indexed_at": metadata.get('indexed_at')
                })
        
        return {
            "sitemap_url": sitemap_url,
            "documents": matching_docs,
            "count": len(matching_docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/by-sitemap/{sitemap_url:path}")
async def delete_documents_by_sitemap(sitemap_url: str):
    """Delete all documents for a specific sitemap URL"""
    try:
        # Find all documents with this sitemap URL
        docs_to_delete = []
        for doc_id, metadata in metadata_store.items():
            # Check both sitemap_url and sitemap_identifier (for legacy support)
            if (metadata.get('sitemap_url') == sitemap_url or 
                metadata.get('sitemap_identifier') == f"[SITEMAP: {sitemap_url}]"):
                docs_to_delete.append(doc_id)
        
        if docs_to_delete:
            # Delete from LightRAG
            await rag_instance.adelete_by_doc_id(docs_to_delete)
            
            # Remove from metadata store
            for doc_id in docs_to_delete:
                del metadata_store[doc_id]
            
            # Save metadata
            save_metadata()
        
        return {
            "status": "success",
            "message": f"Deleted {len(docs_to_delete)} documents for sitemap {sitemap_url}",
            "deleted_count": len(docs_to_delete),
            "sitemap_url": sitemap_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/by-id")
async def delete_documents_by_id(request: DeleteByIdRequest):
    """Delete documents by their IDs"""
    try:
        # Delete from LightRAG
        await rag_instance.adelete_by_doc_id(request.doc_ids)
        
        # Remove from metadata store
        deleted_count = 0
        for doc_id in request.doc_ids:
            if doc_id in metadata_store:
                del metadata_store[doc_id]
                deleted_count += 1
        
        # Save metadata
        save_metadata()
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} documents",
            "deleted_ids": request.doc_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents"""
    try:
        # Clear LightRAG storage
        doc_count = len(metadata_store)
        
        if doc_count > 0:
            # Get all doc IDs
            all_doc_ids = list(metadata_store.keys())
            
            # Delete from LightRAG
            await rag_instance.adelete_by_doc_id(all_doc_ids)
            
            # Clear metadata store
            metadata_store.clear()
            
            # Save empty metadata
            save_metadata()
        
        return {
            "status": "success",
            "message": f"Cleared {doc_count} documents"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        param = QueryParam(
            mode=request.mode,
            stream=request.stream
        )
        
        result = await rag_instance.aquery(request.query, param=param)
        
        return {
            "query": request.query,
            "response": result,
            "mode": request.mode
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "9621"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "lightrag_extended_api:app",
        host=host,
        port=port,
        reload=False
    )