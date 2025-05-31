#!/usr/bin/env python3
"""
Extended LightRAG API Server that adds missing functionality for document management
"""
import os
import asyncio
import hashlib
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

# Setup logging
setup_logger("lightrag", level="INFO")

# Models
class TextInsertRequest(BaseModel):
    text: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedTextInsertRequest(BaseModel):
    text: str
    description: Optional[str] = None
    source_url: Optional[str] = None
    sitemap_url: Optional[str] = None
    doc_index: Optional[int] = None
    total_docs: Optional[int] = None

class DocumentMetadata(BaseModel):
    id: str
    file_path: Optional[str] = None
    source_url: Optional[str] = None
    sitemap_url: Optional[str] = None
    description: Optional[str] = None
    indexed_at: Optional[str] = None
    content_summary: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    file_path: Optional[str] = Field(default="")  # Default empty string to avoid validation error
    metadata: Optional[DocumentMetadata] = None

class DeleteByIdRequest(BaseModel):
    doc_ids: List[str]

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    stream: bool = False

# Global variables
rag_instance = None
metadata_store = {}  # In-memory metadata store

def compute_doc_id(content: str) -> str:
    """Compute document ID using MD5 hash of content"""
    return f"doc-{hashlib.md5(content.strip().encode()).hexdigest()}"

def save_metadata_store():
    """Save metadata store to disk"""
    try:
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        metadata_file = os.path.join(working_dir, "document_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata_store, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata store: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global rag_instance
    
    # Startup
    working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize metadata store file
    metadata_file = os.path.join(working_dir, "document_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                global metadata_store
                metadata_store = json.load(f)
        except:
            metadata_store = {}
    
    # Initialize LightRAG
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
    
    yield
    
    # Shutdown - save metadata
    save_metadata_store()

# Create FastAPI app
app = FastAPI(
    title="Extended LightRAG API",
    description="LightRAG API with enhanced document management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebUI if available
webui_path = "/app/webui"
webui_mounted = False

if os.path.exists(webui_path) and os.path.isdir(webui_path):
    print(f"Mounting WebUI from {webui_path}")
    app.mount("/webui", StaticFiles(directory=webui_path, html=True), name="webui")
    webui_mounted = True
else:
    # Try alternative paths
    alt_paths = [
        "/usr/local/lib/python3.11/site-packages/lightrag/api/webui",
        "./webui",
        "../lightrag/api/webui"
    ]
    for path in alt_paths:
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Mounting WebUI from {path}")
            app.mount("/webui", StaticFiles(directory=path, html=True), name="webui")
            webui_path = path
            webui_mounted = True
            break
    
    if not webui_mounted:
        print("Warning: WebUI directory not found. Web interface will not be available.")

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to WebUI"""
    if webui_mounted:
        return RedirectResponse(url="/webui/")
    else:
        return {"message": "LightRAG Extended API", "webui": "Not available", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "extended-lightrag"}

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
        
        # Create file path based on sitemap URL with domain prefix for better visibility
        if request.sitemap_url:
            # Extract domain from sitemap URL for better display when truncated
            parsed_url = urlparse(request.sitemap_url)
            domain = parsed_url.netloc.replace('www.', '')
            # Format: "[domain] sitemap.xml" - domain in brackets for visibility
            file_path = f"[{domain}] sitemap.xml"
        else:
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
        
        # Save metadata after each insert
        save_metadata_store()
        
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
        
        # Save metadata after each insert
        save_metadata_store()
        
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

@app.get("/documents")
async def get_documents():
    """Get all documents with proper file_path handling"""
    try:
        documents = []
        
        # Try to get documents from LightRAG's doc_status storage
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            try:
                # Try different methods to get documents from storage
                doc_status_storage = rag_instance.doc_status
                
                # Method 1: Try to get all documents directly
                if hasattr(doc_status_storage, 'get_all'):
                    try:
                        all_docs = await doc_status_storage.get_all()
                        if all_docs:
                            for doc_id, doc_data in all_docs.items():
                                # Get metadata from our store
                                metadata = metadata_store.get(doc_id, {})
                                
                                # Ensure file_path exists
                                file_path = metadata.get('file_path', doc_data.get('file_path', f"text/{doc_id}.txt"))
                                
                                documents.append({
                                    "id": doc_id,
                                    "file_path": file_path,
                                    "metadata": metadata,
                                    "status": doc_data.get('status', 'processed')
                                })
                    except Exception as e:
                        print(f"Error with get_all method: {e}")
                
                # Method 2: Try to iterate through storage if it's dict-like
                if not documents and hasattr(doc_status_storage, '_data'):
                    try:
                        storage_data = doc_status_storage._data
                        if isinstance(storage_data, dict):
                            for doc_id, doc_data in storage_data.items():
                                # Get metadata from our store
                                metadata = metadata_store.get(doc_id, {})
                                
                                # Ensure file_path exists
                                file_path = metadata.get('file_path', f"text/{doc_id}.txt")
                                
                                documents.append({
                                    "id": doc_id,
                                    "file_path": file_path,
                                    "metadata": metadata,
                                    "status": 'processed'
                                })
                    except Exception as e:
                        print(f"Error accessing _data: {e}")
                
                # Method 3: Try JSON storage file directly
                if not documents:
                    try:
                        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                        doc_status_file = os.path.join(working_dir, "doc_status.json")
                        if os.path.exists(doc_status_file):
                            with open(doc_status_file, 'r') as f:
                                doc_status_data = json.load(f)
                                for doc_id, doc_data in doc_status_data.items():
                                    # Get metadata from our store
                                    metadata = metadata_store.get(doc_id, {})
                                    
                                    # Ensure file_path exists
                                    file_path = metadata.get('file_path', f"text/{doc_id}.txt")
                                    
                                    documents.append({
                                        "id": doc_id,
                                        "file_path": file_path,
                                        "metadata": metadata,
                                        "status": doc_data.get('status', 'processed')
                                    })
                    except Exception as e:
                        print(f"Error reading doc_status.json: {e}")
            except Exception as e:
                print(f"Error accessing doc_status storage: {e}")
        
        # If no documents found in doc_status, use metadata store
        if not documents:
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
        print(f"Error in get_documents: {e}")
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
        
        # Save updated metadata
        save_metadata_store()
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} documents",
            "deleted_ids": request.doc_ids
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
            # Check both sitemap_identifier (legacy) and sitemap_url
            if (metadata.get('sitemap_url') == sitemap_url or 
                metadata.get('sitemap_identifier') == f"[SITEMAP: {sitemap_url}]"):
                docs_to_delete.append(doc_id)
        
        if docs_to_delete:
            # Delete from LightRAG
            await rag_instance.adelete_by_doc_id(docs_to_delete)
            
            # Remove from metadata store
            for doc_id in docs_to_delete:
                del metadata_store[doc_id]
            
            # Save updated metadata
            save_metadata_store()
        
        return {
            "status": "success",
            "message": f"Deleted {len(docs_to_delete)} documents for sitemap {sitemap_url}",
            "deleted_count": len(docs_to_delete),
            "sitemap_url": sitemap_url
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