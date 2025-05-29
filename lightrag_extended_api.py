#!/usr/bin/env python3
"""
Extended LightRAG API Server that adds missing functionality for document management
"""
import os
import asyncio
import hashlib
import pkg_resources
from typing import List, Dict, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
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
        import json
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
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata_store, f, indent=2)
    except:
        pass

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

# Mount webui static files
try:
    # Try to find the webui in the installed lightrag package
    # Get the package location
    package_path = pkg_resources.get_distribution('lightrag-hku').location
    webui_path = os.path.join(package_path, 'lightrag', 'api', 'webui')
    
    # Check if webui exists in the package
    if os.path.exists(webui_path):
        app.mount("/webui", StaticFiles(directory=webui_path, html=True), name="webui")
        print(f"WebUI mounted from: {webui_path}")
    else:
        # Try alternative location (if running from source)
        alt_webui_path = os.path.join(os.path.dirname(__file__), 'webui')
        if os.path.exists(alt_webui_path):
            app.mount("/webui", StaticFiles(directory=alt_webui_path, html=True), name="webui")
            print(f"WebUI mounted from: {alt_webui_path}")
        else:
            print("Warning: WebUI static files not found")
except Exception as e:
    print(f"Warning: Could not mount WebUI: {e}")

@app.get("/")
async def root():
    """Redirect to webui"""
    return RedirectResponse(url="/webui")

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
        # Get documents from LightRAG's document status storage
        doc_status = rag_instance.doc_status
        
        documents = []
        if hasattr(doc_status, 'get_all_docs'):
            raw_docs = await doc_status.get_all_docs()
            
            for doc in raw_docs:
                doc_id = doc.get('id', '')
                # Get metadata from our store
                metadata = metadata_store.get(doc_id, {})
                
                # Ensure file_path exists
                file_path = metadata.get('file_path', doc.get('file_path', f"text/{doc_id}.txt"))
                
                documents.append({
                    "id": doc_id,
                    "file_path": file_path,
                    "metadata": metadata,
                    "status": doc.get('status', 'processed')
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