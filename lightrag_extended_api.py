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
    document_id: Optional[str] = None  # Accept document_id from n8n workflow

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

def generate_display_name_from_file_path(file_path: str, doc_id: str) -> str:
    """Generate a display name from a file path for legacy documents"""
    if "[" in file_path and "]" in file_path:
        parts = file_path.split("] ", 1)
        if len(parts) == 2:
            domain_part = parts[0] + "]"
            path_part = parts[1]
            if "/" in path_part:
                last_part = path_part.split("/")[-1]
                return f"{domain_part} {last_part}"
            else:
                return file_path
        else:
            return file_path
    else:
        return f"text/{doc_id[:8]}..."

async def cleanup_all_document_traces(doc_ids: List[str]):
    """Clean up all traces of documents from all LightRAG storage components"""
    try:
        # 1. Delete from vector storage
        if hasattr(rag_instance, 'vector_storage') and rag_instance.vector_storage:
            try:
                # Try to delete vectors associated with the document IDs
                for doc_id in doc_ids:
                    # LightRAG stores vectors with chunk IDs, we need to find all chunks
                    await rag_instance.vector_storage.delete_by_doc_id(doc_id)
            except Exception as e:
                print(f"Error deleting from vector storage: {e}")
        
        # 2. Delete from KV storage (full documents and chunks)
        if hasattr(rag_instance, 'kv_storage') and rag_instance.kv_storage:
            try:
                # Delete full documents
                for doc_id in doc_ids:
                    await rag_instance.kv_storage.delete({doc_id})
                    
                # Delete text chunks - LightRAG stores chunks with keys like "chunk-{doc_id}-{chunk_index}"
                # We need to find and delete all chunks for each document
                if hasattr(rag_instance.kv_storage, '_data'):
                    keys_to_delete = []
                    for key in rag_instance.kv_storage._data.keys():
                        for doc_id in doc_ids:
                            if key.startswith(f"chunk-{doc_id}"):
                                keys_to_delete.append(key)
                    if keys_to_delete:
                        await rag_instance.kv_storage.delete(set(keys_to_delete))
            except Exception as e:
                print(f"Error deleting from KV storage: {e}")
        
        # 3. Delete from doc status storage
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                await rag_instance.doc_status.delete({doc_id for doc_id in doc_ids})
            except Exception as e:
                print(f"Error deleting from doc status storage: {e}")
        
        # 4. Clean up graph storage (entities and relationships from these documents)
        if hasattr(rag_instance, 'graph_storage') and rag_instance.graph_storage:
            try:
                # This is more complex as we need to identify which entities/relationships
                # came from these documents. LightRAG may not track this directly.
                # For now, we'll leave the graph as is, but in a production system,
                # you might want to track document-entity mappings
                pass
            except Exception as e:
                print(f"Error with graph storage cleanup: {e}")
        
        # 5. Clear any caches that might contain document data
        if hasattr(rag_instance, 'llm_response_cache'):
            try:
                # Clear cache entries related to these documents
                # This is a simple approach - in production you might want more granular control
                await rag_instance.aclear_cache()
            except Exception as e:
                print(f"Error clearing cache: {e}")
                
    except Exception as e:
        print(f"Error in cleanup_all_document_traces: {e}")
        raise

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
        
        # Check if document_id was provided by n8n workflow
        if request.document_id:
            # The document_id from n8n is already in the format we want to display
            file_path = request.document_id
            display_name = request.document_id
            
            # Store the full URL path in metadata for reference
            full_path = None
            if request.source_url:
                parsed_url = urlparse(request.source_url)
                domain = parsed_url.netloc.replace('www.', '')
                path = parsed_url.path.strip('/')
                full_path = f"[{domain}] {path}" if path else f"[{domain}]"
        else:
            # Create file path based on source URL (actual page) for better visibility
            display_name = None
            if request.source_url:
                # Extract domain and path from source URL for better display
                parsed_url = urlparse(request.source_url)
                domain = parsed_url.netloc.replace('www.', '')
                path = parsed_url.path.strip('/')  # Remove leading and trailing slashes
                
                # For file_path, store the full path information
                # Format: "[domain.com] full/path"
                if path:
                    file_path = f"[{domain}] {path}"
                    # For display_name, show only the last part of the path
                    path_parts = path.split('/')
                    last_part = path_parts[-1] if path_parts[-1] else (path_parts[-2] if len(path_parts) > 1 else path)
                    display_name = f"[{domain}] {last_part}"
                else:
                    # For root domain (https://ai.pydantic.dev/)
                    file_path = f"[{domain}]"
                    display_name = f"[{domain}]"
            else:
                file_path = f"text/{doc_id}.txt"
                display_name = f"text/{doc_id[:8]}..."  # Shortened ID for display
        
        # Store metadata
        metadata_entry = {
            "id": doc_id,
            "file_path": file_path,
            "display_name": display_name,
            "source_url": request.source_url,
            "description": request.description,
            "indexed_at": datetime.utcnow().isoformat(),
            "sitemap_url": request.sitemap_url,
            "doc_index": request.doc_index,
            "total_docs": request.total_docs,
            "content_summary": enriched_content[:200] + "..." if len(enriched_content) > 200 else enriched_content
        }
        
        # Add full_path if we have it (when using document_id from n8n)
        if request.document_id and 'full_path' in locals():
            metadata_entry["full_path"] = full_path
            
        metadata_store[doc_id] = metadata_entry
        
        # Save metadata after each insert
        save_metadata_store()
        
        # Insert into LightRAG with file path
        await rag_instance.ainsert(enriched_content, file_paths=[file_path])
        
        # Log for debugging
        print(f"Document inserted - ID: {doc_id}, file_path: {file_path}, document_id from n8n: {request.document_id}")
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path,
            "display_name": display_name
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
        
        # Generate display_name from file_path
        display_name = generate_display_name_from_file_path(file_path, doc_id)
        
        # Store metadata
        metadata_store[doc_id] = {
            "id": doc_id,
            "file_path": file_path,
            "display_name": display_name,
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
                                
                                # Handle legacy documents without display_name
                                display_name = metadata.get('display_name')
                                if not display_name:
                                    display_name = generate_display_name_from_file_path(file_path, doc_id)
                                
                                documents.append({
                                    "id": doc_id,
                                    "file_path": file_path,
                                    "display_name": display_name,
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
                                
                                # Handle legacy documents without display_name
                                display_name = metadata.get('display_name')
                                if not display_name:
                                    display_name = generate_display_name_from_file_path(file_path, doc_id)
                                
                                documents.append({
                                    "id": doc_id,
                                    "file_path": file_path,
                                    "display_name": display_name,
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
                                    
                                    # Handle legacy documents without display_name
                                    display_name = metadata.get('display_name')
                                    if not display_name:
                                        # Generate display_name from file_path for legacy documents
                                        if "[" in file_path and "]" in file_path:
                                            parts = file_path.split("] ", 1)
                                            if len(parts) == 2:
                                                domain_part = parts[0] + "]"
                                                path_part = parts[1]
                                                if "/" in path_part:
                                                    last_part = path_part.split("/")[-1]
                                                    display_name = f"{domain_part} {last_part}"
                                                else:
                                                    display_name = file_path
                                            else:
                                                display_name = file_path
                                        else:
                                            display_name = f"text/{doc_id[:8]}..."
                                    
                                    documents.append({
                                        "id": doc_id,
                                        "file_path": file_path,
                                        "display_name": display_name,
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
                file_path = metadata.get('file_path', f"text/{doc_id}.txt")
                display_name = metadata.get('display_name')
                
                # Handle legacy documents without display_name
                if not display_name:
                    # Generate display_name from file_path for legacy documents
                    if "[" in file_path and "]" in file_path:
                        parts = file_path.split("] ", 1)
                        if len(parts) == 2:
                            domain_part = parts[0] + "]"
                            path_part = parts[1]
                            if "/" in path_part:
                                last_part = path_part.split("/")[-1]
                                display_name = f"{domain_part} {last_part}"
                            else:
                                display_name = file_path
                        else:
                            display_name = file_path
                    else:
                        display_name = f"text/{doc_id[:8]}..."
                
                documents.append({
                    "id": doc_id,
                    "file_path": file_path,
                    "display_name": display_name,
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
                file_path = metadata.get('file_path', f"text/{doc_id}.txt")
                display_name = metadata.get('display_name')
                if not display_name:
                    display_name = generate_display_name_from_file_path(file_path, doc_id)
                    
                matching_docs.append({
                    "doc_id": doc_id,
                    "source_url": metadata.get('source_url'),
                    "file_path": file_path,
                    "display_name": display_name,
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
        # First clean up all document traces from LightRAG storages
        await cleanup_all_document_traces(request.doc_ids)
        
        # Then call the standard delete method
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
            # First clean up all document traces from LightRAG storages
            await cleanup_all_document_traces(docs_to_delete)
            
            # Then call the standard delete method
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