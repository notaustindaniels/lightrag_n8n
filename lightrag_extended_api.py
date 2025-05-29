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
    
    # Ensure doc_status storage is available
    if not hasattr(rag_instance, 'doc_status') or not rag_instance.doc_status:
        print("Warning: doc_status storage not initialized properly")
    else:
        print("doc_status storage initialized successfully")
    
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
        
        # Update the doc_status if available
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                await rag_instance.doc_status.upsert({
                    doc_id: {
                        "id": doc_id,
                        "file_path": file_path,
                        "status": "processed",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                })
            except:
                pass
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path,
            "id": doc_id  # Include id for WebUI compatibility
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
        
        # Update the doc_status if available
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                await rag_instance.doc_status.upsert({
                    doc_id: {
                        "id": doc_id,
                        "file_path": file_path,
                        "status": "processed",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                })
            except:
                pass
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path,
            "id": doc_id  # Include id for WebUI compatibility
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get all documents with proper file_path handling"""
    try:
        # Try to get documents from LightRAG's storage
        documents = []
        
        # First, try to get from doc_status if available
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                # Get all documents from the doc status storage
                all_doc_ids = await rag_instance.doc_status.all_keys()
                
                for doc_id in all_doc_ids:
                    doc_status = await rag_instance.doc_status.get(doc_id)
                    if doc_status:
                        # Get metadata from our store
                        metadata = metadata_store.get(doc_id, {})
                        
                        # Create a document entry that matches the expected format
                        doc_entry = {
                            "id": doc_id,
                            "file_path": metadata.get('file_path', doc_status.get('file_path', f"text/{doc_id}.txt")),
                            "status": doc_status.get('status', 'processed'),
                            "created_at": metadata.get('indexed_at', doc_status.get('created_at', '')),
                            "updated_at": metadata.get('indexed_at', doc_status.get('updated_at', '')),
                            "description": metadata.get('description', ''),
                            "metadata": metadata
                        }
                        documents.append(doc_entry)
            except Exception as e:
                print(f"Error accessing doc_status: {e}")
        
        # If no documents from doc_status, use our metadata store
        if not documents:
            for doc_id, metadata in metadata_store.items():
                doc_entry = {
                    "id": doc_id,
                    "file_path": metadata.get('file_path', f"text/{doc_id}.txt"),
                    "status": "processed",
                    "created_at": metadata.get('indexed_at', ''),
                    "updated_at": metadata.get('indexed_at', ''),
                    "description": metadata.get('description', ''),
                    "metadata": metadata
                }
                documents.append(doc_entry)
        
        # Return in the format expected by the WebUI
        return documents
        
    except Exception as e:
        print(f"Error in get_documents: {e}")
        # Return empty list instead of error to prevent WebUI from breaking
        return []

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

# Additional endpoints for WebUI compatibility
@app.get("/storage/kv")
async def get_kv_storage():
    """Get key-value storage statistics"""
    try:
        kv_storage = rag_instance.kv_storage
        if hasattr(kv_storage, 'all_keys'):
            keys = await kv_storage.all_keys()
            return {
                "total_keys": len(keys),
                "storage_type": type(kv_storage).__name__
            }
        return {"total_keys": 0, "storage_type": "unknown"}
    except:
        return {"total_keys": 0, "storage_type": "unknown"}

@app.get("/storage/graph")
async def get_graph_storage():
    """Get graph storage statistics"""
    try:
        graph_storage = rag_instance.graph_storage
        nodes = await graph_storage.get_all_nodes()
        edges = await graph_storage.get_all_edges()
        return {
            "total_nodes": len(nodes) if nodes else 0,
            "total_edges": len(edges) if edges else 0,
            "storage_type": type(graph_storage).__name__
        }
    except:
        return {"total_nodes": 0, "total_edges": 0, "storage_type": "unknown"}

@app.get("/storage/vector")
async def get_vector_storage():
    """Get vector storage statistics"""
    try:
        vector_storage = rag_instance.vector_storage
        if hasattr(vector_storage, 'size'):
            size = await vector_storage.size()
            return {
                "total_vectors": size,
                "storage_type": type(vector_storage).__name__
            }
        return {"total_vectors": 0, "storage_type": "unknown"}
    except:
        return {"total_vectors": 0, "storage_type": "unknown"}

@app.get("/config")
async def get_config():
    """Get configuration for WebUI"""
    return {
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        "working_dir": os.getenv("WORKING_DIR", "/app/data/rag_storage"),
        "max_parallel_insert": 2,
        "enable_llm_cache": True,
        "features": {
            "query": True,
            "insert": True,
            "delete": True,
            "export": False,
            "visualize": False
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics for WebUI dashboard"""
    try:
        doc_count = len(metadata_store)
        
        # Try to get more accurate counts from storages
        try:
            if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
                all_keys = await rag_instance.doc_status.all_keys()
                doc_count = max(doc_count, len(all_keys))
        except:
            pass
        
        return {
            "total_documents": doc_count,
            "total_chunks": 0,  # Would need to query chunk storage
            "total_entities": 0,  # Would need to query entity storage
            "total_relations": 0,  # Would need to query relation storage
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0,
            "error": str(e)
        }