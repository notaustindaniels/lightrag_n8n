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
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any

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
    global rag_instance, metadata_store
    
    # Startup
    working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize metadata store file
    metadata_file = os.path.join(working_dir, "document_metadata.json")
    if os.path.exists(metadata_file):
        import json
        try:
            with open(metadata_file, 'r') as f:
                metadata_store = json.load(f)
                print(f"Loaded {len(metadata_store)} documents from metadata store")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            metadata_store = {}
    else:
        print("No existing metadata found, starting fresh")
    
    # Initialize LightRAG
    try:
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
        print("LightRAG initialized successfully")
    except Exception as e:
        print(f"Error initializing LightRAG: {e}")
        raise
    
    yield
    
    # Shutdown - save metadata
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata_store, f, indent=2)
            print(f"Saved {len(metadata_store)} documents to metadata store")
    except Exception as e:
        print(f"Error saving metadata: {e}")

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

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    print(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    return response

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

@app.get("/status")
async def get_status():
    """Get overall system status"""
    try:
        doc_count = len(metadata_store)
        
        return {
            "status": "running",
            "documents": {
                "total": doc_count,
                "processed": doc_count,
                "pending": 0,
                "failed": 0
            },
            "storage": {
                "working_dir": os.getenv("WORKING_DIR", "/app/data/rag_storage")
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug/storage")
async def debug_storage():
    """Debug endpoint to check storage status"""
    try:
        # Check what's in our metadata store
        metadata_count = len(metadata_store)
        
        # Try to get documents from LightRAG's internal storage
        lightrag_docs = []
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                if hasattr(rag_instance.doc_status, 'get_all_docs'):
                    lightrag_docs = await rag_instance.doc_status.get_all_docs()
                elif hasattr(rag_instance.doc_status, 'all'):
                    lightrag_docs = await rag_instance.doc_status.all()
            except Exception as e:
                lightrag_docs = f"Error accessing LightRAG storage: {str(e)}"
        
        return {
            "metadata_store_count": metadata_count,
            "metadata_store_ids": list(metadata_store.keys())[:10],  # First 10 IDs
            "lightrag_storage": lightrag_docs if isinstance(lightrag_docs, str) else len(lightrag_docs),
            "rag_instance_type": type(rag_instance).__name__,
            "doc_status_type": type(rag_instance.doc_status).__name__ if hasattr(rag_instance, 'doc_status') else "None"
        }
        
    except Exception as e:
        return {"error": str(e)}

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
        
        # Insert into LightRAG with file path and ID
        await rag_instance.ainsert(enriched_content, file_paths=[file_path], ids=[doc_id])
        
        print(f"Document inserted: {doc_id} with file_path: {file_path}")
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path
        }
        
    except Exception as e:
        print(f"Error inserting document: {str(e)}")
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
        
        # Insert into LightRAG with file path and ID
        await rag_instance.ainsert(request.text, file_paths=[file_path], ids=[doc_id])
        
        print(f"Document inserted: {doc_id} with file_path: {file_path}")
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path
        }
        
    except Exception as e:
        print(f"Error inserting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get all documents with proper file_path handling"""
    try:
        # Initialize response structure
        documents = []
        
        # First, get all documents from our metadata store
        for doc_id, metadata in metadata_store.items():
            doc_info = {
                "id": doc_id,
                "file_path": metadata.get('file_path', f"text/{doc_id}.txt"),
                "description": metadata.get('description', ''),
                "source_url": metadata.get('source_url', ''),
                "indexed_at": metadata.get('indexed_at', ''),
                "status": "processed",
                "content_summary": metadata.get('content_summary', '')
            }
            documents.append(doc_info)
        
        # Try to also get documents from LightRAG's internal storage
        # This ensures compatibility with the webui
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                lightrag_docs = []
                if hasattr(rag_instance.doc_status, 'get_all_docs'):
                    lightrag_docs = await rag_instance.doc_status.get_all_docs()
                elif hasattr(rag_instance.doc_status, 'all'):
                    lightrag_docs = await rag_instance.doc_status.all()
                
                # Merge LightRAG docs with our metadata
                for doc in lightrag_docs:
                    doc_id = doc.get('id', '')
                    if doc_id and doc_id not in metadata_store:
                        # This is a document in LightRAG but not in our store
                        documents.append({
                            "id": doc_id,
                            "file_path": doc.get('file_path', f"text/{doc_id}.txt"),
                            "description": doc.get('description', ''),
                            "status": doc.get('status', 'processed'),
                            "content_summary": doc.get('content_summary', '')
                        })
            except Exception as e:
                print(f"Warning: Could not fetch from LightRAG storage: {e}")
        
        # Return in the format the webui expects
        response = {
            "total": len(documents),
            "documents": documents,
            "statuses": {
                "processed": [d for d in documents if d.get('status') == 'processed'],
                "pending": [d for d in documents if d.get('status') == 'pending'],
                "failed": [d for d in documents if d.get('status') == 'failed']
            }
        }
        
        print(f"Returning {len(documents)} documents")
        return response
        
    except Exception as e:
        print(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/status")
async def get_documents_status():
    """Get document processing status - compatibility endpoint for webui"""
    try:
        # Count documents by status
        processed_count = len(metadata_store)
        
        return {
            "total": processed_count,
            "processed": processed_count,
            "pending": 0,
            "failed": 0,
            "statuses": {
                "processed": list(metadata_store.keys()),
                "pending": [],
                "failed": []
            }
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
        if hasattr(rag_instance, 'adelete_by_doc_id'):
            await rag_instance.adelete_by_doc_id(request.doc_ids)
        elif hasattr(rag_instance, 'delete_by_doc_id'):
            rag_instance.delete_by_doc_id(request.doc_ids)
        else:
            print("Warning: LightRAG instance doesn't have delete_by_doc_id method")
        
        # Remove from metadata store
        deleted_count = 0
        for doc_id in request.doc_ids:
            if doc_id in metadata_store:
                del metadata_store[doc_id]
                deleted_count += 1
        
        print(f"Deleted {deleted_count} documents from metadata store")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} documents",
            "deleted_ids": request.doc_ids
        }
        
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
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

@app.post("/documents/file")
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Upload a single document file"""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Compute document ID
        doc_id = compute_doc_id(text_content)
        
        # Use filename as file_path
        file_path = file.filename
        
        # Store metadata
        metadata_store[doc_id] = {
            "id": doc_id,
            "file_path": file_path,
            "description": description,
            "indexed_at": datetime.utcnow().isoformat(),
            "content_summary": text_content[:200] + "..." if len(text_content) > 200 else text_content
        }
        
        # Insert into LightRAG
        await rag_instance.ainsert(text_content, file_paths=[file_path], ids=[doc_id])
        
        print(f"File uploaded: {file_path} as document {doc_id}")
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "doc_id": doc_id,
            "file_path": file_path
        }
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple document files"""
    try:
        results = []
        
        for file in files:
            # Read file content
            content = await file.read()
            text_content = content.decode('utf-8')
            
            # Compute document ID
            doc_id = compute_doc_id(text_content)
            
            # Use filename as file_path
            file_path = file.filename
            
            # Store metadata
            metadata_store[doc_id] = {
                "id": doc_id,
                "file_path": file_path,
                "indexed_at": datetime.utcnow().isoformat(),
                "content_summary": text_content[:200] + "..." if len(text_content) > 200 else text_content
            }
            
            # Insert into LightRAG
            await rag_instance.ainsert(text_content, file_paths=[file_path], ids=[doc_id])
            
            results.append({
                "doc_id": doc_id,
                "file_path": file_path,
                "status": "success"
            })
        
        print(f"Batch upload: {len(files)} files processed")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(files)} files",
            "results": results
        }
        
    except Exception as e:
        print(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))