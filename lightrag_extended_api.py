#!/usr/bin/env python3
"""
Extended LightRAG Server - Properly integrates with the original server
"""
import os
import sys
import hashlib
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import HTTPException
import logging

# Set up environment variables before importing LightRAG
os.environ.setdefault("WORKING_DIR", "/app/data/rag_storage")
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "9621")

# Import the LightRAG server components
try:
    from lightrag.api.lightrag_server import get_application, global_args
    from lightrag.api.config import parse_args
    import uvicorn
except ImportError as e:
    print(f"Error importing LightRAG server: {e}")
    print("Make sure lightrag-hku[api] is installed")
    sys.exit(1)

# Models for our custom endpoints
class EnhancedTextInsertRequest(BaseModel):
    text: str
    description: Optional[str] = None
    source_url: Optional[str] = None
    sitemap_url: Optional[str] = None
    doc_index: Optional[int] = None
    total_docs: Optional[int] = None

class DeleteByIdRequest(BaseModel):
    doc_ids: List[str]

# Global metadata store
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
        except:
            metadata_store = {}

def save_metadata():
    """Save metadata to file"""
    if metadata_file_path:
        try:
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata_store, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

def add_custom_endpoints(app):
    """Add our custom endpoints to the existing app"""
    
    # Import here to avoid circular imports
    from lightrag.api.utils_api import get_combined_auth_dependency
    from fastapi import Depends
    
    # Get the RAG instance from the app
    # The RAG instance is created during app initialization
    # We'll need to access it through the routes
    rag_instance = None
    
    # Get auth dependency
    api_key = os.getenv("LIGHTRAG_API_KEY")
    combined_auth = get_combined_auth_dependency(api_key)
    
    @app.post("/documents/text/enhanced", dependencies=[Depends(combined_auth)])
    async def insert_text_enhanced(request: EnhancedTextInsertRequest):
        """Enhanced text insertion with full metadata support"""
        nonlocal rag_instance
        
        # Get RAG instance from document routes if not already set
        if not rag_instance:
            for route in app.routes:
                if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__closure__'):
                    for cell in route.endpoint.__closure__:
                        if hasattr(cell.cell_contents, '__class__'):
                            if cell.cell_contents.__class__.__name__ == 'LightRAG':
                                rag_instance = cell.cell_contents
                                break
                if rag_instance:
                    break
        
        if not rag_instance:
            raise HTTPException(status_code=500, detail="RAG instance not found")
        
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
    
    @app.get("/documents/by-sitemap/{sitemap_url:path}", dependencies=[Depends(combined_auth)])
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
    
    @app.delete("/documents/by-sitemap/{sitemap_url:path}", dependencies=[Depends(combined_auth)])
    async def delete_documents_by_sitemap(sitemap_url: str):
        """Delete all documents for a specific sitemap URL"""
        nonlocal rag_instance
        
        # Get RAG instance if not already set
        if not rag_instance:
            for route in app.routes:
                if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__closure__'):
                    for cell in route.endpoint.__closure__:
                        if hasattr(cell.cell_contents, '__class__'):
                            if cell.cell_contents.__class__.__name__ == 'LightRAG':
                                rag_instance = cell.cell_contents
                                break
                if rag_instance:
                    break
        
        if not rag_instance:
            raise HTTPException(status_code=500, detail="RAG instance not found")
        
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
    
    @app.delete("/documents/by-id", dependencies=[Depends(combined_auth)])
    async def delete_documents_by_id(request: DeleteByIdRequest):
        """Delete documents by their IDs"""
        nonlocal rag_instance
        
        # Get RAG instance if not already set
        if not rag_instance:
            for route in app.routes:
                if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__closure__'):
                    for cell in route.endpoint.__closure__:
                        if hasattr(cell.cell_contents, '__class__'):
                            if cell.cell_contents.__class__.__name__ == 'LightRAG':
                                rag_instance = cell.cell_contents
                                break
                if rag_instance:
                    break
        
        if not rag_instance:
            raise HTTPException(status_code=500, detail="RAG instance not found")
        
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
    
    # Override the /documents/text endpoint to ensure file_path is set
    from lightrag.api.routers.document_routes import TextInsertRequest
    
    @app.post("/documents/text", dependencies=[Depends(combined_auth)])
    async def insert_text_with_filepath(request: TextInsertRequest):
        """Standard text insertion with file_path support"""
        nonlocal rag_instance
        
        # Get RAG instance if not already set
        if not rag_instance:
            for route in app.routes:
                if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__closure__'):
                    for cell in route.endpoint.__closure__:
                        if hasattr(cell.cell_contents, '__class__'):
                            if cell.cell_contents.__class__.__name__ == 'LightRAG':
                                rag_instance = cell.cell_contents
                                break
                if rag_instance:
                    break
        
        if not rag_instance:
            raise HTTPException(status_code=500, detail="RAG instance not found")
        
        try:
            # Compute document ID
            doc_id = compute_doc_id(request.text)
            
            # Use provided file_path or create one
            file_path = getattr(request, 'file_path', None) or f"text/{doc_id}.txt"
            
            # Store metadata
            metadata_store[doc_id] = {
                "id": doc_id,
                "file_path": file_path,
                "description": getattr(request, 'description', None),
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
    
    print("Custom endpoints added to LightRAG server")

if __name__ == "__main__":
    # Load metadata
    load_metadata()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create the app using the factory function
    app = get_application(args)
    
    # Add our custom endpoints
    add_custom_endpoints(app)
    
    # Run the server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    
    print(f"Starting Extended LightRAG Server on {host}:{port}")
    print("Web UI will be available at the root URL")
    
    uvicorn.run(app, host=host, port=port)