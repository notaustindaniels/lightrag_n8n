#!/usr/bin/env python3
"""
Extended LightRAG API Server that adds missing functionality while preserving the web UI
"""
import os
import sys
import asyncio
import hashlib
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Add the LightRAG API module to the path
import lightrag.api.lightrag_server as lightrag_server
from lightrag.api.models import (
    InsertDocumentsRequest,
    InsertDocumentsResponse,
    QueryRequest
)

# Import the FastAPI app from LightRAG
from lightrag.api.lightrag_server import app, rag_instance

# Models for extended functionality
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
        except Exception as e:
            print(f"Error loading metadata: {e}")
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
            print(f"Error saving metadata: {e}")

# Load metadata on startup
load_metadata()

# Add extended endpoints to the existing app
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
        
        # Use the standard insert endpoint
        insert_request = InsertDocumentsRequest(
            texts=[enriched_content],
            file_paths=[file_path]
        )
        
        # Get the RAG instance
        rag = await lightrag_server.get_rag_instance()
        
        # Insert using the standard method
        await rag.ainsert(insert_request.texts, file_paths=insert_request.file_paths)
        
        return {
            "status": "success",
            "message": "Document inserted successfully",
            "doc_id": doc_id,
            "file_path": file_path
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
            # Get the RAG instance
            rag = await lightrag_server.get_rag_instance()
            
            # Delete from LightRAG
            await rag.adelete_by_doc_id(docs_to_delete)
            
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
        # Get the RAG instance
        rag = await lightrag_server.get_rag_instance()
        
        # Delete from LightRAG
        await rag.adelete_by_doc_id(request.doc_ids)
        
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

# Override the standard /documents/text endpoint to fix file_path issue
original_insert_text = None
for route in app.routes:
    if route.path == "/documents/text" and route.methods == {"POST"}:
        original_insert_text = route.endpoint
        app.routes.remove(route)
        break

@app.post("/documents/text")
async def insert_text_fixed(request: InsertDocumentsRequest):
    """Fixed text insertion that ensures file_path is set"""
    try:
        texts = request.texts
        file_paths = []
        
        # Generate file paths for each text
        for text in texts:
            doc_id = compute_doc_id(text)
            file_path = f"text/{doc_id}.txt"
            file_paths.append(file_path)
            
            # Store basic metadata
            metadata_store[doc_id] = {
                "id": doc_id,
                "file_path": file_path,
                "description": request.description,
                "indexed_at": datetime.utcnow().isoformat(),
                "content_summary": text[:200] + "..." if len(text) > 200 else text
            }
        
        # Save metadata
        save_metadata()
        
        # Get the RAG instance
        rag = await lightrag_server.get_rag_instance()
        
        # Insert with file paths
        await rag.ainsert(texts, file_paths=file_paths)
        
        return InsertDocumentsResponse(
            status="success",
            message=f"Successfully inserted {len(texts)} documents"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Run the standard LightRAG server with our extensions
    uvicorn.run(
        "lightrag_extended_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "9621")),
        reload=False
    )