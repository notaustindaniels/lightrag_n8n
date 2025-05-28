#!/usr/bin/env python3
"""
Extended LightRAG Server - Runs the original server with additional endpoints
This preserves the web UI while adding enhanced document management
"""
import os
import sys
import asyncio
import hashlib
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# First, set up environment variables that the LightRAG server expects
os.environ.setdefault("WORKING_DIR", "/app/data/rag_storage")
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "9621")

# Import the original LightRAG server
try:
    # Import the server module
    import lightrag.api.lightrag_server as server_module
    from fastapi import HTTPException
    from pydantic import BaseModel, Field
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

def add_custom_endpoints(app, rag_instance):
    """Add our custom endpoints to the existing app"""
    
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
    
    # Override the existing /documents endpoint to handle file_path properly
    @app.api_route("/documents", methods=["GET"], include_in_schema=False)
    async def get_documents_extended():
        """Get all documents with proper file_path handling"""
        try:
            # Get the original response
            from lightrag.api.lightrag_server import get_all_docs_with_status
            result = await get_all_docs_with_status()
            
            # Ensure all documents have file_path
            if "statuses" in result:
                for status_type in ["processed", "pending", "failed"]:
                    if status_type in result["statuses"]:
                        for doc in result["statuses"][status_type]:
                            if not doc.get("file_path"):
                                # Try to get from metadata store
                                doc_id = doc.get("id")
                                if doc_id and doc_id in metadata_store:
                                    doc["file_path"] = metadata_store[doc_id].get("file_path", f"text/{doc_id}.txt")
                                else:
                                    doc["file_path"] = f"text/{doc.get('id', 'unknown')}.txt"
            
            return result
            
        except Exception as e:
            # If the original endpoint fails, provide a fallback
            return {
                "statuses": {
                    "processed": [],
                    "pending": [],
                    "failed": []
                },
                "total": 0
            }
    
    print("Custom endpoints added to LightRAG server")

if __name__ == "__main__":
    # Load metadata
    load_metadata()
    
    # Get the app and rag instance from the server module
    # The lightrag server creates these when imported
    app = server_module.app
    rag_instance = server_module.rag_instance
    
    # Add our custom endpoints
    add_custom_endpoints(app, rag_instance)
    
    # Run the server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    
    print(f"Starting Extended LightRAG Server on {host}:{port}")
    print("Web UI will be available at the root URL")
    
    uvicorn.run(app, host=host, port=port)


    #This is a test
    