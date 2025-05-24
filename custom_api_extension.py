#!/usr/bin/env python3
"""
Extended LightRAG API Server with Custom Python API Endpoints
This demonstrates how to add Python API functionality to the REST API server
"""
import os
import sys
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Import the original API server components
# This allows us to reuse existing endpoints
sys.path.append('/tmp/lightrag')
from lightrag.api.main import app as original_app
from lightrag.api.routers import management, query

# Setup logging
setup_logger("lightrag", level="INFO")

# Configuration
WORKING_DIR = os.getenv("WORKING_DIR", "/app/data/rag_storage")
INPUT_DIR = os.getenv("INPUT_DIR", "/app/data/inputs")

# Create extended app
app = FastAPI(
    title="Extended LightRAG API",
    description="LightRAG API with additional Python API functionality",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance = None

# Request/Response Models
class DeleteDocumentRequest(BaseModel):
    doc_ids: List[str]
    
class DeleteDocumentResponse(BaseModel):
    status: str
    message: str
    deleted_ids: List[str]

class EntityMergeRequest(BaseModel):
    source_entities: List[str]
    target_entity: str
    merge_strategy: Optional[dict] = None
    target_entity_data: Optional[dict] = None

class EntityEditRequest(BaseModel):
    entity_name: str
    updated_data: dict
    
class RelationEditRequest(BaseModel):
    src_id: str
    tgt_id: str
    updated_data: dict

class DeleteByEntityRequest(BaseModel):
    entity_names: List[str]

# Dependency to get RAG instance
async def get_rag():
    global rag_instance
    if rag_instance is None:
        rag_instance = await initialize_rag()
    return rag_instance

# Initialize RAG
async def initialize_rag():
    """Initialize the LightRAG instance with configuration"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# Mount original API routes
app.mount("/api/v1", original_app)

# Custom endpoints using Python API
@app.delete("/documents/by-id", response_model=DeleteDocumentResponse)
async def delete_documents_by_id(
    request: DeleteDocumentRequest,
    rag: LightRAG = Depends(get_rag)
):
    """
    Delete documents by their IDs using the Python API's delete_by_doc_id method
    
    This endpoint exposes the Python API functionality that isn't available
    in the standard REST API.
    """
    try:
        # Use the Python API method
        await rag.adelete_by_doc_id(request.doc_ids)
        
        return DeleteDocumentResponse(
            status="success",
            message=f"Successfully deleted {len(request.doc_ids)} documents",
            deleted_ids=request.doc_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entities/merge")
async def merge_entities(
    request: EntityMergeRequest,
    rag: LightRAG = Depends(get_rag)
):
    """
    Merge multiple entities into a single target entity
    
    Uses the Python API's merge_entities method
    """
    try:
        await rag.amerge_entities(
            source_entities=request.source_entities,
            target_entity=request.target_entity,
            merge_strategy=request.merge_strategy,
            target_entity_data=request.target_entity_data
        )
        
        return {
            "status": "success",
            "message": f"Successfully merged {len(request.source_entities)} entities into {request.target_entity}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/entities")
async def edit_entity(
    request: EntityEditRequest,
    rag: LightRAG = Depends(get_rag)
):
    """
    Edit an existing entity's attributes
    
    Uses the Python API's edit_entity method
    """
    try:
        result = await rag.aedit_entity(
            entity_name=request.entity_name,
            updated_data=request.updated_data
        )
        
        return {
            "status": "success",
            "message": f"Successfully updated entity {request.entity_name}",
            "entity": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/relations")
async def edit_relation(
    request: RelationEditRequest,
    rag: LightRAG = Depends(get_rag)
):
    """
    Edit a relation between two entities
    
    Uses the Python API's edit_relation method
    """
    try:
        result = await rag.aedit_relation(
            src_id=request.src_id,
            tgt_id=request.tgt_id,
            updated_data=request.updated_data
        )
        
        return {
            "status": "success",
            "message": f"Successfully updated relation between {request.src_id} and {request.tgt_id}",
            "relation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/entities/by-name")
async def delete_by_entity(
    request: DeleteByEntityRequest,
    rag: LightRAG = Depends(get_rag)
):
    """
    Delete entities by their names
    
    Uses the Python API's delete_by_entity method
    """
    try:
        await rag.adelete_by_entity(request.entity_names)
        
        return {
            "status": "success",
            "message": f"Successfully deleted {len(request.entity_names)} entities"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache(
    modes: Optional[List[str]] = None,
    rag: LightRAG = Depends(get_rag)
):
    """
    Clear LightRAG cache
    
    Uses the Python API's clear_cache method
    """
    try:
        await rag.aclear_cache(modes=modes)
        
        return {
            "status": "success",
            "message": f"Successfully cleared cache for modes: {modes if modes else 'all'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/python-api/status")
async def python_api_status():
    """Check if Python API extensions are available"""
    return {
        "status": "active",
        "available_endpoints": [
            "DELETE /documents/by-id",
            "POST /entities/merge",
            "PATCH /entities",
            "PATCH /relations",
            "DELETE /entities/by-name",
            "POST /cache/clear"
        ],
        "message": "Python API extensions are active"
    }

# Main entry point
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Run the extended server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 9621)),
        log_level="info"
    )