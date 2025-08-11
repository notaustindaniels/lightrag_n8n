"""
Enhanced query route with comprehensive logging for debugging LightRAG hybrid mode
"""
import json
import logging
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from lightrag import QueryParam

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

class EnhancedQueryRequest(BaseModel):
    query: str
    workspace: Optional[str] = None
    mode: str = "hybrid"
    debug: bool = False
    stream: bool = False
    top_k: int = 10
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

class DebugQueryResponse(BaseModel):
    response: str
    mode: str
    workspace: str
    debug_info: Optional[Dict[str, Any]] = None

@router.post("/query/debug")
async def debug_query(request: EnhancedQueryRequest):
    """
    Enhanced query endpoint with debugging information
    """
    from lightrag_extended_api import workspace_instances, WorkspaceManager, default_workspace
    
    debug_info = {
        "request": request.dict(),
        "workspace_status": {},
        "retrieval_stats": {},
        "context_preview": {},
        "errors": []
    }
    
    try:
        # 1. Get workspace instance
        workspace_name = request.workspace or default_workspace
        
        if workspace_name not in workspace_instances:
            await WorkspaceManager.ensure_workspace_exists(workspace_name)
        
        rag = workspace_instances.get(workspace_name)
        if not rag:
            debug_info["errors"].append(f"Failed to get RAG instance for workspace: {workspace_name}")
            raise HTTPException(status_code=404, detail=f"Workspace {workspace_name} not found")
        
        # 2. Check workspace status
        debug_info["workspace_status"] = {
            "name": workspace_name,
            "has_entities": hasattr(rag, 'entities_vdb') and rag.entities_vdb is not None,
            "has_relationships": hasattr(rag, 'relationships_vdb') and rag.relationships_vdb is not None,
            "has_chunks": hasattr(rag, 'chunks_vdb') and rag.chunks_vdb is not None,
            "has_graph": hasattr(rag, 'chunk_entity_relation_graph') and rag.chunk_entity_relation_graph is not None
        }
        
        # 3. Create query parameters
        query_param = QueryParam(
            mode=request.mode,
            stream=False,
            top_k=request.top_k,
            max_token_for_text_unit=request.max_token_for_text_unit,
            max_token_for_global_context=request.max_token_for_global_context,
            max_token_for_local_context=request.max_token_for_local_context
        )
        
        # 4. Get context only first (for debugging)
        if request.debug:
            query_param.only_need_context = True
            context = await rag.aquery(request.query, param=query_param)
            
            if context:
                debug_info["context_preview"] = {
                    "length": len(str(context)),
                    "preview": str(context)[:500] + "..." if len(str(context)) > 500 else str(context),
                    "has_entities": "entities" in str(context).lower(),
                    "has_relationships": "relationship" in str(context).lower()
                }
            else:
                debug_info["context_preview"] = {"error": "No context retrieved"}
            
            # Try to get entity counts
            try:
                if rag.entities_vdb:
                    entities = await rag.entities_vdb.query(request.query, top_k=5)
                    debug_info["retrieval_stats"]["entities_found"] = len(entities) if entities else 0
                    debug_info["retrieval_stats"]["entity_samples"] = [str(e)[:100] for e in (entities[:3] if entities else [])]
            except Exception as e:
                debug_info["errors"].append(f"Entity retrieval error: {str(e)}")
            
            # Try to get relationship counts
            try:
                if rag.relationships_vdb:
                    relationships = await rag.relationships_vdb.query(request.query, top_k=5)
                    debug_info["retrieval_stats"]["relationships_found"] = len(relationships) if relationships else 0
                    debug_info["retrieval_stats"]["relationship_samples"] = [str(r)[:100] for r in (relationships[:3] if relationships else [])]
            except Exception as e:
                debug_info["errors"].append(f"Relationship retrieval error: {str(e)}")
            
            # Try to get chunk counts
            try:
                if rag.chunks_vdb:
                    chunks = await rag.chunks_vdb.query(request.query, top_k=5)
                    debug_info["retrieval_stats"]["chunks_found"] = len(chunks) if chunks else 0
                    debug_info["retrieval_stats"]["chunk_samples"] = [str(c)[:100] for c in (chunks[:3] if chunks else [])]
            except Exception as e:
                debug_info["errors"].append(f"Chunk retrieval error: {str(e)}")
        
        # 5. Get the actual response
        query_param.only_need_context = False
        response = await rag.aquery(request.query, param=query_param)
        
        # 6. Analyze response
        if request.debug:
            debug_info["response_analysis"] = {
                "length": len(str(response)),
                "is_generic": "I don't have" in str(response) or "No information" in str(response),
                "mentions_entities": any(entity in str(response) for entity in ["LightRAG", "Python", "Knowledge Graph"]),
                "response_preview": str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
            }
        
        return DebugQueryResponse(
            response=response,
            mode=request.mode,
            workspace=workspace_name,
            debug_info=debug_info if request.debug else None
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        debug_info["errors"].append(f"Main error: {str(e)}")
        
        if request.debug:
            raise HTTPException(status_code=500, detail=json.dumps(debug_info, indent=2))
        else:
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/workspaces/stats")
async def get_workspace_stats():
    """
    Get statistics for all workspaces
    """
    from lightrag_extended_api import workspace_instances, workspace_metadata
    
    stats = {}
    
    for workspace_name, rag in workspace_instances.items():
        workspace_stats = {
            "documents": len(workspace_metadata.get(workspace_name, {})),
            "storage_status": {
                "has_entities": False,
                "has_relationships": False,
                "has_chunks": False,
                "entity_count": 0,
                "relationship_count": 0,
                "chunk_count": 0
            }
        }
        
        # Try to get entity count
        try:
            if hasattr(rag, 'entities_vdb') and rag.entities_vdb:
                if hasattr(rag.entities_vdb, '_data'):
                    workspace_stats["storage_status"]["entity_count"] = len(rag.entities_vdb._data)
                    workspace_stats["storage_status"]["has_entities"] = len(rag.entities_vdb._data) > 0
        except:
            pass
        
        # Try to get relationship count
        try:
            if hasattr(rag, 'relationships_vdb') and rag.relationships_vdb:
                if hasattr(rag.relationships_vdb, '_data'):
                    workspace_stats["storage_status"]["relationship_count"] = len(rag.relationships_vdb._data)
                    workspace_stats["storage_status"]["has_relationships"] = len(rag.relationships_vdb._data) > 0
        except:
            pass
        
        # Try to get chunk count
        try:
            if hasattr(rag, 'chunks_vdb') and rag.chunks_vdb:
                if hasattr(rag.chunks_vdb, '_data'):
                    workspace_stats["storage_status"]["chunk_count"] = len(rag.chunks_vdb._data)
                    workspace_stats["storage_status"]["has_chunks"] = len(rag.chunks_vdb._data) > 0
        except:
            pass
        
        stats[workspace_name] = workspace_stats
    
    return stats