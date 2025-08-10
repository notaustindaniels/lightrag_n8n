import os
import json
import time
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from lightrag import QueryParam

router = APIRouter()

class FilteredQueryRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None
    mode: str = "hybrid"
    stream: bool = False
    only_need_context: bool = False
    response_type: str = "default"
    top_k: int = 50
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

class SourceListResponse(BaseModel):
    sources: List[Dict[str, Any]]
    total: int

def get_workspace_instances():
    from lightrag_extended_api import workspace_instances, WorkspaceManager
    return workspace_instances, WorkspaceManager

def get_workspace_metadata():
    from lightrag_extended_api import workspace_metadata
    return workspace_metadata

def get_default_workspace():
    from lightrag_extended_api import default_workspace
    return default_workspace

# Compatibility functions for existing code
def get_rag_instance():
    """Get default workspace RAG instance for backward compatibility"""
    from lightrag_extended_api import workspace_instances, default_workspace
    return workspace_instances.get(default_workspace)

def get_metadata_store():
    """Get default workspace metadata for backward compatibility"""
    from lightrag_extended_api import workspace_metadata, default_workspace
    return workspace_metadata.get(default_workspace, {})

def get_helper_functions():
    from lightrag_extended_api import (
        save_workspace_metadata,
        save_metadata_store,
        get_graph_from_storage,
        force_save_graph_to_disk,
        cleanup_all_document_traces
    )
    return {
        'save_workspace_metadata': save_workspace_metadata,
        'save_metadata_store': save_metadata_store,
        'get_graph_from_storage': get_graph_from_storage,
        'force_save_graph_to_disk': force_save_graph_to_disk,
        'cleanup_all_document_traces': cleanup_all_document_traces
    }

@router.get("/documents/sources")
async def list_document_sources(workspace: Optional[str] = None):
    """
    List all available documentation sources/libraries (workspaces)
    Each source now represents a separate workspace with isolated data
    
    Args:
        workspace: Optional workspace name to filter sources from
    """
    try:
        workspace_instances, WorkspaceManager = get_workspace_instances()
        workspace_metadata = get_workspace_metadata()
        
        sources_map = {}
        
        # If workspace specified, only list that workspace
        workspaces_to_list = [workspace] if workspace else WorkspaceManager.list_workspaces()
        
        # List workspaces as sources
        for workspace_name in workspaces_to_list:
            # Skip if workspace doesn't exist when filtered
            if workspace and workspace_name not in WorkspaceManager.list_workspaces():
                continue
                
            metadata = workspace_metadata.get(workspace_name, {})
            
            sources_map[workspace_name] = {
                "source": workspace_name,
                "workspace": workspace_name,  # Explicit workspace field
                "document_count": len(metadata),
                "example_files": [],
                "description": f"Workspace: {workspace_name}",
                "last_indexed": ""
            }
            
            # Get example files and last indexed time from metadata
            for doc_id, doc_metadata in list(metadata.items())[:5]:
                filename = doc_id.split(']', 1)[1].strip() if ']' in doc_id else doc_id
                sources_map[workspace_name]["example_files"].append(filename)
                
                indexed_at = doc_metadata.get('indexed_at', '')
                if indexed_at and indexed_at > sources_map[workspace_name]["last_indexed"]:
                    sources_map[workspace_name]["last_indexed"] = indexed_at
        
        sources_list = list(sources_map.values())
        sources_list.sort(key=lambda x: x["document_count"], reverse=True)
        
        return SourceListResponse(
            sources=sources_list,
            total=len(sources_list)
        )
        
    except Exception as e:
        print(f"Error listing sources: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/sources")
async def debug_sources():
    """Debug endpoint to understand why sources aren't being detected"""
    rag_instance = get_rag_instance()
    metadata_store = get_metadata_store()
    
    debug_info = {
        "metadata_store_count": len(metadata_store),
        "metadata_store_sample": [],
        "doc_status_methods": [],
        "doc_status_sample": [],
        "sources_found": []
    }
    
    for i, (doc_id, metadata) in enumerate(metadata_store.items()):
        if i < 5:
            debug_info["metadata_store_sample"].append({
                "doc_id": doc_id,
                "has_bracket": doc_id.startswith('[') and ']' in doc_id,
                "file_path": metadata.get('file_path', 'N/A')
            })
    
    if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
        doc_status_storage = rag_instance.doc_status
        
        if hasattr(doc_status_storage, 'get_all'):
            try:
                all_docs = await doc_status_storage.get_all()
                debug_info["doc_status_methods"].append({
                    "method": "get_all",
                    "success": True,
                    "doc_count": len(all_docs) if all_docs else 0,
                    "type": type(all_docs).__name__ if all_docs else "None"
                })
                
                if all_docs:
                    for i, (doc_id, doc_data) in enumerate(all_docs.items()):
                        if i < 5:
                            debug_info["doc_status_sample"].append({
                                "doc_id": doc_id,
                                "has_bracket": doc_id.startswith('[') and ']' in doc_id,
                                "source": doc_id.split(']')[0][1:] if doc_id.startswith('[') and ']' in doc_id else "N/A"
                            })
                            
                            if doc_id.startswith('[') and ']' in doc_id:
                                source = doc_id.split(']')[0][1:]
                                if source not in debug_info["sources_found"]:
                                    debug_info["sources_found"].append(source)
                                    
            except Exception as e:
                debug_info["doc_status_methods"].append({
                    "method": "get_all",
                    "success": False,
                    "error": str(e)
                })
        
        if hasattr(doc_status_storage, '_data'):
            try:
                storage_data = doc_status_storage._data
                debug_info["doc_status_methods"].append({
                    "method": "_data",
                    "success": True,
                    "type": type(storage_data).__name__,
                    "is_dict": isinstance(storage_data, dict),
                    "doc_count": len(storage_data) if isinstance(storage_data, dict) else "N/A"
                })
            except Exception as e:
                debug_info["doc_status_methods"].append({
                    "method": "_data",
                    "success": False,
                    "error": str(e)
                })
        
        try:
            working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
            doc_status_file = os.path.join(working_dir, "doc_status.json")
            debug_info["doc_status_file_exists"] = os.path.exists(doc_status_file)
            
            if os.path.exists(doc_status_file):
                with open(doc_status_file, 'r') as f:
                    file_data = json.load(f)
                debug_info["doc_status_methods"].append({
                    "method": "json_file",
                    "success": True,
                    "doc_count": len(file_data) if isinstance(file_data, dict) else "N/A"
                })
        except Exception as e:
            debug_info["doc_status_methods"].append({
                "method": "json_file",
                "success": False,
                "error": str(e)
            })
    
    return debug_info

@router.get("/documents/by-source/{source}")
async def get_documents_by_source(source: str):
    """
    Get all documents for a specific source/library
    
    Following the same pattern as /documents/by-sitemap/{sitemap_url}
    
    Args:
        source: The source name (without brackets)
    
    Returns:
        List of documents from that source
    """
    try:
        rag_instance = get_rag_instance()
        metadata_store = get_metadata_store()
        
        matching_docs = []
        source_pattern = f"[{source}]"
        
        for doc_id, metadata in metadata_store.items():
            if doc_id.startswith(source_pattern):
                filename = doc_id[len(source_pattern):].strip()
                
                matching_docs.append({
                    "id": doc_id,
                    "doc_id": doc_id,
                    "file_path": metadata.get('file_path', f"text/{doc_id}.txt"),
                    "filename": filename,
                    "display_name": metadata.get('display_name', doc_id),
                    "indexed_at": metadata.get('indexed_at'),
                    "status": "processed"
                })
            else:
                file_path = metadata.get('file_path', '')
                if file_path.startswith(source_pattern):
                    filename = file_path[len(source_pattern):].strip()
                    
                    matching_docs.append({
                        "id": doc_id,
                        "doc_id": doc_id,
                        "file_path": file_path,
                        "filename": filename,
                        "display_name": metadata.get('display_name', file_path),
                        "indexed_at": metadata.get('indexed_at'),
                        "status": "processed"
                    })
        
        processed_doc_ids = {doc["doc_id"] for doc in matching_docs}
        
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            try:
                doc_status_storage = rag_instance.doc_status
                all_docs = None
                
                if hasattr(doc_status_storage, 'get_all'):
                    try:
                        all_docs = await doc_status_storage.get_all()
                    except Exception as e:
                        print(f"Error with get_all method: {e}")
                
                if not all_docs and hasattr(doc_status_storage, '_data'):
                    try:
                        storage_data = doc_status_storage._data
                        if isinstance(storage_data, dict):
                            all_docs = storage_data
                    except Exception as e:
                        print(f"Error accessing _data: {e}")
                
                if not all_docs:
                    try:
                        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                        doc_status_file = os.path.join(working_dir, "doc_status.json")
                        if os.path.exists(doc_status_file):
                            with open(doc_status_file, 'r') as f:
                                all_docs = json.load(f)
                    except Exception as e:
                        print(f"Error reading doc_status.json: {e}")
                
                if all_docs:
                    for doc_id, doc_data in all_docs.items():
                        if doc_id in processed_doc_ids:
                            continue
                        
                        if doc_id.startswith(source_pattern) or doc_id.startswith(source_pattern + " "):
                            filename = doc_id[len(source_pattern):].strip()
                            
                            matching_docs.append({
                                "id": doc_id,
                                "doc_id": doc_id,
                                "file_path": doc_id,
                                "filename": filename,
                                "display_name": doc_id,
                                "indexed_at": doc_data.get('indexed_at', '') if isinstance(doc_data, dict) else '',
                                "status": doc_data.get('status', 'processed') if isinstance(doc_data, dict) else 'processed'
                            })
            except Exception as e:
                print(f"Error checking doc_status: {e}")
        
        matching_docs.sort(key=lambda x: x["filename"])
        
        return {
            "source": source,
            "documents": matching_docs,
            "count": len(matching_docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/by-source/{source}")
async def delete_documents_by_source(source: str):
    """
    Delete all documents for a specific source
    
    Following the same pattern as DELETE /documents/by-sitemap/{sitemap_url}
    
    Args:
        source: The source name (without brackets)
    
    Returns:
        Deletion status
    """
    try:
        rag_instance = get_rag_instance()
        metadata_store = get_metadata_store()
        helpers = get_helper_functions()
        
        docs_to_delete = []
        metadata_keys_to_delete = []
        source_pattern = f"[{source}]"
        
        for doc_id, metadata in metadata_store.items():
            if doc_id.startswith(source_pattern):
                metadata_keys_to_delete.append(doc_id)
                docs_to_delete.append(doc_id)
                
                original_id = metadata.get('original_doc_id')
                if original_id and original_id != doc_id:
                    docs_to_delete.append(original_id)
            else:
                file_path = metadata.get('file_path', '')
                if file_path.startswith(source_pattern):
                    metadata_keys_to_delete.append(doc_id)
                    docs_to_delete.append(doc_id)
                    
                    original_id = metadata.get('original_doc_id', doc_id)
                    if original_id not in docs_to_delete:
                        docs_to_delete.append(original_id)
                    if file_path not in docs_to_delete:
                        docs_to_delete.append(file_path)
        
        if docs_to_delete:
            unique_docs_to_delete = list(set(docs_to_delete))
            
            await helpers['cleanup_all_document_traces'](unique_docs_to_delete)
            
            try:
                await rag_instance.adelete_by_doc_id(unique_docs_to_delete)
            except Exception as e:
                print(f"Error in adelete_by_doc_id: {e}")
            
            for key in metadata_keys_to_delete:
                if key in metadata_store:
                    del metadata_store[key]
            
            helpers['save_metadata_store']()
            
            working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = helpers['get_graph_from_storage'](graph_storage)
                if graph:
                    helpers['force_save_graph_to_disk'](graph, working_dir)
            
            time.sleep(0.5)
        
        return {
            "status": "success",
            "message": f"Deleted {len(metadata_keys_to_delete)} documents for source {source}",
            "deleted_count": len(metadata_keys_to_delete),
            "deleted_ids": unique_docs_to_delete if 'unique_docs_to_delete' in locals() else [],
            "source": source
        }
        
    except Exception as e:
        print(f"Error in delete_documents_by_source: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_with_optional_filtering(request: FilteredQueryRequest):
    """
    Query the RAG system with optional workspace (source) filtering
    """
    try:
        workspace_instances, WorkspaceManager = get_workspace_instances()
        default_workspace = get_default_workspace()
        
        if not request.sources:
            # Query the default workspace if no sources specified
            rag = await WorkspaceManager.get_or_create_instance(default_workspace)
            param = QueryParam(
                mode=request.mode,
                stream=request.stream
            )
            
            result = await rag.aquery(request.query, param=param)
            
            return {
                "query": request.query,
                "response": result,
                "mode": request.mode,
                "workspace": default_workspace
            }
        
        # Query specific workspaces (sources)
        results = []
        for source in request.sources:
            # Each source is a workspace
            if source not in WorkspaceManager.list_workspaces():
                continue
            
            rag = await WorkspaceManager.get_or_create_instance(source)
            param = QueryParam(
                mode=request.mode,
                stream=request.stream
            )
            
            # Query this workspace
            result = await rag.aquery(request.query, param=param)
            results.append({
                "workspace": source,
                "response": result
            })
        
        if not results:
            return {
                "query": request.query,
                "response": f"No documents found for the specified sources: {', '.join(request.sources)}",
                "mode": request.mode,
                "sources_filtered": request.sources,
                "documents_found": 0
            }
        
        # If single workspace, return its result directly
        if len(results) == 1:
            return {
                "query": request.query,
                "response": results[0]["response"],
                "mode": request.mode,
                "workspace": results[0]["workspace"],
                "sources_filtered": request.sources
            }
        
        # For multiple workspaces, combine results
        combined_response = "\n\n".join([
            f"[From {r['workspace']}]:\n{r['response']}" 
            for r in results
        ])
        
        return {
            "query": request.query,
            "response": combined_response,
            "mode": request.mode,
            "sources_filtered": request.sources,
            "workspaces_queried": [r["workspace"] for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def query_stream_with_optional_filtering(request: FilteredQueryRequest):
    """
    Stream query responses with optional source filtering
    
    This extends the standard /query/stream endpoint to support optional source filtering.
    """
    try:
        rag_instance = get_rag_instance()
        
        request.stream = True
        
        if not request.sources:
            param = QueryParam(
                mode=request.mode,
                stream=True,
                only_need_context=request.only_need_context,
                response_type=request.response_type,
                top_k=request.top_k,
                max_token_for_text_unit=request.max_token_for_text_unit,
                max_token_for_global_context=request.max_token_for_global_context,
                max_token_for_local_context=request.max_token_for_local_context
            )
            
            result = await rag_instance.aquery(request.query, param=param)
            
            return {
                "query": request.query,
                "response": result,
                "mode": request.mode,
                "stream": True
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/sources/{source}/stats")
async def get_source_graph_statistics(source: str):
    """
    Get graph statistics for a specific source
    
    Following the pattern of graph endpoints like /graph/label/list
    
    Returns information about the knowledge graph coverage for this source.
    """
    try:
        metadata_store = get_metadata_store()
        
        source_pattern = f"[{source}]"
        stats = {
            "source": source,
            "document_count": 0,
            "total_chunks": 0,
            "entities_count": 0,
            "relationships_count": 0,
            "file_types": {},
            "latest_update": None,
            "oldest_document": None
        }
        
        doc_ids_in_source = set()
        
        for doc_id, metadata in metadata_store.items():
            if doc_id.startswith(source_pattern):
                doc_ids_in_source.add(doc_id)
                stats["document_count"] += 1
                
                file_path = metadata.get('file_path', doc_id)
                if '.' in file_path:
                    ext = file_path.split('.')[-1].lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                
                indexed_at = metadata.get('indexed_at', '')
                if indexed_at:
                    if not stats["latest_update"] or indexed_at > stats["latest_update"]:
                        stats["latest_update"] = indexed_at
                    if not stats["oldest_document"] or indexed_at < stats["oldest_document"]:
                        stats["oldest_document"] = indexed_at
            else:
                file_path = metadata.get('file_path', '')
                if file_path.startswith(source_pattern):
                    doc_ids_in_source.add(doc_id)
                    stats["document_count"] += 1
                    
                    if '.' in file_path:
                        ext = file_path.split('.')[-1].lower()
                        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
                    indexed_at = metadata.get('indexed_at', '')
                    if indexed_at:
                        if not stats["latest_update"] or indexed_at > stats["latest_update"]:
                            stats["latest_update"] = indexed_at
                        if not stats["oldest_document"] or indexed_at < stats["oldest_document"]:
                            stats["oldest_document"] = indexed_at
        
        return stats
        
    except Exception as e:
        print(f"Error getting source stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))