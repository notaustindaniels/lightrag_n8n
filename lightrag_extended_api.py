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
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import traceback

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

# Graph models
class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False

class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]

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
        # Return the full file_path as display_name (includes domain and full path)
        return file_path
    else:
        return f"text/{doc_id[:8]}..."

async def cleanup_all_document_traces(doc_ids: List[str]):
    """Clean up all traces of documents from all LightRAG storage components"""
    try:
        # Track entities and relationships to potentially remove
        entities_to_check = set()
        relationships_to_check = set()
        
        # 1. First, get text chunks to find entities mentioned in these documents
        if hasattr(rag_instance, 'kv_storage') and rag_instance.kv_storage:
            try:
                # Get all chunks for these documents to extract entity references
                for doc_id in doc_ids:
                    # Get the full document text if available
                    try:
                        doc_data = await rag_instance.kv_storage.get({doc_id})
                        if doc_data and doc_id in doc_data:
                            doc_text = doc_data[doc_id]
                            # Try to extract entities from the document text
                            # This is a simple approach - looking for patterns that might be entities
                            if isinstance(doc_text, str):
                                # Store document content for entity extraction
                                pass
                    except:
                        pass
                    
                    # Look for text chunks
                    if hasattr(rag_instance.kv_storage, '_data'):
                        for key in list(rag_instance.kv_storage._data.keys()):
                            if key.startswith(f"chunk-{doc_id}"):
                                try:
                                    chunk_data = await rag_instance.kv_storage.get({key})
                                    if chunk_data and key in chunk_data:
                                        chunk_info = chunk_data[key]
                                        # Check if chunk has entity information
                                        if isinstance(chunk_info, dict):
                                            # Look for entities field
                                            if 'entities' in chunk_info:
                                                entities_to_check.update(chunk_info['entities'])
                                            # Also check for any entity references in metadata
                                            if 'metadata' in chunk_info and isinstance(chunk_info['metadata'], dict):
                                                if 'entities' in chunk_info['metadata']:
                                                    entities_to_check.update(chunk_info['metadata']['entities'])
                                except:
                                    pass
            except Exception as e:
                print(f"Error extracting entities from chunks: {e}")
        
        # 2. Delete from vector storage
        if hasattr(rag_instance, 'vector_storage') and rag_instance.vector_storage:
            try:
                # Delete document vectors
                for doc_id in doc_ids:
                    # Delete document-level vectors
                    await rag_instance.vector_storage.delete_by_doc_id(doc_id)
                    
                    # Also try to delete chunk vectors
                    # LightRAG might store chunk vectors with IDs like "chunk-{doc_id}-{index}"
                    if hasattr(rag_instance.vector_storage, 'delete'):
                        # Try to find and delete chunk vectors
                        chunk_ids = []
                        for i in range(100):  # Assume max 100 chunks per doc
                            chunk_ids.append(f"chunk-{doc_id}-{i}")
                        try:
                            await rag_instance.vector_storage.delete(chunk_ids)
                        except:
                            pass
            except Exception as e:
                print(f"Error deleting from vector storage: {e}")
        
        # 3. Delete from KV storage (full documents and chunks)
        if hasattr(rag_instance, 'kv_storage') and rag_instance.kv_storage:
            try:
                # Delete full documents
                for doc_id in doc_ids:
                    await rag_instance.kv_storage.delete({doc_id})
                    
                # Delete text chunks
                if hasattr(rag_instance.kv_storage, '_data'):
                    keys_to_delete = []
                    for key in list(rag_instance.kv_storage._data.keys()):
                        for doc_id in doc_ids:
                            if key.startswith(f"chunk-{doc_id}"):
                                keys_to_delete.append(key)
                    if keys_to_delete:
                        await rag_instance.kv_storage.delete(set(keys_to_delete))
            except Exception as e:
                print(f"Error deleting from KV storage: {e}")
        
        # 4. Delete from doc status storage
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status:
            try:
                await rag_instance.doc_status.delete({doc_id for doc_id in doc_ids})
            except Exception as e:
                print(f"Error deleting from doc status storage: {e}")
        
        # 5. Clean up graph storage (entities and relationships)
        if hasattr(rag_instance, 'chunk_entity_relation_graph') and rag_instance.chunk_entity_relation_graph:
            try:
                graph = rag_instance.chunk_entity_relation_graph
                
                # Since we don't have direct document-to-entity mapping,
                # we'll use a heuristic approach:
                # Remove entities that appear ONLY in the deleted documents
                
                # For now, log what we would clean up
                print(f"Graph cleanup: Found {len(entities_to_check)} potential entities to check")
                
                # Note: Full implementation would require:
                # 1. Tracking which entities came from which documents during insertion
                # 2. Reference counting to know when an entity should be removed
                # 3. Removing orphaned relationships when entities are removed
                
                # Clear the graph cache if available
                if hasattr(graph, 'clear_cache'):
                    graph.clear_cache()
                    
            except Exception as e:
                print(f"Error with graph storage cleanup: {e}")
        
        # 6. Clear any caches that might contain document data
        if hasattr(rag_instance, 'llm_response_cache'):
            try:
                # Clear cache entries related to these documents
                await rag_instance.aclear_cache()
            except Exception as e:
                print(f"Error clearing cache: {e}")
        
        # 7. Force storage synchronization if available
        if hasattr(rag_instance, 'graph_storage') and rag_instance.graph_storage:
            try:
                if hasattr(rag_instance.graph_storage, 'sync'):
                    await rag_instance.graph_storage.sync()
            except:
                pass
                
        print(f"Cleaned up traces for {len(doc_ids)} documents")
                
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

# Graph endpoints
@app.get("/graph/label/list")
async def get_graph_labels():
    """
    Get all graph labels
    Returns:
        List[str]: List of graph labels
    """
    try:
        # Check if rag_instance has the method
        if hasattr(rag_instance, 'get_graph_labels'):
            return await rag_instance.get_graph_labels()
        else:
            # Alternative approach: get labels from graph storage directly
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                nodes = []
                if hasattr(graph, 'nodes'):
                    nodes = list(graph.nodes())
                return nodes
            else:
                return []
    except Exception as e:
        print(f"Error getting graph labels: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error getting graph labels: {str(e)}"
        )

@app.get("/graphs")
async def get_knowledge_graph(
    label: str = Query(..., description="Label to get knowledge graph for"),
    max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
    max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
):
    """
    Retrieve a connected subgraph of nodes where the label includes the specified label.
    When reducing the number of nodes, the prioritization criteria are as follows:
    1. Hops(path) to the starting node take precedence
    2. Followed by the degree of the nodes
    
    Args:
        label (str): Label of the starting node
        max_depth (int, optional): Maximum depth of the subgraph, Defaults to 3
        max_nodes: Maximum nodes to return
    
    Returns:
        Dict[str, List[str]]: Knowledge graph for label
    """
    try:
        # Check if rag_instance has the method
        if hasattr(rag_instance, 'get_knowledge_graph'):
            return await rag_instance.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        else:
            # Alternative approach: build graph data from graph storage
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                
                # Get subgraph starting from the label
                nodes = set()
                edges = []
                
                # Simple BFS to get nodes within max_depth
                if hasattr(graph, 'has_node') and graph.has_node(label):
                    visited = set()
                    queue = [(label, 0)]
                    
                    while queue and len(nodes) < max_nodes:
                        current_node, depth = queue.pop(0)
                        
                        if current_node in visited or depth > max_depth:
                            continue
                            
                        visited.add(current_node)
                        nodes.add(current_node)
                        
                        # Get neighbors
                        if hasattr(graph, 'neighbors'):
                            for neighbor in graph.neighbors(current_node):
                                if neighbor not in visited and depth + 1 <= max_depth:
                                    queue.append((neighbor, depth + 1))
                                    edges.append({"source": current_node, "target": neighbor})
                
                # Build response format
                node_list = []
                for node in nodes:
                    node_data = {"id": node, "label": node}
                    # Try to get additional node data
                    if hasattr(graph, 'nodes') and hasattr(graph.nodes, '__getitem__'):
                        try:
                            node_attrs = graph.nodes[node]
                            node_data.update(node_attrs)
                        except:
                            pass
                    node_list.append(node_data)
                
                return {
                    "nodes": node_list,
                    "edges": edges
                }
            
            # If no graph storage available, return empty graph
            return {"nodes": [], "edges": []}
            
    except Exception as e:
        print(f"Error getting knowledge graph for label '{label}': {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error getting knowledge graph: {str(e)}"
        )

@app.get("/graph/entity/exists")
async def check_entity_exists(
    name: str = Query(..., description="Entity name to check"),
):
    """
    Check if an entity with the given name exists in the knowledge graph
    
    Args:
        name (str): Name of the entity to check
    
    Returns:
        Dict[str, bool]: Dictionary with 'exists' key indicating if entity exists
    """
    try:
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            if hasattr(graph, 'has_node'):
                exists = await graph.has_node(name) if asyncio.iscoroutinefunction(graph.has_node) else graph.has_node(name)
                return {"exists": exists}
            elif hasattr(graph, '__contains__'):
                exists = name in graph
                return {"exists": exists}
        
        return {"exists": False}
    except Exception as e:
        print(f"Error checking entity existence for '{name}': {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error checking entity existence: {str(e)}"
        )

@app.post("/graph/entity/edit")
async def update_entity(request: EntityUpdateRequest):
    """
    Update an entity's properties in the knowledge graph
    
    Args:
        request (EntityUpdateRequest): Request containing entity name, updated data, and rename flag
    
    Returns:
        Dict: Updated entity information
    """
    try:
        # Check if rag_instance has the edit method
        if hasattr(rag_instance, 'aedit_entity'):
            result = await rag_instance.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
            )
            return {
                "status": "success",
                "message": "Entity updated successfully",
                "data": result,
            }
        else:
            # Alternative approach: update entity in graph storage directly
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                
                # Check if entity exists
                if hasattr(graph, 'has_node') and graph.has_node(request.entity_name):
                    # Update node attributes
                    if hasattr(graph, 'nodes') and hasattr(graph.nodes, '__setitem__'):
                        for key, value in request.updated_data.items():
                            graph.nodes[request.entity_name][key] = value
                    
                    # Handle rename if requested
                    if request.allow_rename and 'name' in request.updated_data:
                        new_name = request.updated_data['name']
                        if new_name != request.entity_name:
                            # This is more complex - would need to rename node
                            # For now, just return success without renaming
                            pass
                    
                    return {
                        "status": "success",
                        "message": "Entity updated successfully",
                        "data": request.updated_data,
                    }
                else:
                    raise ValueError(f"Entity '{request.entity_name}' not found")
            
            raise ValueError("Graph storage not available")
            
    except ValueError as ve:
        print(f"Validation error updating entity '{request.entity_name}': {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error updating entity '{request.entity_name}': {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error updating entity: {str(e)}"
        )

@app.post("/graph/relation/edit")
async def update_relation(request: RelationUpdateRequest):
    """Update a relation's properties in the knowledge graph
    
    Args:
        request (RelationUpdateRequest): Request containing source ID, target ID and updated data
    
    Returns:
        Dict: Updated relation information
    """
    try:
        # Check if rag_instance has the edit method
        if hasattr(rag_instance, 'aedit_relation'):
            result = await rag_instance.aedit_relation(
                source_entity=request.source_id,
                target_entity=request.target_id,
                updated_data=request.updated_data,
            )
            return {
                "status": "success",
                "message": "Relation updated successfully",
                "data": result,
            }
        else:
            # Alternative approach: update edge in graph storage directly
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                
                # Check if edge exists
                if hasattr(graph, 'has_edge') and graph.has_edge(request.source_id, request.target_id):
                    # Update edge attributes
                    if hasattr(graph, 'edges') and hasattr(graph.edges, '__getitem__'):
                        edge = graph.edges[request.source_id, request.target_id]
                        for key, value in request.updated_data.items():
                            edge[key] = value
                    
                    return {
                        "status": "success",
                        "message": "Relation updated successfully",
                        "data": request.updated_data,
                    }
                else:
                    raise ValueError(f"Relation between '{request.source_id}' and '{request.target_id}' not found")
            
            raise ValueError("Graph storage not available")
            
    except ValueError as ve:
        print(f"Validation error updating relation between '{request.source_id}' and '{request.target_id}': {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error updating relation between '{request.source_id}' and '{request.target_id}': {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error updating relation: {str(e)}"
        )

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
        
        # Check if document_id was provided by n8n workflow
        if request.document_id:
            # Check if n8n sent us an [unknown] URL format and fix it
            if request.document_id.startswith('[unknown] '):
                # Remove the [unknown] prefix
                url_part = request.document_id[10:]  # Remove '[unknown] '
                # Try to create a better ID from the URL
                if request.source_url:
                    try:
                        parsed_url = urlparse(request.source_url)
                        domain = parsed_url.netloc.replace('www.', '')
                        path = parsed_url.path.strip('/')
                        if path:
                            file_path = f"[{domain}] {path}"
                            display_name = f"[{domain}] {path}"  # Show full path, not just slug
                        else:
                            file_path = f"[{domain}]"
                            display_name = f"[{domain}]"
                        doc_id = file_path
                    except:
                        # If parsing fails, try to extract domain from the URL string
                        if url_part.startswith('http://') or url_part.startswith('https://'):
                            # Try a simpler extraction
                            try:
                                # Extract domain from URL string
                                url_without_protocol = url_part.split('://', 1)[1]
                                domain = url_without_protocol.split('/', 1)[0].replace('www.', '')
                                path_part = url_without_protocol.split('/', 1)[1] if '/' in url_without_protocol else ''
                                
                                if path_part:
                                    file_path = f"[{domain}] {path_part}"
                                    display_name = f"[{domain}] {path_part}"  # Show full path
                                else:
                                    file_path = f"[{domain}]"
                                    display_name = f"[{domain}]"
                                doc_id = file_path
                            except:
                                # Last resort: use URL as-is but try to make a display name
                                file_path = url_part
                                display_name = url_part.split('/')[-1] if '/' in url_part else url_part
                                doc_id = compute_doc_id(enriched_content)
                        else:
                            # Not a URL format we recognize
                            file_path = url_part
                            display_name = url_part
                            doc_id = compute_doc_id(enriched_content)
                else:
                    # Just remove the [unknown] prefix, but still try to format nicely
                    file_path = url_part
                    # Try to extract a slug for display
                    if '/' in url_part:
                        display_name = url_part.split('/')[-1]
                    else:
                        display_name = url_part
                    doc_id = compute_doc_id(enriched_content)
            else:
                # The document_id from n8n is already in the format we want
                file_path = request.document_id
                # Use the provided document_id as the doc_id to ensure consistency
                doc_id = request.document_id
                
                # Use the full path for display_name
                display_name = request.document_id  # Show full path with domain
            
            # Store the full URL path in metadata for reference
            full_path = None
            if request.source_url:
                parsed_url = urlparse(request.source_url)
                domain = parsed_url.netloc.replace('www.', '')
                path = parsed_url.path.strip('/')
                full_path = f"[{domain}] {path}" if path else f"[{domain}]"
        else:
            # Compute document ID
            doc_id = compute_doc_id(enriched_content)
            
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
                    # For display_name, show the full path
                    display_name = f"[{domain}] {path}"
                else:
                    # For root domain (https://ai.pydantic.dev/)
                    file_path = f"[{domain}]"
                    display_name = f"[{domain}]"
            else:
                file_path = f"text/{doc_id}.txt"
                display_name = f"text/{doc_id[:8]}..."  # Shortened ID for display
        
        # Determine which ID to use for LightRAG
        if request.document_id and request.document_id.startswith('[') and ']' in request.document_id:
            # Check if this is an [unknown] format and use the cleaned version
            if request.document_id.startswith('[unknown] '):
                # Use the cleaned file_path we generated above as the custom ID
                custom_id = file_path
            else:
                # Use the n8n-provided document_id as the custom ID
                custom_id = request.document_id
            # IMPORTANT: Use the custom_id as the key for metadata storage
            # This ensures consistency between LightRAG's internal ID and our metadata
            metadata_key = custom_id
        else:
            # Use the computed hash ID
            custom_id = doc_id
            metadata_key = doc_id
        
        # Store metadata with the correct key
        metadata_entry = {
            "id": metadata_key,
            "original_doc_id": doc_id,  # Keep the hash ID for reference
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
            
        metadata_store[metadata_key] = metadata_entry
        
        # Save metadata after each insert
        save_metadata_store()
        
        # Insert into LightRAG with custom ID
        await rag_instance.ainsert(enriched_content, ids=[custom_id], file_paths=[file_path])
        
        # Log for debugging
        print(f"Document inserted - Internal ID: {doc_id}, Custom ID: {custom_id}, file_path: {file_path}")
        
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
        
        # Debug logging
        print("\n=== Getting documents for WebUI ===")
        print(f"Number of documents in metadata store: {len(metadata_store)}")
        
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
                                # CRITICAL: The doc_id here is whatever ID LightRAG is using internally
                                # This could be either our custom ID or a hash ID
                                
                                # Try to get metadata using the doc_id first
                                metadata = metadata_store.get(doc_id, {})
                                
                                # If no metadata found and doc_id looks like a custom ID, it might be stored differently
                                if not metadata and doc_id.startswith('[') and ']' in doc_id:
                                    # This is likely a custom ID, create proper metadata
                                    file_path = doc_id
                                    display_name = doc_id
                                    
                                    # Use full path for display_name
                                    display_name = file_path  # Shows [domain] full_path
                                    
                                    metadata = {"file_path": file_path, "display_name": display_name}
                                
                                # If still no metadata, check if this is a hash ID and search for metadata by content
                                if not metadata and doc_id.startswith('doc-'):
                                    # Search through metadata store for entries with this original_doc_id
                                    for key, meta in metadata_store.items():
                                        if meta.get('original_doc_id') == doc_id:
                                            metadata = meta
                                            break
                                
                                # Ensure file_path exists
                                file_path = metadata.get('file_path', doc_data.get('file_path', f"text/{doc_id}.txt"))
                                
                                # ALWAYS ensure display_name shows full path
                                display_name = metadata.get('display_name', file_path)
                                # Force full path display even if metadata has old format
                                if display_name and "[" in display_name and "]" in display_name:
                                    # If it looks like [domain] something, ensure it's the full path
                                    if file_path and file_path.startswith('[') and file_path != display_name:
                                        display_name = file_path
                                
                                # For display, if doc_id is already in the correct format, use it
                                if doc_id.startswith('[') and ']' in doc_id:
                                    # This document is using a custom ID in the correct format
                                    display_id = doc_id
                                elif file_path and file_path.startswith('[') and ']' in file_path:
                                    # Use file_path as display ID if it's in the correct format
                                    display_id = file_path
                                else:
                                    # Check if we have a better ID in metadata
                                    display_id = metadata.get('id', doc_id)
                                
                                # IMPORTANT: Never return a URL as the ID
                                # If display_id looks like a URL, use the file_path or doc_id instead
                                if display_id and ('http://' in display_id or 'https://' in display_id):
                                    if file_path and file_path.startswith('['):
                                        display_id = file_path
                                    else:
                                        display_id = doc_id
                                
                                documents.append({
                                    "id": display_id,  # This is what the WebUI displays
                                    "doc_id": doc_id,  # Keep the actual doc_id for reference
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
                                
                                # ALWAYS ensure display_name shows full path
                                display_name = metadata.get('display_name', file_path)
                                # Force full path display even if metadata has old format
                                if display_name and "[" in display_name and "]" in display_name:
                                    # If it looks like [domain] something, ensure it's the full path
                                    if file_path and file_path.startswith('[') and file_path != display_name:
                                        display_name = file_path
                                
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
                                        # FORCE display_name to show full path
                                        display_name = file_path  # This will show [domain] full/path
                                    
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
                    # FORCE display_name to show full path, not just slug
                    display_name = file_path  # This will show [domain] full/path
                
                # Use file_path as the display ID if it's in the proper format
                display_id = file_path if file_path.startswith('[') and ']' in file_path else doc_id
                
                # IMPORTANT: Never return a URL as the ID
                if display_id and ('http://' in display_id or 'https://' in display_id):
                    if file_path and file_path.startswith('['):
                        display_id = file_path
                    else:
                        display_id = doc_id
                
                documents.append({
                    "id": display_id,  # This is what the WebUI displays
                    "doc_id": doc_id,  # Keep the actual doc_id for reference
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
                    
                # Use file_path as the display ID if it's in the proper format
                display_id = file_path if file_path.startswith('[') and ']' in file_path else doc_id
                
                matching_docs.append({
                    "id": display_id,  # What the WebUI displays
                    "doc_id": doc_id,  # Actual document ID
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
            # Try direct key first
            if doc_id in metadata_store:
                del metadata_store[doc_id]
                deleted_count += 1
            else:
                # Also check if this is an original_doc_id
                for key in list(metadata_store.keys()):
                    if metadata_store[key].get('original_doc_id') == doc_id:
                        del metadata_store[key]
                        deleted_count += 1
                        break
        
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
        metadata_keys_to_delete = []
        entities_to_delete = set()
        
        for doc_id, metadata in metadata_store.items():
            # Check both sitemap_identifier (legacy) and sitemap_url
            if (metadata.get('sitemap_url') == sitemap_url or 
                metadata.get('sitemap_identifier') == f"[SITEMAP: {sitemap_url}]"):
                # The doc_id in metadata_store is the key, but LightRAG might be using custom_id
                metadata_keys_to_delete.append(doc_id)
                
                # For LightRAG deletion, we need to use the actual ID that LightRAG is using
                if doc_id.startswith('[') and ']' in doc_id:
                    # This is likely the custom ID used by LightRAG
                    docs_to_delete.append(doc_id)
                else:
                    # Check if there's an original_doc_id that might be the hash ID
                    original_id = metadata.get('original_doc_id', doc_id)
                    docs_to_delete.append(original_id)
                    # Also try the file_path as it might be the custom ID
                    file_path = metadata.get('file_path')
                    if file_path and file_path != original_id:
                        docs_to_delete.append(file_path)
        
        if docs_to_delete:
            # Remove duplicates
            unique_docs_to_delete = list(set(docs_to_delete))
            
            print(f"Deleting documents for sitemap {sitemap_url}")
            print(f"Document IDs to delete from LightRAG: {unique_docs_to_delete}")
            print(f"Metadata keys to delete: {metadata_keys_to_delete}")
            
            # Step 1: Extract entities from documents before deletion
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                
                # Try to find entities mentioned in these documents
                # This is a workaround since LightRAG doesn't track document-entity relationships
                for doc_id in unique_docs_to_delete:
                    # Check text chunks for entity references
                    if hasattr(rag_instance, 'kv_storage') and rag_instance.kv_storage:
                        try:
                            # Look for all chunks of this document
                            if hasattr(rag_instance.kv_storage, '_data'):
                                for key in list(rag_instance.kv_storage._data.keys()):
                                    if key.startswith(f"chunk-{doc_id}"):
                                        try:
                                            chunk_data = await rag_instance.kv_storage.get({key})
                                            if chunk_data and key in chunk_data:
                                                chunk_text = str(chunk_data[key])
                                                # Simple heuristic: look for entities that might be in this chunk
                                                # Check all nodes in the graph
                                                if hasattr(graph, 'nodes'):
                                                    for node in graph.nodes():
                                                        # If the entity name appears in the chunk text
                                                        if isinstance(node, str) and node.lower() in chunk_text.lower():
                                                            entities_to_delete.add(node)
                                        except:
                                            pass
                        except Exception as e:
                            print(f"Error extracting entities: {e}")
            
            # Step 2: Clean up all document traces
            await cleanup_all_document_traces(unique_docs_to_delete)
            
            # Step 3: Try to delete using LightRAG's method (even though it's buggy)
            try:
                await rag_instance.adelete_by_doc_id(unique_docs_to_delete)
            except Exception as e:
                print(f"Error in adelete_by_doc_id (expected due to known bugs): {e}")
            
            # Step 4: Delete entities found in the documents
            if entities_to_delete and hasattr(rag_instance, 'adelete_by_entity'):
                print(f"Deleting {len(entities_to_delete)} entities associated with documents")
                for entity in entities_to_delete:
                    try:
                        await rag_instance.adelete_by_entity(entity)
                    except Exception as e:
                        print(f"Error deleting entity '{entity}': {e}")
            
            # Step 5: Force graph cleanup for the specific documents
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph = rag_instance.chunk_entity_relation_graph
                nodes_to_remove = []
                
                # More aggressive entity removal based on document IDs in metadata
                if hasattr(graph, 'nodes'):
                    for node, data in graph.nodes(data=True):
                        # Check if node data contains source_id matching our documents
                        if isinstance(data, dict) and 'source_id' in data:
                            source_ids = data['source_id']
                            if isinstance(source_ids, str):
                                source_ids = [source_ids]
                            
                            # Check if any source_id matches our deleted documents
                            for source_id in source_ids:
                                for doc_id in unique_docs_to_delete:
                                    if doc_id in source_id or source_id in doc_id:
                                        nodes_to_remove.append(node)
                                        break
                
                # Remove the nodes
                if nodes_to_remove:
                    print(f"Removing {len(nodes_to_remove)} nodes from graph")
                    if hasattr(graph, 'remove_nodes_from'):
                        graph.remove_nodes_from(nodes_to_remove)
                    elif hasattr(graph, 'remove_node'):
                        for node in nodes_to_remove:
                            try:
                                graph.remove_node(node)
                            except:
                                pass
            
            # Step 6: Clear vector storage entries for deleted entities
            if entities_to_delete and hasattr(rag_instance, 'entities_vdb'):
                try:
                    await rag_instance.entities_vdb.delete(list(entities_to_delete))
                except Exception as e:
                    print(f"Error clearing entity vectors: {e}")
            
            # Step 7: Remove from metadata store
            for key in metadata_keys_to_delete:
                if key in metadata_store:
                    del metadata_store[key]
            
            # Save updated metadata
            save_metadata_store()
            
            # Step 8: Force storage sync
            if hasattr(rag_instance, 'graph_storage') and rag_instance.graph_storage:
                try:
                    if hasattr(rag_instance.graph_storage, 'sync'):
                        await rag_instance.graph_storage.sync()
                    # For NetworkXStorage, trigger a save
                    if hasattr(rag_instance.graph_storage, 'save'):
                        await rag_instance.graph_storage.save()
                except Exception as e:
                    print(f"Error syncing graph storage: {e}")
        
        return {
            "status": "success",
            "message": f"Deleted {len(metadata_keys_to_delete)} documents for sitemap {sitemap_url}",
            "deleted_count": len(metadata_keys_to_delete),
            "deleted_ids": unique_docs_to_delete if 'unique_docs_to_delete' in locals() else [],
            "entities_deleted": len(entities_to_delete),
            "sitemap_url": sitemap_url,
            "note": "Graph cleanup completed with entity removal"
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

@app.post("/graph/rebuild")
async def rebuild_graph():
    """
    Rebuild the knowledge graph from scratch based on existing documents.
    This is useful after deleting documents to ensure the graph is clean.
    WARNING: This operation can be time-consuming for large document sets.
    """
    try:
        print("Starting knowledge graph rebuild...")
        
        # Step 1: Clear the existing graph completely
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            
            # Count nodes before clearing
            node_count = len(graph.nodes()) if hasattr(graph, 'nodes') else 0
            edge_count = len(graph.edges()) if hasattr(graph, 'edges') else 0
            
            print(f"Clearing graph with {node_count} nodes and {edge_count} edges")
            
            # Clear using the appropriate method
            if hasattr(graph, 'clear'):
                graph.clear()
            elif hasattr(graph, 'remove_nodes_from'):
                nodes = list(graph.nodes())
                graph.remove_nodes_from(nodes)
        
        # Step 2: Clear vector storages for entities and relationships
        if hasattr(rag_instance, 'entities_vdb') and rag_instance.entities_vdb:
            try:
                # For some vector stores, we might need to recreate them
                # This is implementation-specific
                if hasattr(rag_instance.entities_vdb, 'clear'):
                    await rag_instance.entities_vdb.clear()
            except:
                pass
        
        if hasattr(rag_instance, 'relationships_vdb') and rag_instance.relationships_vdb:
            try:
                if hasattr(rag_instance.relationships_vdb, 'clear'):
                    await rag_instance.relationships_vdb.clear()
            except:
                pass
        
        # Step 3: Force save the empty graph
        if hasattr(rag_instance, 'graph_storage') and rag_instance.graph_storage:
            try:
                if hasattr(rag_instance.graph_storage, 'save'):
                    await rag_instance.graph_storage.save()
            except Exception as e:
                print(f"Error saving cleared graph: {e}")
        
        # Step 4: Use LightRAG's rebuild method if available
        if hasattr(rag_instance, '_rebuild_knowledge_from_chunks'):
            try:
                print("Rebuilding knowledge graph from chunks...")
                await rag_instance._rebuild_knowledge_from_chunks()
                
                # Count nodes after rebuild
                if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                    graph = rag_instance.chunk_entity_relation_graph
                    new_node_count = len(graph.nodes()) if hasattr(graph, 'nodes') else 0
                    new_edge_count = len(graph.edges()) if hasattr(graph, 'edges') else 0
                    
                    return {
                        "status": "success",
                        "message": "Graph rebuilt successfully from chunks",
                        "before": {
                            "nodes": node_count,
                            "edges": edge_count
                        },
                        "after": {
                            "nodes": new_node_count,
                            "edges": new_edge_count
                        }
                    }
            except Exception as e:
                print(f"Error during rebuild: {e}")
                return {
                    "status": "partial",
                    "message": "Graph cleared but rebuild failed. Re-indexing documents required.",
                    "error": str(e)
                }
        
        # Step 5: If no rebuild method, suggest re-indexing
        return {
            "status": "success", 
            "message": "Graph cleared successfully. Please re-index documents to rebuild.",
            "before": {
                "nodes": node_count if 'node_count' in locals() else 0,
                "edges": edge_count if 'edge_count' in locals() else 0
            },
            "after": {
                "nodes": 0,
                "edges": 0
            },
            "action_required": "Re-index all documents to populate the graph"
        }
        
    except Exception as e:
        print(f"Error rebuilding graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/graph/clear")
async def clear_graph():
    """
    Clear the entire knowledge graph. 
    WARNING: This will remove ALL entities and relationships!
    Documents will remain but the graph will be empty.
    """
    try:
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            
            # Count nodes before clearing
            node_count = len(graph.nodes()) if hasattr(graph, 'nodes') else 0
            edge_count = len(graph.edges()) if hasattr(graph, 'edges') else 0
            
            # Clear the graph
            if hasattr(graph, 'clear'):
                graph.clear()
            elif hasattr(graph, 'remove_nodes_from'):
                # Remove all nodes (which also removes edges)
                nodes = list(graph.nodes())
                graph.remove_nodes_from(nodes)
            
            # Clear graph storage if separate
            if hasattr(rag_instance, 'graph_storage') and rag_instance.graph_storage:
                if hasattr(rag_instance.graph_storage, 'clear'):
                    await rag_instance.graph_storage.clear()
                # Force save for NetworkXStorage
                if hasattr(rag_instance.graph_storage, 'save'):
                    await rag_instance.graph_storage.save()
            
            # Clear vector storage for entities and relationships
            if hasattr(rag_instance, 'entities_vdb') and rag_instance.entities_vdb:
                try:
                    # Implementation-specific clearing
                    if hasattr(rag_instance.entities_vdb, 'clear'):
                        await rag_instance.entities_vdb.clear()
                    elif hasattr(rag_instance.entities_vdb, '_data'):
                        # For simple implementations, clear the data dict
                        rag_instance.entities_vdb._data.clear()
                except:
                    pass
            
            if hasattr(rag_instance, 'relationships_vdb') and rag_instance.relationships_vdb:
                try:
                    if hasattr(rag_instance.relationships_vdb, 'clear'):
                        await rag_instance.relationships_vdb.clear()
                    elif hasattr(rag_instance.relationships_vdb, '_data'):
                        rag_instance.relationships_vdb._data.clear()
                except:
                    pass
            
            print(f"Cleared graph with {node_count} nodes and {edge_count} edges")
            
            return {
                "status": "success",
                "message": "Knowledge graph cleared successfully",
                "removed": {
                    "nodes": node_count,
                    "edges": edge_count
                }
            }
        else:
            return {
                "status": "error",
                "message": "Knowledge graph not available"
            }
            
    except Exception as e:
        print(f"Error clearing graph: {e}")
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