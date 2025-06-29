#!/usr/bin/env python3
"""
Extended LightRAG API Server that adds missing functionality for document management
"""
import os
import asyncio
import hashlib
import json
import networkx as nx
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
    """Enhanced cleanup that ensures all traces are removed and properly persisted"""
    try:
        print(f"Starting enhanced cleanup for documents: {doc_ids}")
        
        # Create a set of all possible document ID formats to check
        all_doc_id_patterns = set()
        for doc_id in doc_ids:
            all_doc_id_patterns.add(doc_id)
            if doc_id.startswith('[') and ']' in doc_id:
                parts = doc_id.split('] ', 1)
                if len(parts) == 2:
                    domain_part = parts[0] + ']'
                    all_doc_id_patterns.add(domain_part)
        
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        
        # Step 1: Clean vector databases FIRST
        # This is important because the graph might reference these
        vdb_files = ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"]
        
        entities_to_remove = set()
        relationships_to_remove = set()
        
        for vdb_file in vdb_files:
            vdb_path = os.path.join(working_dir, vdb_file)
            if os.path.exists(vdb_path):
                try:
                    with open(vdb_path, 'r') as f:
                        vdb_data = json.load(f)
                    
                    if 'data' in vdb_data and isinstance(vdb_data['data'], list):
                        original_count = len(vdb_data['data'])
                        filtered_data = []
                        
                        for entry in vdb_data['data']:
                            should_remove = False
                            
                            # Check various fields for document references
                            for field in ['doc_id', 'source_id', 'file_path', 'chunk_id']:
                                value = entry.get(field, '')
                                for pattern in all_doc_id_patterns:
                                    if pattern in str(value):
                                        should_remove = True
                                        
                                        # Track what we're removing for graph cleanup
                                        if vdb_file == "vdb_entities.json":
                                            entities_to_remove.add(entry.get('id', ''))
                                        elif vdb_file == "vdb_relationships.json":
                                            src = entry.get('source', '')
                                            tgt = entry.get('target', '')
                                            if src and tgt:
                                                relationships_to_remove.add((src, tgt))
                                        break
                                if should_remove:
                                    break
                            
                            if not should_remove:
                                filtered_data.append(entry)
                        
                        vdb_data['data'] = filtered_data
                        
                        # Clear matrix if data changed
                        if len(filtered_data) != original_count and 'matrix' in vdb_data:
                            vdb_data['matrix'] = []
                        
                        # Save immediately
                        with open(vdb_path, 'w') as f:
                            json.dump(vdb_data, f, indent=2)
                        
                        removed = original_count - len(filtered_data)
                        print(f"Removed {removed} entries from {vdb_file}")
                        
                except Exception as e:
                    print(f"Error updating {vdb_file}: {e}")
        
        # Step 2: Clean the in-memory graph
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            
            # Find all nodes and edges to remove
            nodes_to_remove = set()
            edges_to_remove = set()
            
            # Check all nodes
            for node_id, node_data in list(graph.nodes(data=True)):
                source_id = node_data.get('source_id', '')
                file_path = node_data.get('file_path', '')
                
                for pattern in all_doc_id_patterns:
                    if pattern in source_id or pattern in file_path:
                        nodes_to_remove.add(node_id)
                        break
            
            # Check all edges
            for src, tgt, edge_data in list(graph.edges(data=True)):
                # Remove if either node is being removed
                if src in nodes_to_remove or tgt in nodes_to_remove:
                    edges_to_remove.add((src, tgt))
                    continue
                
                # Check edge's source_id
                source_id = edge_data.get('source_id', '')
                for pattern in all_doc_id_patterns:
                    if pattern in source_id:
                        edges_to_remove.add((src, tgt))
                        break
            
            # Remove edges first
            for src, tgt in edges_to_remove:
                try:
                    graph.remove_edge(src, tgt)
                except:
                    pass
            
            # Remove nodes
            for node in nodes_to_remove:
                try:
                    graph.remove_node(node)
                except:
                    pass
            
            print(f"Removed {len(nodes_to_remove)} nodes and {len(edges_to_remove)} edges from graph")
            
            # Step 3: Save the graph multiple ways to ensure persistence
            graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
            
            # Method 1: Direct NetworkX save
            try:
                nx.write_graphml(graph, graphml_path)
                print(f"Saved graph using NetworkX to {graphml_path}")
            except Exception as e:
                print(f"Error saving with NetworkX: {e}")
            
            # Method 2: Use graph storage save methods
            if hasattr(rag_instance, 'graph_storage'):
                try:
                    if hasattr(rag_instance.graph_storage, 'save'):
                        await rag_instance.graph_storage.save()
                        print("Saved using graph_storage.save()")
                    elif hasattr(rag_instance.graph_storage, '_save'):
                        rag_instance.graph_storage._save()
                        print("Saved using graph_storage._save()")
                except Exception as e:
                    print(f"Error with graph storage save: {e}")
        
        # Step 4: Clean other storages
        # Clean KV storage
        if hasattr(rag_instance, 'kv_storage'):
            try:
                for doc_id in doc_ids:
                    await rag_instance.kv_storage.delete({doc_id})
                
                # Also clean chunks
                if hasattr(rag_instance.kv_storage, 'get_all'):
                    all_data = await rag_instance.kv_storage.get_all()
                    keys_to_delete = set()
                    for key in all_data.keys():
                        for pattern in all_doc_id_patterns:
                            if pattern in key:
                                keys_to_delete.add(key)
                                break
                    if keys_to_delete:
                        await rag_instance.kv_storage.delete(keys_to_delete)
            except Exception as e:
                print(f"Error cleaning KV storage: {e}")
        
        # Clean doc status
        if hasattr(rag_instance, 'doc_status'):
            try:
                await rag_instance.doc_status.delete(set(doc_ids))
            except Exception as e:
                print(f"Error cleaning doc status: {e}")
        
        # Step 5: Clear any caches
        if hasattr(rag_instance, 'llm_response_cache'):
            try:
                await rag_instance.aclear_cache()
            except Exception as e:
                print(f"Error clearing cache: {e}")
        
        # Step 6: Final verification and persistence
        # Make sure all JSON files are saved
        json_files = ["kv_store_full_docs.json", "kv_store_text_chunks.json", "doc_status.json"]
        for json_file in json_files:
            file_path = os.path.join(working_dir, json_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Remove any entries related to deleted docs
                    if isinstance(data, dict):
                        keys_to_remove = []
                        for key in data.keys():
                            for pattern in all_doc_id_patterns:
                                if pattern in key:
                                    keys_to_remove.append(key)
                                    break
                        
                        for key in keys_to_remove:
                            data.pop(key, None)
                        
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                except Exception as e:
                    print(f"Error cleaning {json_file}: {e}")
        
        print("Enhanced cleanup completed successfully")
        
    except Exception as e:
        print(f"Critical error in enhanced cleanup: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Check if GraphML file exists and is valid
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graphml_path):
        try:
            # Try to load it to check if it's valid
            test_graph = nx.read_graphml(graphml_path)
            print(f"Found existing graph with {test_graph.number_of_nodes()} nodes and {test_graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Error reading existing GraphML file: {e}")
            # Create an empty GraphML file
            empty_graph = nx.Graph()
            nx.write_graphml(empty_graph, graphml_path)
            print("Created new empty GraphML file")
    
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
    
    # Ensure the graph is properly loaded
    if hasattr(rag_instance, 'graph_storage') and hasattr(rag_instance.graph_storage, 'load'):
        try:
            await rag_instance.graph_storage.load()
            print("Graph storage loaded successfully")
        except Exception as e:
            print(f"Error loading graph storage: {e}")
    
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

@app.get("/graph/status")
async def get_graph_status():
    """Get the current status of the knowledge graph"""
    try:
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            
            status = {
                "exists": True,
                "type": type(graph).__name__,
                "nodes": graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                "edges": graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
                "storage_type": type(rag_instance.graph_storage).__name__ if hasattr(rag_instance, 'graph_storage') else "Unknown"
            }
            
            # Check all relevant files
            working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
            
            # Check GraphML file
            graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
            status["graphml_file_exists"] = os.path.exists(graphml_file)
            if status["graphml_file_exists"]:
                status["graphml_file_size"] = os.path.getsize(graphml_file)
            
            # Check vector database files
            vector_db_files = {
                "vdb_entities.json": "Entity embeddings",
                "vdb_relationships.json": "Relationship embeddings",
                "vdb_chunks.json": "Document chunk embeddings"
            }
            
            status["vector_databases"] = {}
            for file_name, description in vector_db_files.items():
                file_path = os.path.join(working_dir, file_name)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            entity_count = len(data.get('data', []))
                            status["vector_databases"][file_name] = {
                                "exists": True,
                                "description": description,
                                "size": os.path.getsize(file_path),
                                "entity_count": entity_count
                            }
                    except Exception as e:
                        status["vector_databases"][file_name] = {
                            "exists": True,
                            "error": str(e)
                        }
                else:
                    status["vector_databases"][file_name] = {
                        "exists": False,
                        "description": description
                    }
            
            # Sample some nodes if they exist
            if status["nodes"] > 0 and hasattr(graph, 'nodes'):
                sample_nodes = list(graph.nodes())[:5]
                status["sample_nodes"] = sample_nodes
            
            # Total entity count from vector DBs
            total_entities_in_vdbs = sum(
                vdb.get("entity_count", 0) 
                for vdb in status["vector_databases"].values() 
                if "entity_count" in vdb
            )
            status["total_entities_in_vector_dbs"] = total_entities_in_vdbs
            
            return status
        else:
            return {
                "exists": False,
                "message": "Knowledge graph not initialized"
            }
            
    except Exception as e:
        print(f"Error getting graph status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/graph/clear")
async def clear_knowledge_graph():
    """Clear the entire knowledge graph"""
    try:
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            
            # Get node and edge counts before clearing
            node_count = graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0
            edge_count = graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0
            
            if hasattr(graph, 'clear'):
                print(f"Clearing knowledge graph with {node_count} nodes and {edge_count} edges")
                graph.clear()
                
                # Critical: Save the cleared graph to the GraphML file
                working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
                
                # Method 1: Use NetworkX write_graphml directly to ensure it's saved
                try:
                    nx.write_graphml(graph, graphml_path)
                    print(f"Saved cleared graph to {graphml_path} using NetworkX")
                except Exception as e:
                    print(f"Error saving with NetworkX: {e}")
                
                # Method 2: Try the storage's save methods
                saved = False
                if hasattr(rag_instance.graph_storage, 'save'):
                    try:
                        await rag_instance.graph_storage.save()
                        print("Saved using async graph_storage.save()")
                        saved = True
                    except Exception as e:
                        print(f"Error with async save: {e}")
                        
                if not saved and hasattr(rag_instance.graph_storage, '_save'):
                    try:
                        rag_instance.graph_storage._save()
                        print("Saved using sync graph_storage._save()")
                        saved = True
                    except Exception as e:
                        print(f"Error with sync _save: {e}")
                
                # Method 3: If storage has upsert method, trigger a save by upserting nothing
                if not saved and hasattr(rag_instance.graph_storage, 'upsert'):
                    try:
                        await rag_instance.graph_storage.upsert([], [])
                        print("Triggered save via empty upsert")
                    except Exception as e:
                        print(f"Error with upsert trigger: {e}")
                
                # CRITICAL: Clear ALL graph-related and vector database files
                files_to_clear = [
                    # Graph files
                    "graph_chunk_entity_relation.graphml",
                    "graph_data.json",
                    "graph_cache.json",
                    # Vector database files - THESE ARE CRITICAL FOR WEBUI
                    "vdb_entities.json",
                    "vdb_relationships.json",
                    "vdb_chunks.json",
                    # Additional possible vector storage files
                    "entity_embedding.json",
                    "relationship_embedding.json",
                    "document_graph_storage.json"
                ]
                
                print(f"Clearing all graph and vector database files in {working_dir}")
                
                for file_name in files_to_clear:
                    file_path = os.path.join(working_dir, file_name)
                    if os.path.exists(file_path):
                        try:
                            if file_name.endswith('.graphml'):
                                # Ensure GraphML file contains empty graph
                                empty_graph = nx.Graph()
                                nx.write_graphml(empty_graph, file_path)
                                print(f"Wrote empty graph to {file_path}")
                            elif file_name.startswith('vdb_'):
                                # For vector database files, write the proper empty structure
                                empty_vdb = {
                                    "embedding_dim": 1536,  # Default OpenAI embedding dimension
                                    "data": [],
                                    "matrix": []
                                }
                                with open(file_path, 'w') as f:
                                    json.dump(empty_vdb, f)
                                print(f"Cleared vector database: {file_path}")
                            else:
                                # Clear JSON files
                                with open(file_path, 'w') as f:
                                    json.dump({}, f)
                                print(f"Cleared {file_path}")
                        except Exception as e:
                            print(f"Error handling {file_path}: {e}")
                            # Try to delete the file if clearing fails
                            try:
                                os.remove(file_path)
                                print(f"Deleted {file_path}")
                            except Exception as del_e:
                                print(f"Error deleting {file_path}: {del_e}")
                
                # Try to clear entity and relationship vector stores if accessible
                possible_vdb_attrs = [
                    'entities_vdb', 'relationships_vdb', 'chunks_vdb',
                    'entity_vdb', 'relation_vdb', 'chunk_vdb'
                ]
                
                for attr_name in possible_vdb_attrs:
                    if hasattr(rag_instance, attr_name):
                        vdb = getattr(rag_instance, attr_name)
                        if vdb and hasattr(vdb, 'clear'):
                            try:
                                await vdb.clear()
                                print(f"Cleared {attr_name}")
                            except Exception as e:
                                print(f"Error clearing {attr_name}: {e}")
                
                return {
                    "status": "success",
                    "message": "Knowledge graph and all vector databases cleared successfully",
                    "cleared": {
                        "nodes": node_count,
                        "edges": edge_count,
                        "files_cleared": files_to_clear
                    }
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Graph clear method not available"
                )
        else:
            raise HTTPException(
                status_code=404,
                detail="Knowledge graph not found"
            )
            
    except Exception as e:
        print(f"Error clearing knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/debug")
async def debug_graph_sources():
    """Debug endpoint to check all possible sources of graph data"""
    try:
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        debug_info = {
            "working_dir": working_dir,
            "in_memory_graph": {},
            "files": {},
            "rag_attributes": []
        }
        
        # Check in-memory graph
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            debug_info["in_memory_graph"] = {
                "exists": True,
                "nodes": graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                "edges": graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
                "type": type(graph).__name__
            }
        else:
            debug_info["in_memory_graph"]["exists"] = False
        
        # Check all potential files
        files_to_check = [
            "graph_chunk_entity_relation.graphml",
            "vdb_entities.json",
            "vdb_relationships.json",
            "vdb_chunks.json",
            "graph_data.json",
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json",
            "doc_status.json"
        ]
        
        for file_name in files_to_check:
            file_path = os.path.join(working_dir, file_name)
            if os.path.exists(file_path):
                file_info = {
                    "exists": True,
                    "size": os.path.getsize(file_path)
                }
                
                # For JSON files, try to get entity count
                if file_name.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                if 'data' in data:
                                    file_info["entity_count"] = len(data['data'])
                                else:
                                    file_info["key_count"] = len(data)
                            elif isinstance(data, list):
                                file_info["item_count"] = len(data)
                    except Exception as e:
                        file_info["read_error"] = str(e)
                
                debug_info["files"][file_name] = file_info
            else:
                debug_info["files"][file_name] = {"exists": False}
        
        # Check RAG instance attributes
        for attr in dir(rag_instance):
            if not attr.startswith('_') and ('vdb' in attr or 'vector' in attr or 'graph' in attr):
                debug_info["rag_attributes"].append(attr)
        
        return debug_info
        
    except Exception as e:
        print(f"Error in debug endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/reload")
async def reload_graph():
    """Force reload the graph from the GraphML file"""
    try:
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        
        if os.path.exists(graphml_path):
            # Load the graph from file
            loaded_graph = nx.read_graphml(graphml_path)
            node_count = loaded_graph.number_of_nodes()
            edge_count = loaded_graph.number_of_edges()
            
            # Replace the in-memory graph
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                rag_instance.chunk_entity_relation_graph = loaded_graph
                
                # Also update the graph storage's reference if it has one
                if hasattr(rag_instance.graph_storage, '_graph'):
                    rag_instance.graph_storage._graph = loaded_graph
                elif hasattr(rag_instance.graph_storage, 'graph'):
                    rag_instance.graph_storage.graph = loaded_graph
                
                return {
                    "status": "success",
                    "message": f"Graph reloaded from {graphml_path}",
                    "nodes": node_count,
                    "edges": edge_count
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Could not find graph reference in RAG instance"
                )
        else:
            return {
                "status": "warning",
                "message": f"GraphML file not found at {graphml_path}",
                "nodes": 0,
                "edges": 0
            }
            
    except Exception as e:
        print(f"Error reloading graph: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
            # Remove duplicates from docs_to_delete
            unique_docs_to_delete = list(set(docs_to_delete))
            
            print(f"Deleting documents for sitemap {sitemap_url}")
            print(f"Document IDs to delete from LightRAG: {unique_docs_to_delete}")
            print(f"Metadata keys to delete: {metadata_keys_to_delete}")
            
            # CRITICAL: Call the enhanced cleanup function FIRST
            # This will clean all storage files and persist changes
            await cleanup_all_document_traces(unique_docs_to_delete)
            
            # Then call the standard delete method (this might be redundant but ensures compatibility)
            try:
                await rag_instance.adelete_by_doc_id(unique_docs_to_delete)
            except Exception as e:
                print(f"Error in adelete_by_doc_id: {e}")
                # Continue even if this fails, as cleanup_all_document_traces should have done the work
            
            # Remove from metadata store using the correct keys
            for key in metadata_keys_to_delete:
                if key in metadata_store:
                    del metadata_store[key]
            
            # Save updated metadata
            save_metadata_store()
            
            # Force a reload of the graph in the WebUI by clearing any cached references
            # This ensures the WebUI will read the updated files
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                # Force reload from disk
                working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
                if os.path.exists(graphml_path):
                    try:
                        import networkx as nx
                        # Reload the graph from the saved file to ensure consistency
                        reloaded_graph = nx.read_graphml(graphml_path)
                        rag_instance.chunk_entity_relation_graph = reloaded_graph
                        
                        # Update the graph storage reference if it exists
                        if hasattr(rag_instance, 'graph_storage'):
                            if hasattr(rag_instance.graph_storage, '_graph'):
                                rag_instance.graph_storage._graph = reloaded_graph
                            elif hasattr(rag_instance.graph_storage, 'graph'):
                                rag_instance.graph_storage.graph = reloaded_graph
                        
                        print(f"Reloaded graph from disk with {reloaded_graph.number_of_nodes()} nodes")
                    except Exception as e:
                        print(f"Error reloading graph: {e}")
        
        return {
            "status": "success",
            "message": f"Deleted {len(metadata_keys_to_delete)} documents for sitemap {sitemap_url}",
            "deleted_count": len(metadata_keys_to_delete),
            "deleted_ids": unique_docs_to_delete if 'unique_docs_to_delete' in locals() else [],
            "sitemap_url": sitemap_url,
            "graph_cleared": True
        }
        
    except Exception as e:
        print(f"Error in delete_documents_by_sitemap: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/graph/force-clear-all")
async def force_clear_all_graph_data():
    """Force clear ALL graph and vector database files - nuclear option"""
    try:
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        
        # List of ALL files that might contain graph or entity data
        files_to_delete = [
            # Graph files
            "graph_chunk_entity_relation.graphml",
            "graph_data.json",
            "graph_cache.json",
            # Vector database files
            "vdb_entities.json",
            "vdb_relationships.json", 
            "vdb_chunks.json",
            # Key-value stores that might contain entities
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json",
            "kv_store_llm_response_cache.json",
            # Document status
            "doc_status.json",
            # Any other potential files
            "entity_embedding.json",
            "relationship_embedding.json",
            "document_graph_storage.json",
            "entities.json",
            "relationships.json",
            "chunks.json"
        ]
        
        deleted_files = []
        errors = []
        
        # Clear in-memory graph
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph = rag_instance.chunk_entity_relation_graph
            if hasattr(graph, 'clear'):
                graph.clear()
                print("Cleared in-memory graph")
        
        # Delete or clear all files
        for file_name in files_to_delete:
            file_path = os.path.join(working_dir, file_name)
            if os.path.exists(file_path):
                try:
                    # For safety, rename to backup first
                    backup_path = file_path + ".backup"
                    os.rename(file_path, backup_path)
                    
                    # Create empty file based on type
                    if file_name.endswith('.graphml'):
                        empty_graph = nx.Graph()
                        nx.write_graphml(empty_graph, file_path)
                    elif file_name.startswith('vdb_'):
                        # Vector database format
                        empty_vdb = {
                            "embedding_dim": 1536,
                            "data": [],
                            "matrix": []
                        }
                        with open(file_path, 'w') as f:
                            json.dump(empty_vdb, f)
                    else:
                        # Empty JSON object
                        with open(file_path, 'w') as f:
                            json.dump({}, f)
                    
                    # Delete backup
                    os.remove(backup_path)
                    deleted_files.append(file_name)
                    print(f"Cleared {file_name}")
                    
                except Exception as e:
                    errors.append({"file": file_name, "error": str(e)})
                    print(f"Error clearing {file_name}: {e}")
        
        # Clear metadata store
        global metadata_store
        metadata_store = {}
        save_metadata_store()
        
        # Clear any caches
        if hasattr(rag_instance, 'aclear_cache'):
            try:
                await rag_instance.aclear_cache()
                print("Cleared LLM cache")
            except Exception as e:
                print(f"Error clearing cache: {e}")
        
        return {
            "status": "success",
            "message": "Force cleared all graph and vector database files",
            "deleted_files": deleted_files,
            "errors": errors,
            "total_files_cleared": len(deleted_files)
        }
        
    except Exception as e:
        print(f"Error in force clear: {e}")
        import traceback
        traceback.print_exc()
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