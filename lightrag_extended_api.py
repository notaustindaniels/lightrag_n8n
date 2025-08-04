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
import shutil
import time
import mimetypes
import io
import chardet

# Document parsing imports
import docx
import pypdf
import pptx
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import yaml
import csv
import magic

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

class SourceListResponse(BaseModel):
    sources: List[Dict[str, Any]]
    total: int

class FilteredQueryRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None  # Optional to allow backward compatibility
    mode: str = "hybrid"
    stream: bool = False
    # Keep compatibility with official QueryParam fields
    only_need_context: bool = False
    response_type: str = "default"
    top_k: int = 50
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

# Global variables
rag_instance = None
metadata_store = {}  # In-memory metadata store

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'txt', 'md', 'docx', 'pdf', 'pptx', 'rtf', 'odt', 'epub', 
    'html', 'htm', 'tex', 'json', 'xml', 'yaml', 'yml', 'csv', 
    'log', 'conf', 'ini', 'properties', 'sql', 'bat', 'sh', 
    'c', 'cpp', 'py', 'java', 'js', 'ts', 'swift', 'go', 
    'rb', 'php', 'css', 'scss', 'less'
}

# MIME type mapping
MIME_TO_EXT = {
    'text/plain': ['txt', 'log', 'conf', 'ini', 'properties'],
    'text/markdown': ['md'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['docx'],
    'application/pdf': ['pdf'],
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['pptx'],
    'application/rtf': ['rtf'],
    'application/vnd.oasis.opendocument.text': ['odt'],
    'application/epub+zip': ['epub'],
    'text/html': ['html', 'htm'],
    'text/x-tex': ['tex'],
    'application/json': ['json'],
    'application/xml': ['xml'],
    'text/xml': ['xml'],
    'application/x-yaml': ['yaml', 'yml'],
    'text/yaml': ['yaml', 'yml'],
    'text/csv': ['csv'],
    'text/x-sql': ['sql'],
    'text/x-sh': ['sh'],
    'text/x-bat': ['bat'],
    'text/x-c': ['c'],
    'text/x-c++': ['cpp'],
    'text/x-python': ['py'],
    'text/x-java': ['java'],
    'text/javascript': ['js'],
    'application/javascript': ['js'],
    'text/x-typescript': ['ts'],
    'text/x-swift': ['swift'],
    'text/x-go': ['go'],
    'text/x-ruby': ['rb'],
    'text/x-php': ['php'],
    'text/css': ['css'],
    'text/x-scss': ['scss'],
    'text/x-less': ['less']
}

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

def extract_text_from_file(file_content: bytes, file_extension: str, filename: str) -> str:
    """Extract text from various file formats"""
    try:
        # Text-based files
        if file_extension in ['txt', 'md', 'log', 'conf', 'ini', 'properties', 'sql', 
                             'bat', 'sh', 'c', 'cpp', 'py', 'java', 'js', 'ts', 
                             'swift', 'go', 'rb', 'php', 'css', 'scss', 'less', 'tex']:
            # Detect encoding
            detected = chardet.detect(file_content)
            encoding = detected['encoding'] or 'utf-8'
            return file_content.decode(encoding, errors='replace')
        
        # Microsoft Word
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file_content))
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text)
            return '\n\n'.join(paragraphs)
        
        # PDF
        elif file_extension == 'pdf':
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            return '\n\n'.join(text_parts)
        
        # PowerPoint
        elif file_extension == 'pptx':
            prs = pptx.Presentation(io.BytesIO(file_content))
            text_parts = []
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, 'text') and shape.text.strip():
                        slide_text.append(shape.text)
                if slide_text:
                    text_parts.append(f"--- Slide {slide_num} ---\n" + '\n'.join(slide_text))
            return '\n\n'.join(text_parts)
        
        # HTML
        elif file_extension in ['html', 'htm']:
            soup = BeautifulSoup(file_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text
            text = soup.get_text()
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Drop blank lines
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        
        # EPUB
        elif file_extension == 'epub':
            book = epub.read_epub(io.BytesIO(file_content))
            text_parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text = soup.get_text()
                    if text.strip():
                        text_parts.append(text)
            return '\n\n'.join(text_parts)
        
        # JSON
        elif file_extension == 'json':
            data = json.loads(file_content.decode('utf-8'))
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        # XML
        elif file_extension == 'xml':
            soup = BeautifulSoup(file_content, 'xml')
            return soup.prettify()
        
        # YAML
        elif file_extension in ['yaml', 'yml']:
            data = yaml.safe_load(file_content.decode('utf-8'))
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        
        # CSV
        elif file_extension == 'csv':
            detected = chardet.detect(file_content)
            encoding = detected['encoding'] or 'utf-8'
            text = file_content.decode(encoding, errors='replace')
            # Parse CSV and format as readable text
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if rows:
                # Format as markdown table
                result = []
                if len(rows) > 1:
                    # Assume first row is header
                    headers = rows[0]
                    result.append(' | '.join(headers))
                    result.append(' | '.join(['---'] * len(headers)))
                    for row in rows[1:]:
                        result.append(' | '.join(row))
                else:
                    # Just one row
                    result.append(' | '.join(rows[0]))
                return '\n'.join(result)
            return text
        
        # RTF and ODT would require additional libraries (python-rtf, odfpy)
        # For now, try to extract as plain text
        elif file_extension in ['rtf', 'odt']:
            # Basic text extraction - won't preserve formatting
            detected = chardet.detect(file_content)
            encoding = detected['encoding'] or 'utf-8'
            text = file_content.decode(encoding, errors='replace')
            # For RTF, try to extract visible text (very basic)
            if file_extension == 'rtf':
                import re
                # Remove RTF commands (basic approach)
                text = re.sub(r'\\[a-z]+[0-9]*\s?', '', text)
                text = re.sub(r'[{}]', '', text)
            return text
        
        else:
            # Fallback - try to decode as text
            detected = chardet.detect(file_content)
            encoding = detected['encoding'] or 'utf-8'
            return file_content.decode(encoding, errors='replace')
            
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        # Fallback - try to decode as text
        try:
            detected = chardet.detect(file_content)
            encoding = detected['encoding'] or 'utf-8'
            return file_content.decode(encoding, errors='replace')
        except:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not extract text from {filename}: {str(e)}"
            )

def generate_display_name_from_file_path(file_path: str, doc_id: str) -> str:
    """Generate a display name from a file path for legacy documents"""
    if "[" in file_path and "]" in file_path:
        # Return the full file_path as display_name (includes domain and full path)
        return file_path
    else:
        return f"text/{doc_id[:8]}..."

def get_graph_from_storage(storage_or_graph):
    """Safely get NetworkX graph from storage object or return graph directly"""
    if storage_or_graph is None:
        return None
    
    # Check if it's already a NetworkX graph
    if hasattr(storage_or_graph, 'nodes') and hasattr(storage_or_graph, 'edges'):
        # Check if it's a direct NetworkX graph by trying to access nodes
        try:
            # Try to call a NetworkX-specific method
            storage_or_graph.number_of_nodes()
            return storage_or_graph
        except AttributeError:
            pass
    
    # Try to get graph from storage object
    if hasattr(storage_or_graph, '_graph'):
        return storage_or_graph._graph
    elif hasattr(storage_or_graph, 'graph'):
        return storage_or_graph.graph
    elif hasattr(storage_or_graph, 'get_graph'):
        return storage_or_graph.get_graph()
    elif hasattr(storage_or_graph, 'data'):
        return storage_or_graph.data
    
    # If all else fails, return the object itself
    return storage_or_graph

def force_save_graph_to_disk(graph, working_dir):
    """Force save graph to disk with multiple fallback methods"""
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    
    try:
        # Method 1: Direct NetworkX write
        nx.write_graphml(graph, graphml_path)
        print(f"Saved graph to {graphml_path} using NetworkX")
        
        # Verify the file was written
        if os.path.exists(graphml_path):
            file_size = os.path.getsize(graphml_path)
            print(f"GraphML file size: {file_size} bytes")
            
            # Double-check by reading it back
            test_graph = nx.read_graphml(graphml_path)
            print(f"Verified: graph has {test_graph.number_of_nodes()} nodes and {test_graph.number_of_edges()} edges")
        
        return True
    except Exception as e:
        print(f"Error in force_save_graph_to_disk: {e}")
        
        # Fallback: Write to temp file and move
        try:
            temp_path = graphml_path + ".tmp"
            nx.write_graphml(graph, temp_path)
            shutil.move(temp_path, graphml_path)
            print(f"Saved graph using temp file method")
            return True
        except Exception as e2:
            print(f"Error in temp file method: {e2}")
            return False

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
        
        # Step 1: Clean vector databases FIRST and SAVE IMMEDIATELY
        vdb_files = ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"]
        
        entities_to_remove = set()
        relationships_to_remove = set()
        
        for vdb_file in vdb_files:
            vdb_path = os.path.join(working_dir, vdb_file)
            if os.path.exists(vdb_path):
                try:
                    # Read the file
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
                        if len(filtered_data) != original_count:
                            vdb_data['matrix'] = []
                        
                        # Save immediately with backup
                        backup_path = vdb_path + ".backup"
                        shutil.copy2(vdb_path, backup_path)
                        
                        with open(vdb_path, 'w') as f:
                            json.dump(vdb_data, f, indent=2)
                        
                        # Verify the write
                        with open(vdb_path, 'r') as f:
                            verify_data = json.load(f)
                            if len(verify_data.get('data', [])) == len(filtered_data):
                                os.remove(backup_path)  # Remove backup if successful
                                print(f"Successfully updated {vdb_file}, removed {original_count - len(filtered_data)} entries")
                            else:
                                shutil.move(backup_path, vdb_path)  # Restore backup
                                raise Exception("Verification failed after write")
                        
                except Exception as e:
                    print(f"Error updating {vdb_file}: {e}")
        
        # Step 2: Clean the in-memory graph AND SAVE IMMEDIATELY
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            if graph is None:
                print("Warning: Could not access graph from storage")
            else:
                # Get initial counts
                initial_nodes = graph.number_of_nodes()
                initial_edges = graph.number_of_edges()
                
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
                print(f"Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                
                # Step 3: CRITICAL - Force save the graph multiple ways
                graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
                
                # Create backup before saving
                if os.path.exists(graphml_path):
                    backup_path = graphml_path + ".backup"
                    shutil.copy2(graphml_path, backup_path)
                
                # Force save the graph
                save_success = force_save_graph_to_disk(graph, working_dir)
                
                if save_success:
                    # Also try storage save methods
                    if hasattr(rag_instance, 'graph_storage'):
                        try:
                            if hasattr(rag_instance.graph_storage, 'save'):
                                await rag_instance.graph_storage.save()
                                print("Also saved using graph_storage.save()")
                        except Exception as e:
                            print(f"Note: graph_storage.save() failed but file was saved: {e}")
                    
                    # Update the graph storage's internal reference
                    if hasattr(graph_storage, '_graph'):
                        graph_storage._graph = graph
                    elif hasattr(graph_storage, 'graph'):
                        graph_storage.graph = graph
                    
                    # Also update if graph_storage is the storage itself
                    if hasattr(rag_instance, 'graph_storage'):
                        if hasattr(rag_instance.graph_storage, '_graph'):
                            rag_instance.graph_storage._graph = graph
                        elif hasattr(rag_instance.graph_storage, 'graph'):
                            rag_instance.graph_storage.graph = graph
        
        # Step 4: Clean other JSON storage files
        json_files_to_clean = {
            "kv_store_full_docs.json": "doc_id",
            "kv_store_text_chunks.json": "chunk_id",
            "doc_status.json": "doc_id"
        }
        
        for json_file, id_field in json_files_to_clean.items():
            file_path = os.path.join(working_dir, json_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        keys_to_remove = []
                        for key in data.keys():
                            for pattern in all_doc_id_patterns:
                                if pattern in key:
                                    keys_to_remove.append(key)
                                    break
                        
                        for key in keys_to_remove:
                            data.pop(key, None)
                        
                        # Save with backup
                        backup_path = file_path + ".backup"
                        shutil.copy2(file_path, backup_path)
                        
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        os.remove(backup_path)
                        print(f"Cleaned {len(keys_to_remove)} entries from {json_file}")
                        
                except Exception as e:
                    print(f"Error cleaning {json_file}: {e}")
        
        # Step 5: Clean KV storage if available
        if hasattr(rag_instance, 'kv_storage'):
            try:
                # Delete by doc_ids
                await rag_instance.kv_storage.delete(set(doc_ids))
                
                # Also clean any chunks
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
                        print(f"Deleted {len(keys_to_delete)} keys from KV storage")
            except Exception as e:
                print(f"Error cleaning KV storage: {e}")
        
        # Step 6: Clean doc status
        if hasattr(rag_instance, 'doc_status'):
            try:
                await rag_instance.doc_status.delete(set(doc_ids))
            except Exception as e:
                print(f"Error cleaning doc status: {e}")
        
        # Step 7: Clear any caches
        if hasattr(rag_instance, 'llm_response_cache'):
            try:
                await rag_instance.aclear_cache()
                print("Cleared LLM response cache")
            except Exception as e:
                print(f"Error clearing cache: {e}")
        
        # Step 8: Final verification - ensure files are actually updated
        time.sleep(0.5)  # Small delay to ensure filesystem sync
        
        # Verify GraphML file
        graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graphml_path):
            try:
                verify_graph = nx.read_graphml(graphml_path)
                print(f"Final verification: GraphML has {verify_graph.number_of_nodes()} nodes and {verify_graph.number_of_edges()} edges")
            except Exception as e:
                print(f"Warning: Could not verify GraphML file: {e}")
        
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
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                
                if graph and hasattr(graph, 'nodes'):
                    nodes = list(graph.nodes())
                    return nodes
                else:
                    return []
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
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                
                if graph is None:
                    return {"nodes": [], "edges": []}
                
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
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            if graph:
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
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                
                if graph is None:
                    raise ValueError("Could not access graph from storage")
                
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
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                
                if graph is None:
                    raise ValueError("Could not access graph from storage")
                
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
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            status = {
                "exists": True,
                "type": type(graph_storage).__name__,
                "graph_type": type(graph).__name__ if graph else "None",
                "nodes": graph.number_of_nodes() if graph and hasattr(graph, 'number_of_nodes') else 0,
                "edges": graph.number_of_edges() if graph and hasattr(graph, 'number_of_edges') else 0,
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
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            if graph is None:
                raise HTTPException(
                    status_code=500,
                    detail="Could not access graph from storage"
                )
            
            # Get node and edge counts before clearing
            node_count = graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0
            edge_count = graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0
            
            if hasattr(graph, 'clear'):
                print(f"Clearing knowledge graph with {node_count} nodes and {edge_count} edges")
                graph.clear()
                
                # Critical: Save the cleared graph to the GraphML file
                working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
                
                # Force save the cleared graph
                force_save_graph_to_disk(graph, working_dir)
                
                # Also try the storage's save methods
                if hasattr(rag_instance, 'graph_storage'):
                    if hasattr(rag_instance.graph_storage, 'save'):
                        try:
                            await rag_instance.graph_storage.save()
                            print("Also saved using graph_storage.save()")
                        except Exception as e:
                            print(f"Note: graph_storage.save() failed: {e}")
                    
                    if hasattr(rag_instance.graph_storage, '_save'):
                        try:
                            rag_instance.graph_storage._save()
                            print("Also saved using graph_storage._save()")
                        except Exception as e:
                            print(f"Note: graph_storage._save() failed: {e}")
                
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
                                    json.dump(empty_vdb, f, indent=2)
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
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            debug_info["in_memory_graph"] = {
                "exists": True,
                "storage_type": type(graph_storage).__name__,
                "graph_type": type(graph).__name__ if graph else "None",
                "nodes": graph.number_of_nodes() if graph and hasattr(graph, 'number_of_nodes') else 0,
                "edges": graph.number_of_edges() if graph and hasattr(graph, 'number_of_edges') else 0
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
                graph_storage = rag_instance.chunk_entity_relation_graph
                
                # Update the graph in the storage object
                if hasattr(graph_storage, '_graph'):
                    graph_storage._graph = loaded_graph
                elif hasattr(graph_storage, 'graph'):
                    graph_storage.graph = loaded_graph
                elif hasattr(graph_storage, 'set_graph'):
                    graph_storage.set_graph(loaded_graph)
                else:
                    # If storage doesn't have a way to set the graph, replace the whole object
                    rag_instance.chunk_entity_relation_graph = loaded_graph
                
                # Also update the graph storage's reference if it has one
                if hasattr(rag_instance, 'graph_storage'):
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
    """Enhanced text insertion with full metadata support and processing verification"""
    try:
        # Compute document ID
        doc_id = compute_doc_id(request.text)
        
        # Handle source_url processing
        if request.source_url:
            enriched_content = f"Source: {request.source_url}\n"
            if request.description:
                enriched_content += f"Description: {request.description}\n"
            enriched_content += f"\n{request.text}"
        else:
            enriched_content = request.text
        
        # Determine file_path
        if request.document_id:
            if request.document_id.startswith('[') and ']' in request.document_id:
                # Handle [domain] format from n8n
                if request.document_id.startswith('[unknown] '):
                    # Clean up [unknown] prefix
                    cleaned_id = request.document_id.replace('[unknown] ', '')
                    if request.source_url:
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(request.source_url)
                            domain = parsed.netloc
                            file_path = f"[{domain}] {cleaned_id}"
                            full_path = request.source_url  # Store the full URL separately
                        except:
                            file_path = f"[unknown] {cleaned_id}"
                    else:
                        file_path = f"[unknown] {cleaned_id}"
                else:
                    # Use the provided document_id as-is
                    file_path = request.document_id
                    if request.source_url:
                        full_path = request.source_url
            else:
                # Regular document_id, create file_path
                if request.source_url:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(request.source_url)
                        domain = parsed.netloc
                        file_path = f"[{domain}] {request.document_id}"
                        full_path = request.source_url
                    except:
                        file_path = f"text/{request.document_id}.txt"
                else:
                    file_path = f"text/{request.document_id}.txt"
        else:
            # No document_id provided, use computed hash
            if request.source_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(request.source_url)
                    domain = parsed.netloc
                    # Use the last part of the URL path as filename
                    path_parts = parsed.path.strip('/').split('/')
                    page_name = path_parts[-1] if path_parts and path_parts[-1] else 'index'
                    file_path = f"[{domain}] {page_name}"
                    full_path = request.source_url
                except:
                    file_path = f"text/{doc_id}.txt"
            else:
                file_path = f"text/{doc_id}.txt"
        
        # Generate display_name from file_path
        if file_path.startswith('[') and ']' in file_path:
            # For [domain] format, use the full path as display name
            display_name = file_path
        else:
            # For other formats, use file_path or generate one
            if len(file_path) > 50:
                display_name = f"text/{doc_id[:8]}..."  # Shortened ID for display
            else:
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

        # Insert into LightRAG with custom ID and wait for processing to complete
        await rag_instance.ainsert(enriched_content, ids=[custom_id], file_paths=[file_path])
        
        # Wait for processing to complete and verify success
        import asyncio
        max_retries = 30  # Wait up to 30 seconds
        retry_count = 0
        
        while retry_count < max_retries:
            await asyncio.sleep(1)  # Wait 1 second between checks
            
            # Check document status
            if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
                try:
                    doc_status_data = await rag_instance.doc_status.get(custom_id)
                    if doc_status_data:
                        status = doc_status_data.get('status', 'unknown')
                        if status == 'processed':
                            # Successfully processed - now store metadata
                            metadata_store[metadata_key] = metadata_entry
                            save_metadata_store()
                            
                            return {
                                "status": "success",
                                "message": "Document inserted and processed successfully",
                                "doc_id": metadata_key,
                                "file_path": file_path,
                                "processing_status": "completed"
                            }
                        elif status == 'failed':
                            error_msg = doc_status_data.get('error', 'Unknown processing error')
                            raise HTTPException(
                                status_code=500, 
                                detail=f"Document processing failed: {error_msg}"
                            )
                        # If status is 'pending' or 'processing', continue waiting
                except Exception as e:
                    print(f"Error checking document status: {e}")
            
            retry_count += 1
        
        # If we get here, processing timed out
        raise HTTPException(
            status_code=500, 
            detail="Document processing timed out. The document may still be processing in the background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in insert_text_enhanced: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/text")
async def insert_text(request: TextInsertRequest):
    """Standard text insertion with file_path support and processing verification"""
    try:
        # Compute document ID
        doc_id = compute_doc_id(request.text)
        
        # Use provided file_path or create one
        file_path = request.file_path if request.file_path else f"text/{doc_id}.txt"
        
        # Generate display_name from file_path
        display_name = generate_display_name_from_file_path(file_path, doc_id)
        
        # Store metadata
        metadata_entry = {
            "id": doc_id,
            "file_path": file_path,
            "display_name": display_name,
            "description": request.description,
            "indexed_at": datetime.utcnow().isoformat(),
            "content_summary": request.text[:200] + "..." if len(request.text) > 200 else request.text
        }
        
        # Insert into LightRAG with file path and wait for processing
        await rag_instance.ainsert(request.text, file_paths=[file_path])
        
        # Wait for processing to complete and verify success
        import asyncio
        max_retries = 30  # Wait up to 30 seconds
        retry_count = 0
        
        while retry_count < max_retries:
            await asyncio.sleep(1)  # Wait 1 second between checks
            
            # Check document status
            if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
                try:
                    doc_status_data = await rag_instance.doc_status.get(doc_id)
                    if doc_status_data:
                        status = doc_status_data.get('status', 'unknown')
                        if status == 'processed':
                            # Successfully processed - now store metadata
                            metadata_store[doc_id] = metadata_entry
                            save_metadata_store()
                            
                            return {
                                "status": "success",
                                "message": "Document inserted and processed successfully",
                                "doc_id": doc_id,
                                "file_path": file_path,
                                "processing_status": "completed"
                            }
                        elif status == 'failed':
                            error_msg = doc_status_data.get('error', 'Unknown processing error')
                            raise HTTPException(
                                status_code=500, 
                                detail=f"Document processing failed: {error_msg}"
                            )
                        # If status is 'pending' or 'processing', continue waiting
                except Exception as e:
                    print(f"Error checking document status: {e}")
            
            retry_count += 1
        
        # If we get here, processing timed out
        raise HTTPException(
            status_code=500, 
            detail="Document processing timed out. The document may still be processing in the background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in insert_text: {e}")
        import traceback
        traceback.print_exc()
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
                                    "status": doc_data.get('status', 'unknown'),
                                    "error": doc_data.get('error') if doc_data.get('status') == 'failed' else None
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
                                    "status": doc_data.get('status', 'unknown'),
                                    "error": doc_data.get('error') if doc_data.get('status') == 'failed' else None
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
                                        "status": "processed",  # From metadata store, assume processed
                                        "error": None
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
        
        # Categorize documents by status
        processed = [doc for doc in documents if doc.get('status') == 'processed']
        pending = [doc for doc in documents if doc.get('status') == 'pending']
        processing = [doc for doc in documents if doc.get('status') == 'processing']
        failed = [doc for doc in documents if doc.get('status') == 'failed']
        unknown = [doc for doc in documents if doc.get('status') not in ['processed', 'pending', 'processing', 'failed']]
        
        return {
            "statuses": {
                "processed": processed,
                "pending": pending,
                "processing": processing,
                "failed": failed,
                "unknown": unknown
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
                # Check if this document was inserted with a custom ID
                metadata_keys_to_delete.append(doc_id)
                
                # For LightRAG deletion, we need to use the actual ID that LightRAG is using
                # This could be the doc_id itself (if it's a custom ID like [domain] path)
                # or it could be in the metadata
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
            
            # First clean up all document traces from LightRAG storages
            # This will also clear and save the graph properly
            await cleanup_all_document_traces(unique_docs_to_delete)
            
            # Then call the standard delete method
            try:
                await rag_instance.adelete_by_doc_id(unique_docs_to_delete)
            except Exception as e:
                print(f"Error in adelete_by_doc_id: {e}")
                # Continue even if this fails, as cleanup_all_document_traces should have done most of the work
            
            # Remove from metadata store using the correct keys
            for key in metadata_keys_to_delete:
                if key in metadata_store:
                    del metadata_store[key]
            
            # Save updated metadata
            save_metadata_store()
            
            # Force a final save of the graph to ensure persistence
            working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                if graph:
                    force_save_graph_to_disk(graph, working_dir)
            
            # Also ensure all vector DB files are saved with proper structure
            print("Ensuring all vector database files are properly saved...")
            vdb_files = ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"]
            for file_name in vdb_files:
                file_path = os.path.join(working_dir, file_name)
                if os.path.exists(file_path):
                    try:
                        # Read and re-save to ensure it's properly formatted
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Ensure proper structure
                        if not isinstance(data, dict):
                            data = {"embedding_dim": 1536, "data": [], "matrix": []}
                        if 'data' not in data:
                            data['data'] = []
                        if 'matrix' not in data:
                            data['matrix'] = []
                        if 'embedding_dim' not in data:
                            data['embedding_dim'] = 1536
                            
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"Verified and saved {file_name}")
                    except Exception as e:
                        print(f"Error verifying {file_name}: {e}")
                        # Create empty structure if corrupted
                        empty_structure = {"embedding_dim": 1536, "data": [], "matrix": []}
                        with open(file_path, 'w') as f:
                            json.dump(empty_structure, f, indent=2)
                        print(f"Reset {file_name} to empty structure")
            
            # Force save all LightRAG storages
            if hasattr(rag_instance, 'graph_storage'):
                storage = rag_instance.graph_storage
                if hasattr(storage, 'save'):
                    try:
                        await storage.save()
                        print("Force saved graph storage")
                    except Exception as e:
                        print(f"Error force saving graph storage: {e}")
                elif hasattr(storage, '_save'):
                    try:
                        storage._save()
                        print("Force saved graph storage (sync)")
                    except Exception as e:
                        print(f"Error force saving graph storage (sync): {e}")
            
            # Small delay to ensure filesystem sync
            time.sleep(0.5)
        
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
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            if graph and hasattr(graph, 'clear'):
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

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    custom_name: Optional[str] = Form(None)
):
    """Upload and process a document file
    
    Args:
        file: The uploaded file
        custom_name: Optional custom name prefix for the file
    
    Returns:
        Document metadata including ID and file path
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Get file extension
        filename = file.filename
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
        
        # Validate file type
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        
        # Extract text from file
        extracted_text = extract_text_from_file(file_content, file_extension, filename)
        
        # Compute document ID
        doc_id = compute_doc_id(extracted_text)
        
        # Generate file path with custom name if provided
        if custom_name:
            # Format: [custom_name] original_filename
            file_path = f"[{custom_name}] {filename}"
            display_name = file_path
        else:
            file_path = f"uploaded/{filename}"
            display_name = filename
        
        # Add metadata header to content
        metadata_parts = [
            f"[FILE: {filename}]",
            f"[TYPE: {file_extension.upper()}]",
            f"[UPLOADED: {datetime.utcnow().isoformat()}]"
        ]
        if custom_name:
            metadata_parts.append(f"[CUSTOM_NAME: {custom_name}]")
        
        enriched_content = "\n".join(metadata_parts) + "\n\n" + extracted_text
        
        # Store metadata
        metadata_entry = {
            "id": doc_id,
            "file_path": file_path,
            "display_name": display_name,
            "original_filename": filename,
            "custom_name": custom_name,
            "file_type": file_extension,
            "indexed_at": datetime.utcnow().isoformat(),
            "content_summary": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        }
        
        metadata_store[doc_id] = metadata_entry
        save_metadata_store()
        
        # Insert into LightRAG
        await rag_instance.ainsert(enriched_content, ids=[doc_id], file_paths=[file_path])
        
        return {
            "status": "success",
            "message": f"Document '{filename}' uploaded and processed successfully",
            "doc_id": doc_id,
            "file_path": file_path,
            "display_name": display_name,
            "file_type": file_extension,
            "text_length": len(extracted_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload/bulk")
async def upload_documents_bulk(
    files: List[UploadFile] = File(...),
    custom_name: Optional[str] = Form(None)
):
    """Upload and process multiple document files"""
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file using existing logic
            file_content = await file.read()
            filename = file.filename
            file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
            
            if file_extension not in SUPPORTED_EXTENSIONS:
                results.append({
                    "filename": filename,
                    "status": "error",
                    "error": f"Unsupported file type: {file_extension}"
                })
                continue
            
            # Extract text and process (existing logic)
            extracted_text = extract_text_from_file(file_content, file_extension, filename)
            doc_id = compute_doc_id(extracted_text)
            
            # Generate file path
            if custom_name:
                file_path = f"[{custom_name}] {filename}"
            else:
                file_path = f"bulk_upload/{filename}"
            
            # Add metadata
            metadata_parts = [
                f"[FILE: {filename}]",
                f"[TYPE: {file_extension.upper()}]",
                f"[BULK_UPLOAD: {i+1} of {len(files)}]",
                f"[UPLOADED: {datetime.utcnow().isoformat()}]"
            ]
            
            enriched_content = "\n".join(metadata_parts) + "\n\n" + extracted_text
            
            # Store metadata
            metadata_entry = {
                "id": doc_id,
                "file_path": file_path,
                "display_name": filename,
                "original_filename": filename,
                "file_type": file_extension,
                "indexed_at": datetime.utcnow().isoformat(),
                "bulk_upload": True,
                "batch_index": i + 1,
                "batch_total": len(files)
            }
            
            metadata_store[doc_id] = metadata_entry
            
            # Insert into LightRAG
            await rag_instance.ainsert(enriched_content, ids=[doc_id], file_paths=[file_path])
            
            results.append({
                "filename": filename,
                "status": "success",
                "doc_id": doc_id,
                "file_path": file_path
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    # Save metadata after all uploads
    save_metadata_store()
    
    return {
        "status": "completed",
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }


@app.get("/documents/sources")
async def list_document_sources():
    """
    List all available documentation sources/libraries
    Returns unique sources extracted from document IDs in format [source] filename
    """
    try:
        sources_map = {}  # source -> {count, example_docs, description}
        processed_doc_ids = set()  # Track which doc_ids we've already counted
        
        # Use the EXACT same logic as /documents endpoint
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            try:
                doc_status_storage = rag_instance.doc_status
                all_docs = None
                
                # Method 1: Try to get all documents directly
                if hasattr(doc_status_storage, 'get_all'):
                    try:
                        all_docs = await doc_status_storage.get_all()
                    except Exception as e:
                        print(f"Error with get_all method: {e}")
                
                # Method 2: Try to iterate through storage if it's dict-like
                if not all_docs and hasattr(doc_status_storage, '_data'):
                    try:
                        storage_data = doc_status_storage._data
                        if isinstance(storage_data, dict):
                            all_docs = storage_data
                    except Exception as e:
                        print(f"Error accessing _data: {e}")
                
                # Method 3: Try JSON storage file directly
                if not all_docs:
                    try:
                        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                        doc_status_file = os.path.join(working_dir, "doc_status.json")
                        if os.path.exists(doc_status_file):
                            with open(doc_status_file, 'r') as f:
                                all_docs = json.load(f)
                    except Exception as e:
                        print(f"Error reading doc_status.json: {e}")
                
                # Process documents if we found any
                if all_docs:
                    for doc_id, doc_data in all_docs.items():
                        # Check if doc_id matches [source] pattern
                        if doc_id.startswith('[') and ']' in doc_id:
                            source_name = doc_id.split(']')[0][1:]
                            processed_doc_ids.add(doc_id)
                            
                            if source_name not in sources_map:
                                sources_map[source_name] = {
                                    "source": source_name,
                                    "document_count": 0,
                                    "example_files": [],
                                    "description": "",
                                    "last_indexed": ""
                                }
                            
                            sources_map[source_name]["document_count"] += 1
                            
                            # Add example file (limit to 5)
                            if len(sources_map[source_name]["example_files"]) < 5:
                                # Extract filename after ]
                                filename = doc_id.split(']', 1)[1].strip() if ']' in doc_id else doc_id
                                sources_map[source_name]["example_files"].append(filename)
                            
                            # Update last indexed time if available
                            if isinstance(doc_data, dict):
                                updated_at = doc_data.get('updated_at', '')
                                if updated_at and (not sources_map[source_name]["last_indexed"] or 
                                                 updated_at > sources_map[source_name]["last_indexed"]):
                                    sources_map[source_name]["last_indexed"] = updated_at
                                    
            except Exception as e:
                print(f"Error accessing doc_status storage: {e}")
                import traceback
                traceback.print_exc()
        
        # Also check metadata store
        for doc_id, metadata in metadata_store.items():
            if doc_id in processed_doc_ids:
                continue
                
            if doc_id.startswith('[') and ']' in doc_id:
                source_name = doc_id.split(']')[0][1:]
                
                if source_name not in sources_map:
                    sources_map[source_name] = {
                        "source": source_name,
                        "document_count": 0,
                        "example_files": [],
                        "description": metadata.get('description', ''),
                        "last_indexed": metadata.get('indexed_at', '')
                    }
                
                sources_map[source_name]["document_count"] += 1
                
                if len(sources_map[source_name]["example_files"]) < 5:
                    filename = doc_id.split(']', 1)[1].strip() if ']' in doc_id else doc_id
                    sources_map[source_name]["example_files"].append(filename)
                
                indexed_at = metadata.get('indexed_at', '')
                if indexed_at and indexed_at > sources_map[source_name]["last_indexed"]:
                    sources_map[source_name]["last_indexed"] = indexed_at
        
        # Convert to list and sort by document count
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

@app.get("/debug/sources")
async def debug_sources():
    """Debug endpoint to understand why sources aren't being detected"""
    debug_info = {
        "metadata_store_count": len(metadata_store),
        "metadata_store_sample": [],
        "doc_status_methods": [],
        "doc_status_sample": [],
        "sources_found": []
    }
    
    # Check metadata store
    for i, (doc_id, metadata) in enumerate(metadata_store.items()):
        if i < 5:  # Show first 5
            debug_info["metadata_store_sample"].append({
                "doc_id": doc_id,
                "has_bracket": doc_id.startswith('[') and ']' in doc_id,
                "file_path": metadata.get('file_path', 'N/A')
            })
    
    # Check doc_status with all methods
    if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
        doc_status_storage = rag_instance.doc_status
        
        # Method 1: get_all
        if hasattr(doc_status_storage, 'get_all'):
            try:
                all_docs = await doc_status_storage.get_all()
                debug_info["doc_status_methods"].append({
                    "method": "get_all",
                    "success": True,
                    "doc_count": len(all_docs) if all_docs else 0,
                    "type": type(all_docs).__name__ if all_docs else "None"
                })
                
                # Sample docs
                if all_docs:
                    for i, (doc_id, doc_data) in enumerate(all_docs.items()):
                        if i < 5:
                            debug_info["doc_status_sample"].append({
                                "doc_id": doc_id,
                                "has_bracket": doc_id.startswith('[') and ']' in doc_id,
                                "source": doc_id.split(']')[0][1:] if doc_id.startswith('[') and ']' in doc_id else "N/A"
                            })
                            
                            # Extract sources
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
        
        # Method 2: _data
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
        
        # Method 3: JSON file
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

@app.get("/documents/by-source/{source}")
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
        matching_docs = []
        source_pattern = f"[{source}]"
        
        # Search in metadata store
        for doc_id, metadata in metadata_store.items():
            # PRIORITIZE checking doc_id first
            if doc_id.startswith(source_pattern):
                # Extract filename after the source pattern
                filename = doc_id[len(source_pattern):].strip()
                
                matching_docs.append({
                    "id": doc_id,  # Use the actual doc_id as the display ID
                    "doc_id": doc_id,
                    "file_path": metadata.get('file_path', f"text/{doc_id}.txt"),
                    "filename": filename,
                    "display_name": metadata.get('display_name', doc_id),
                    "indexed_at": metadata.get('indexed_at'),
                    "status": "processed"
                })
            else:
                # Only check file_path if doc_id doesn't match
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
        
        # Also check doc_status (same logic as /documents/sources)
        processed_doc_ids = {doc["doc_id"] for doc in matching_docs}  # Track what we already found
        
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            try:
                doc_status_storage = rag_instance.doc_status
                all_docs = None
                
                # Method 1: Try to get all documents directly
                if hasattr(doc_status_storage, 'get_all'):
                    try:
                        all_docs = await doc_status_storage.get_all()
                    except Exception as e:
                        print(f"Error with get_all method: {e}")
                
                # Method 2: Try to iterate through storage if it's dict-like
                if not all_docs and hasattr(doc_status_storage, '_data'):
                    try:
                        storage_data = doc_status_storage._data
                        if isinstance(storage_data, dict):
                            all_docs = storage_data
                    except Exception as e:
                        print(f"Error accessing _data: {e}")
                
                # Method 3: Try JSON storage file directly
                if not all_docs:
                    try:
                        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
                        doc_status_file = os.path.join(working_dir, "doc_status.json")
                        if os.path.exists(doc_status_file):
                            with open(doc_status_file, 'r') as f:
                                all_docs = json.load(f)
                    except Exception as e:
                        print(f"Error reading doc_status.json: {e}")
                
                # Process documents if we found any
                if all_docs:
                    for doc_id, doc_data in all_docs.items():
                        # Skip if we already processed this doc from metadata_store
                        if doc_id in processed_doc_ids:
                            continue
                        
                        # Check if doc_id matches [source] pattern
                        if doc_id.startswith(source_pattern) or doc_id.startswith(source_pattern + " "):
                            filename = doc_id[len(source_pattern):].strip()
                            
                            matching_docs.append({
                                "id": doc_id,
                                "doc_id": doc_id,
                                "file_path": doc_id,  # Use doc_id as file_path since we don't have metadata
                                "filename": filename,
                                "display_name": doc_id,
                                "indexed_at": doc_data.get('indexed_at', '') if isinstance(doc_data, dict) else '',
                                "status": doc_data.get('status', 'processed') if isinstance(doc_data, dict) else 'processed'
                            })
            except Exception as e:
                print(f"Error checking doc_status: {e}")
        
        # Sort by filename
        matching_docs.sort(key=lambda x: x["filename"])
        
        # Return in the same format as /documents
        return {
            "source": source,
            "documents": matching_docs,
            "count": len(matching_docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/by-source/{source}")
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
        # Find all documents with this source
        docs_to_delete = []
        metadata_keys_to_delete = []
        source_pattern = f"[{source}]"
        
        for doc_id, metadata in metadata_store.items():
            # PRIORITIZE checking doc_id first
            if doc_id.startswith(source_pattern):
                metadata_keys_to_delete.append(doc_id)
                docs_to_delete.append(doc_id)
                
                # Also add any variations that might exist
                original_id = metadata.get('original_doc_id')
                if original_id and original_id != doc_id:
                    docs_to_delete.append(original_id)
            else:
                # Only check file_path if doc_id doesn't match
                file_path = metadata.get('file_path', '')
                if file_path.startswith(source_pattern):
                    metadata_keys_to_delete.append(doc_id)
                    docs_to_delete.append(doc_id)
                    
                    # Add variations
                    original_id = metadata.get('original_doc_id', doc_id)
                    if original_id not in docs_to_delete:
                        docs_to_delete.append(original_id)
                    if file_path not in docs_to_delete:
                        docs_to_delete.append(file_path)
        
        if docs_to_delete:
            unique_docs_to_delete = list(set(docs_to_delete))
            
            # Clean up all document traces
            await cleanup_all_document_traces(unique_docs_to_delete)
            
            # Call standard delete
            try:
                await rag_instance.adelete_by_doc_id(unique_docs_to_delete)
            except Exception as e:
                print(f"Error in adelete_by_doc_id: {e}")
            
            # Remove from metadata store
            for key in metadata_keys_to_delete:
                if key in metadata_store:
                    del metadata_store[key]
            
            # Save updated metadata
            save_metadata_store()
            
            # Force save the graph
            working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
            if hasattr(rag_instance, 'chunk_entity_relation_graph'):
                graph_storage = rag_instance.chunk_entity_relation_graph
                graph = get_graph_from_storage(graph_storage)
                if graph:
                    force_save_graph_to_disk(graph, working_dir)
            
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

@app.post("/query")
async def query_with_optional_filtering(request: FilteredQueryRequest):
    """
    Query the RAG system with optional source filtering
    """
    try:
        # If no sources specified, use standard query
        if not request.sources:
            # Only pass the parameters that QueryParam actually accepts
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
        
        # If sources are specified, filter the search
        allowed_doc_ids = set()
        source_patterns = [f"[{source}]" for source in request.sources]
        
        # Collect doc_ids from metadata store
        for doc_id, metadata in metadata_store.items():
            file_path = metadata.get('file_path', doc_id)
            for pattern in source_patterns:
                # Check both with and without space after bracket for compatibility
                if (doc_id.startswith(pattern) or doc_id.startswith(pattern + " ") or 
                    file_path.startswith(pattern) or file_path.startswith(pattern + " ")):
                    allowed_doc_ids.add(doc_id)
                    if file_path != doc_id:
                        allowed_doc_ids.add(file_path)
                    break
        
        # Also check doc_status
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            try:
                all_docs = await rag_instance.doc_status.get_all()
                for doc_id in all_docs.keys():
                    for pattern in source_patterns:
                        # Check both with and without space after bracket for compatibility
                        if doc_id.startswith(pattern) or doc_id.startswith(pattern + " "):
                            allowed_doc_ids.add(doc_id)
                            break
            except:
                pass
        
        if not allowed_doc_ids:
            return {
                "query": request.query,
                "response": f"No documents found for the specified sources: {', '.join(request.sources)}",
                "mode": request.mode,
                "sources_filtered": request.sources,
                "documents_found": 0
            }
        
        # Enhance query with source context
        source_context = f"Focus on information from these sources: {', '.join(request.sources)}. "
        enhanced_query = source_context + request.query
        
        # Perform the query
        param = QueryParam(
            mode=request.mode,
            stream=request.stream,
            only_need_context=request.only_need_context,
            response_type=request.response_type,
            top_k=request.top_k,
            max_token_for_text_unit=request.max_token_for_text_unit,
            max_token_for_global_context=request.max_token_for_global_context,
            max_token_for_local_context=request.max_token_for_local_context
        )
        
        result = await rag_instance.aquery(enhanced_query, param=param)
        
        return {
            "query": request.query,
            "response": result,
            "mode": request.mode,
            "sources_filtered": request.sources,
            "documents_found": len(allowed_doc_ids)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streaming version
@app.post("/query/stream")
async def query_stream_with_optional_filtering(request: FilteredQueryRequest):
    """
    Stream query responses with optional source filtering
    
    This extends the standard /query/stream endpoint to support optional source filtering.
    """
    try:
        # Set stream to True for this endpoint
        request.stream = True
        
        # If no sources specified, use standard streaming query
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
            
            # This would need to be implemented as a proper streaming response
            # For now, returning a simple response
            result = await rag_instance.aquery(request.query, param=param)
            
            return {
                "query": request.query,
                "response": result,
                "mode": request.mode,
                "stream": True
            }
        
        # Similar filtering logic as above but for streaming
        # Implementation would depend on your streaming infrastructure
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint following graph endpoint patterns
@app.get("/graph/sources/{source}/stats")
async def get_source_graph_statistics(source: str):
    """
    Get graph statistics for a specific source
    
    Following the pattern of graph endpoints like /graph/label/list
    
    Returns information about the knowledge graph coverage for this source.
    """
    try:
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
        
        # Count documents and gather stats
        doc_ids_in_source = set()
        
        for doc_id, metadata in metadata_store.items():
            # PRIORITIZE checking doc_id first
            if doc_id.startswith(source_pattern):
                doc_ids_in_source.add(doc_id)
                stats["document_count"] += 1
                
                # Track file types from file_path
                file_path = metadata.get('file_path', doc_id)
                if '.' in file_path:
                    ext = file_path.split('.')[-1].lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                
                # Track dates
                indexed_at = metadata.get('indexed_at', '')
                if indexed_at:
                    if not stats["latest_update"] or indexed_at > stats["latest_update"]:
                        stats["latest_update"] = indexed_at
                    if not stats["oldest_document"] or indexed_at < stats["oldest_document"]:
                        stats["oldest_document"] = indexed_at
            else:
                # Only check file_path if doc_id doesn't match
                file_path = metadata.get('file_path', '')
                if file_path.startswith(source_pattern):
                    doc_ids_in_source.add(doc_id)
                    stats["document_count"] += 1
                    
                    # Track file types
                    if '.' in file_path:
                        ext = file_path.split('.')[-1].lower()
                        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
                    # Track dates
                    indexed_at = metadata.get('indexed_at', '')
                    if indexed_at:
                        if not stats["latest_update"] or indexed_at > stats["latest_update"]:
                            stats["latest_update"] = indexed_at
                        if not stats["oldest_document"] or indexed_at < stats["oldest_document"]:
                            stats["oldest_document"] = indexed_at
        
        # Get chunk and entity counts from vector databases
        working_dir = os.getenv("WORKING_DIR", "/app/data/rag_storage")
        
        # Check chunks
        chunks_file = os.path.join(working_dir, "vdb_chunks.json")
        if os.path.exists(chunks_file):
            try:
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                    if 'data' in chunks_data:
                        for chunk in chunks_data['data']:
                            if chunk.get('source_id', '') in doc_ids_in_source:
                                stats["total_chunks"] += 1
            except:
                pass
        
        # Check entities and relationships
        if hasattr(rag_instance, 'chunk_entity_relation_graph'):
            graph_storage = rag_instance.chunk_entity_relation_graph
            graph = get_graph_from_storage(graph_storage)
            
            if graph:
                # Count entities and relationships from this source
                for node_id, node_data in graph.nodes(data=True):
                    source_id = node_data.get('source_id', '')
                    for doc_id in doc_ids_in_source:
                        if doc_id in source_id:
                            stats["entities_count"] += 1
                            break
                
                for src, tgt, edge_data in graph.edges(data=True):
                    source_id = edge_data.get('source_id', '')
                    for doc_id in doc_ids_in_source:
                        if doc_id in source_id:
                            stats["relationships_count"] += 1
                            break
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/status/{doc_id}")
async def get_document_status(doc_id: str):
    """Get the processing status of a specific document"""
    try:
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            doc_status_data = await rag_instance.doc_status.get(doc_id)
            if doc_status_data:
                return {
                    "doc_id": doc_id,
                    "status": doc_status_data.get('status', 'unknown'),
                    "error": doc_status_data.get('error'),
                    "created_at": doc_status_data.get('created_at'),
                    "updated_at": doc_status_data.get('updated_at'),
                    "content_length": doc_status_data.get('content_length'),
                    "chunks_count": doc_status_data.get('chunks_count')
                }
            else:
                raise HTTPException(status_code=404, detail="Document not found")
        else:
            raise HTTPException(status_code=500, detail="Document status storage not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/status")
async def get_all_document_statuses():
    """Get processing status of all documents"""
    try:
        if hasattr(rag_instance, 'doc_status') and rag_instance.doc_status is not None:
            # Get documents by status
            pending_docs = await rag_instance.doc_status.get_docs_by_status("pending")
            processing_docs = await rag_instance.doc_status.get_docs_by_status("processing") 
            processed_docs = await rag_instance.doc_status.get_docs_by_status("processed")
            failed_docs = await rag_instance.doc_status.get_docs_by_status("failed")
            
            return {
                "statuses": {
                    "pending": {
                        "count": len(pending_docs),
                        "documents": list(pending_docs.keys())
                    },
                    "processing": {
                        "count": len(processing_docs),
                        "documents": list(processing_docs.keys())
                    },
                    "processed": {
                        "count": len(processed_docs),
                        "documents": list(processed_docs.keys())
                    },
                    "failed": {
                        "count": len(failed_docs),
                        "documents": [{"id": doc_id, "error": doc_data.get('error', 'Unknown error')} 
                                    for doc_id, doc_data in failed_docs.items()]
                    }
                },
                "total": len(pending_docs) + len(processing_docs) + len(processed_docs) + len(failed_docs)
            }
        else:
            raise HTTPException(status_code=500, detail="Document status storage not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup code for running directly
if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    
    print(f"Starting LightRAG Extended API Server on {host}:{port}")
    print(f"Working directory: {os.getenv('WORKING_DIR', '/app/data/rag_storage')}")
    
    # Run the server
    uvicorn.run(
        "lightrag_extended_api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

    # test comment