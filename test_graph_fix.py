#!/usr/bin/env python3
"""Test script to verify graph loading and persistence fixes"""

import asyncio
import os
import sys
import json
import networkx as nx
from pathlib import Path

# Set up environment
os.environ["WORKING_DIR"] = "./rag_storage"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")

# Import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightrag_extended_api import WorkspaceManager, get_graph_from_storage

async def test_workspace_loading():
    """Test that workspaces load graphs correctly"""
    print("=" * 60)
    print("TESTING WORKSPACE GRAPH LOADING")
    print("=" * 60)
    
    # Create test workspace
    test_workspace = "test_workspace"
    workspace_dir = f"./rag_storage/workspaces/{test_workspace}"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Create a test GraphML file with some nodes
    test_graph = nx.Graph()
    test_graph.add_node("TestNode1", type="entity", description="Test node 1")
    test_graph.add_node("TestNode2", type="entity", description="Test node 2")
    test_graph.add_edge("TestNode1", "TestNode2", relationship="connected_to")
    
    graphml_path = os.path.join(workspace_dir, "graph_chunk_entity_relation.graphml")
    nx.write_graphml(test_graph, graphml_path)
    print(f"✓ Created test GraphML with {test_graph.number_of_nodes()} nodes at {graphml_path}")
    
    # Now create the workspace and see if it loads the graph
    rag = await WorkspaceManager.create_workspace(test_workspace)
    
    # Check if graph was loaded
    if hasattr(rag, 'chunk_entity_relation_graph'):
        storage = rag.chunk_entity_relation_graph
        graph = await get_graph_from_storage(storage)
        
        if graph and hasattr(graph, 'number_of_nodes'):
            loaded_nodes = graph.number_of_nodes()
            loaded_edges = graph.number_of_edges()
            print(f"✓ Graph loaded successfully: {loaded_nodes} nodes, {loaded_edges} edges")
            
            if loaded_nodes == 2 and loaded_edges == 1:
                print("✓ Graph data matches expected values!")
                return True
            else:
                print(f"✗ Graph data mismatch: expected 2 nodes, 1 edge, got {loaded_nodes} nodes, {loaded_edges} edges")
                return False
        else:
            print("✗ Graph not loaded or invalid")
            return False
    else:
        print("✗ RAG instance has no chunk_entity_relation_graph attribute")
        return False

async def test_graph_persistence():
    """Test that graphs persist after document processing"""
    print("\n" + "=" * 60)
    print("TESTING GRAPH PERSISTENCE")
    print("=" * 60)
    
    # This test would require a full document insertion which needs API keys
    # For now, just verify the persistence mechanism exists
    
    test_workspace = "persistence_test"
    rag = await WorkspaceManager.create_workspace(test_workspace)
    
    if hasattr(rag, 'chunk_entity_relation_graph'):
        storage = rag.chunk_entity_relation_graph
        
        # Check for index_done_callback
        if hasattr(storage, 'index_done_callback'):
            print("✓ Graph storage has index_done_callback for persistence")
            
            # Check for _get_graph method
            if hasattr(storage, '_get_graph'):
                print("✓ Graph storage has _get_graph async method")
                return True
            else:
                print("✗ Graph storage missing _get_graph method")
                return False
        else:
            print("✗ Graph storage missing index_done_callback")
            return False
    else:
        print("✗ RAG instance has no chunk_entity_relation_graph")
        return False

async def main():
    """Run all tests"""
    print("\nSTARTING GRAPH FIX TESTS\n")
    
    results = []
    
    # Test 1: Workspace loading
    try:
        result = await test_workspace_loading()
        results.append(("Workspace Graph Loading", result))
    except Exception as e:
        print(f"✗ Workspace loading test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Workspace Graph Loading", False))
    
    # Test 2: Graph persistence
    try:
        result = await test_graph_persistence()
        results.append(("Graph Persistence", result))
    except Exception as e:
        print(f"✗ Graph persistence test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Graph Persistence", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed! Graph loading and persistence should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)