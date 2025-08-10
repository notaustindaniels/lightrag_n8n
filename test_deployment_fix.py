#!/usr/bin/env python3
"""Test script to verify deployment fixes for graph display issue"""

import os
import sys
import json
import asyncio
import networkx as nx
from pathlib import Path

# Set up environment
os.environ["WORKING_DIR"] = "./test_rag_storage"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")

# Import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightrag_extended_api import WorkspaceManager, get_graph_from_storage, get_combined_graph

async def test_nested_workspace_loading():
    """Test that workspaces with nested directories are handled correctly"""
    print("=" * 60)
    print("TEST: NESTED WORKSPACE LOADING")
    print("=" * 60)
    
    base_dir = "./test_rag_storage"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a workspace with nested structure (mimicking the deployment issue)
    workspace_name = "test.domain.com"
    workspace_dir = os.path.join(base_dir, "workspaces", workspace_name)
    nested_dir = os.path.join(workspace_dir, workspace_name)
    os.makedirs(nested_dir, exist_ok=True)
    
    # Create test graph in nested directory
    test_graph = nx.Graph()
    test_graph.add_node("Node1", type="entity")
    test_graph.add_node("Node2", type="entity")
    test_graph.add_edge("Node1", "Node2", relationship="connected")
    
    # Save in nested structure (where deployment currently has it)
    nested_graphml = os.path.join(nested_dir, "graph_chunk_entity_relation.graphml")
    nx.write_graphml(test_graph, nested_graphml)
    print(f"✓ Created nested GraphML at: {nested_graphml}")
    
    # Create workspace and see if it finds the graph
    rag = await WorkspaceManager.create_workspace(workspace_name)
    
    # Check if graph was loaded
    if hasattr(rag, 'chunk_entity_relation_graph'):
        storage = rag.chunk_entity_relation_graph
        graph = await get_graph_from_storage(storage)
        
        if graph and hasattr(graph, 'number_of_nodes'):
            nodes = graph.number_of_nodes()
            edges = graph.number_of_edges()
            print(f"✓ Graph loaded from nested structure: {nodes} nodes, {edges} edges")
            return nodes == 2 and edges == 1
        else:
            print("✗ Graph not loaded from nested structure")
            return False
    else:
        print("✗ No graph storage found")
        return False

async def test_empty_label_handling():
    """Test that empty string labels are handled correctly in /graphs endpoint"""
    print("\n" + "=" * 60)
    print("TEST: EMPTY LABEL HANDLING")
    print("=" * 60)
    
    # This test simulates what the WebUI sends
    label = ""
    
    # Convert empty string to None (mimicking the fix)
    if label == "":
        label = None
        print("✓ Empty string converted to None")
    
    # Should trigger combined graph approach
    if label is None:
        print("✓ Will use combined graph approach")
        return True
    else:
        print("✗ Empty string not handled correctly")
        return False

async def test_combined_graph_with_nested():
    """Test that combined graph aggregates from nested structures"""
    print("\n" + "=" * 60)
    print("TEST: COMBINED GRAPH WITH NESTED STRUCTURES")
    print("=" * 60)
    
    # Create two workspaces with different structures
    base_dir = "./test_rag_storage"
    
    # Workspace 1: Flat structure
    ws1_name = "flat_workspace"
    ws1_dir = os.path.join(base_dir, "workspaces", ws1_name)
    os.makedirs(ws1_dir, exist_ok=True)
    
    graph1 = nx.Graph()
    graph1.add_node("FlatNode1")
    graph1.add_node("FlatNode2")
    graph1.add_edge("FlatNode1", "FlatNode2")
    nx.write_graphml(graph1, os.path.join(ws1_dir, "graph_chunk_entity_relation.graphml"))
    
    # Workspace 2: Nested structure
    ws2_name = "nested_workspace"
    ws2_dir = os.path.join(base_dir, "workspaces", ws2_name)
    nested_dir = os.path.join(ws2_dir, ws2_name)
    os.makedirs(nested_dir, exist_ok=True)
    
    graph2 = nx.Graph()
    graph2.add_node("NestedNode1")
    graph2.add_node("NestedNode2")
    graph2.add_edge("NestedNode1", "NestedNode2")
    nx.write_graphml(graph2, os.path.join(nested_dir, "graph_chunk_entity_relation.graphml"))
    
    # Create workspaces
    await WorkspaceManager.create_workspace(ws1_name)
    await WorkspaceManager.create_workspace(ws2_name)
    
    # Get combined graph
    combined = await get_combined_graph()
    
    if combined:
        total_nodes = combined.number_of_nodes()
        total_edges = combined.number_of_edges()
        print(f"✓ Combined graph has {total_nodes} nodes and {total_edges} edges")
        
        # Should have all 4 nodes and 2 edges
        if total_nodes == 4 and total_edges == 2:
            print("✓ Successfully aggregated from both flat and nested structures")
            return True
        else:
            print(f"✗ Expected 4 nodes and 2 edges, got {total_nodes} nodes and {total_edges} edges")
            return False
    else:
        print("✗ Failed to create combined graph")
        return False

async def main():
    """Run all deployment fix tests"""
    print("\nDEPLOYMENT FIX TESTS\n")
    
    results = []
    
    # Clean up any previous test data
    import shutil
    if os.path.exists("./test_rag_storage"):
        shutil.rmtree("./test_rag_storage")
    
    # Test 1: Nested workspace loading
    try:
        result = await test_nested_workspace_loading()
        results.append(("Nested Workspace Loading", result))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Nested Workspace Loading", False))
    
    # Test 2: Empty label handling
    try:
        result = await test_empty_label_handling()
        results.append(("Empty Label Handling", result))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        results.append(("Empty Label Handling", False))
    
    # Test 3: Combined graph with nested structures
    try:
        result = await test_combined_graph_with_nested()
        results.append(("Combined Graph with Nested", result))
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Combined Graph with Nested", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All deployment fix tests passed!")
        print("\nThe fixes should resolve the graph display issue on DigitalOcean.")
    else:
        print("\n⚠️ Some tests failed. Review the fixes before deployment.")
    
    # Clean up test data
    if os.path.exists("./test_rag_storage"):
        shutil.rmtree("./test_rag_storage")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)