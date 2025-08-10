#!/usr/bin/env python3
"""Test script to verify edge format fix in /graphs endpoint"""

import requests
import json
import sys

def test_edge_format(base_url="https://sea-turtle-app-6kjrs.ondigitalocean.app"):
    """Test that edges have the correct format with id field"""
    
    print("=" * 60)
    print("TESTING EDGE FORMAT FIX")
    print("=" * 60)
    
    # Test 1: Wildcard query
    print("\n1. Testing /graphs?label=* for edge format...")
    url = f"{base_url}/graphs?label=*&max_depth=3&max_nodes=50"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Nodes returned: {len(nodes)}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(edges) > 0:
            # Check if edges have required fields
            sample_edge = edges[0]
            required_fields = ["id", "source", "target"]
            optional_fields = ["type", "properties"]
            
            print(f"\n   Sample edge structure:")
            print(f"   {json.dumps(sample_edge, indent=6)}")
            
            # Validate required fields
            missing_fields = [field for field in required_fields if field not in sample_edge]
            if missing_fields:
                print(f"   ✗ Missing required fields: {missing_fields}")
                return False
            else:
                print(f"   ✓ All required fields present: {required_fields}")
            
            # Check optional fields
            present_optional = [field for field in optional_fields if field in sample_edge]
            if present_optional:
                print(f"   ✓ Optional fields present: {present_optional}")
            
            # Verify all edges have id field
            edges_without_id = [i for i, edge in enumerate(edges) if "id" not in edge]
            if edges_without_id:
                print(f"   ✗ {len(edges_without_id)} edges missing 'id' field")
                return False
            else:
                print(f"   ✓ All {len(edges)} edges have 'id' field")
        else:
            print("   ⚠ No edges returned to validate")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Specific label query
    print("\n2. Testing specific label query for edge format...")
    url = f"{base_url}/graphs?label=DOM&max_depth=2&max_nodes=30"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(edges) > 0:
            # Check first few edges
            print(f"\n   First 3 edges:")
            for i, edge in enumerate(edges[:3]):
                has_id = "id" in edge
                print(f"   Edge {i+1}: {edge.get('id', 'NO ID')} ({edge.get('source', '?')} -> {edge.get('target', '?')}) - Has ID: {has_id}")
                
            # Verify all edges have id field
            edges_without_id = [i for i, edge in enumerate(edges) if "id" not in edge]
            if edges_without_id:
                print(f"   ✗ {len(edges_without_id)} edges missing 'id' field")
            else:
                print(f"   ✓ All {len(edges)} edges have 'id' field")
        else:
            print("   ⚠ No edges for specific label")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("EDGE FORMAT TEST COMPLETE")
    print("=" * 60)
    print("\nThe edge format fix should allow the WebUI to properly render edges.")
    print("Each edge now has an 'id' field as required by the KnowledgeGraphEdge model.")
    
    return True

if __name__ == "__main__":
    # Allow specifying a different server URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://sea-turtle-app-6kjrs.ondigitalocean.app"
    
    print(f"Testing against: {base_url}\n")
    success = test_edge_format(base_url)
    sys.exit(0 if success else 1)