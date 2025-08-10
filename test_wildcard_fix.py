#!/usr/bin/env python3
"""Test script to verify wildcard (*) handling in /graphs endpoint"""

import requests
import json
import sys

def test_wildcard_endpoint(base_url="https://sea-turtle-app-6kjrs.ondigitalocean.app"):
    """Test that the wildcard label returns graph data"""
    
    print("=" * 60)
    print("TESTING WILDCARD (*) LABEL HANDLING")
    print("=" * 60)
    
    # Test 1: Wildcard query
    print("\n1. Testing /graphs?label=* ...")
    url = f"{base_url}/graphs?label=*&max_depth=3&max_nodes=100"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Nodes returned: {len(nodes)}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(nodes) > 0:
            print("   ✓ Wildcard query returns nodes!")
            sample_nodes = [node.get("label", node.get("id", "unknown")) for node in nodes[:3]]
            print(f"   Sample nodes: {sample_nodes}")
        else:
            print("   ✗ No nodes returned for wildcard query")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Specific label query (should still work)
    print("\n2. Testing specific label query...")
    url = f"{base_url}/graphs?label=Paged.js%20Library&max_depth=2&max_nodes=50"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Nodes returned: {len(nodes)}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(nodes) > 0:
            print("   ✓ Specific label query still works!")
        else:
            print("   ⚠ Warning: No nodes for specific label (might not exist)")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Empty string query
    print("\n3. Testing empty label query...")
    url = f"{base_url}/graphs?label=&max_depth=3&max_nodes=100"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Nodes returned: {len(nodes)}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(nodes) > 0:
            print("   ✓ Empty label query returns nodes")
        else:
            print("   ⚠ Empty label returns no nodes (might be expected)")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nThe wildcard fix should allow the WebUI to display nodes.")
    print("Navigate to the Knowledge Graph tab to verify.")
    
    return True

if __name__ == "__main__":
    # Allow specifying a different server URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://sea-turtle-app-6kjrs.ondigitalocean.app"
    
    print(f"Testing against: {base_url}\n")
    success = test_wildcard_endpoint(base_url)
    sys.exit(0 if success else 1)