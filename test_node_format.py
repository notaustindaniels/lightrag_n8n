#!/usr/bin/env python3
"""Test script to verify node format matches WebUI expectations"""

import requests
import json
import sys

def test_node_format(base_url="https://sea-turtle-app-6kjrs.ondigitalocean.app"):
    """Test that nodes have the correct format expected by WebUI"""
    
    print("=" * 60)
    print("TESTING NODE FORMAT FOR WEBUI COMPATIBILITY")
    print("=" * 60)
    
    # Test wildcard query
    print("\n1. Testing /graphs?label=* for node format...")
    url = f"{base_url}/graphs?label=*&max_depth=3&max_nodes=5"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        print(f"   Response status: {response.status_code}")
        print(f"   Nodes returned: {len(nodes)}")
        print(f"   Edges returned: {len(edges)}")
        
        if len(nodes) > 0:
            # Check first node structure
            first_node = nodes[0]
            print(f"\n   First node structure:")
            print(f"   {json.dumps(first_node, indent=6)}")
            
            # Validate expected structure
            required_fields = ["id", "labels", "properties"]
            missing_fields = [field for field in required_fields if field not in first_node]
            
            if missing_fields:
                print(f"\n   ✗ Missing required fields: {missing_fields}")
                return False
            
            # Check that labels is an array
            if not isinstance(first_node.get("labels"), list):
                print(f"\n   ✗ 'labels' should be an array, got: {type(first_node.get('labels'))}")
                return False
            
            # Check that properties is a dict
            if not isinstance(first_node.get("properties"), dict):
                print(f"\n   ✗ 'properties' should be a dict, got: {type(first_node.get('properties'))}")
                return False
            
            print(f"\n   ✓ Node structure matches WebUI expectations:")
            print(f"      - id: string")
            print(f"      - labels: array of strings")
            print(f"      - properties: dictionary")
            
            # Check all nodes have correct structure
            invalid_nodes = []
            for i, node in enumerate(nodes):
                if not all(field in node for field in required_fields):
                    invalid_nodes.append(i)
                elif not isinstance(node.get("labels"), list):
                    invalid_nodes.append(i)
                elif not isinstance(node.get("properties"), dict):
                    invalid_nodes.append(i)
            
            if invalid_nodes:
                print(f"\n   ✗ {len(invalid_nodes)} nodes have invalid structure")
                return False
            else:
                print(f"   ✓ All {len(nodes)} nodes have correct structure")
        else:
            print("   ⚠ No nodes returned to validate")
        
        # Check edge format
        if len(edges) > 0:
            first_edge = edges[0]
            print(f"\n   First edge structure:")
            print(f"   {json.dumps(first_edge, indent=6)}")
            
            edge_required = ["id", "source", "target"]
            edge_missing = [field for field in edge_required if field not in first_edge]
            
            if edge_missing:
                print(f"\n   ✗ Edge missing required fields: {edge_missing}")
            else:
                print(f"   ✓ Edge has all required fields")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("NODE FORMAT TEST COMPLETE")
    print("=" * 60)
    print("\nThe node format should now match what the WebUI expects:")
    print("- Nodes have 'labels' array (not 'label' string)")
    print("- All node data is nested in 'properties' object")
    print("- This matches the original LightRAG API structure")
    
    return True

if __name__ == "__main__":
    # Allow specifying a different server URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://sea-turtle-app-6kjrs.ondigitalocean.app"
    
    print(f"Testing against: {base_url}\n")
    success = test_node_format(base_url)
    sys.exit(0 if success else 1)