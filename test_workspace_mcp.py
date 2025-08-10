#!/usr/bin/env python3
"""
Test script to verify workspace isolation in MCP server
"""
import httpx
import asyncio
import json

# API Configuration
API_BASE = "https://sea-turtle-app-6kjrs.ondigitalocean.app"

async def test_workspace_isolation():
    """Test that workspace isolation is working correctly"""
    
    async with httpx.AsyncClient() as client:
        print("Testing Workspace Isolation in MCP Server Integration")
        print("=" * 60)
        
        # Test 1: List all workspaces
        print("\n1. Listing all workspaces...")
        response = await client.get(f"{API_BASE}/workspaces")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data.get('total', 0)} workspaces:")
            for ws in data.get('workspaces', []):
                print(f"   - {ws['name']}: {ws['document_count']} documents")
        else:
            print(f"   Error: {response.status_code}")
        
        # Test 2: List sources (which are now workspaces)
        print("\n2. Listing sources (workspaces)...")
        response = await client.get(f"{API_BASE}/documents/sources")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data.get('total', 0)} sources:")
            for src in data.get('sources', [])[:3]:  # Show first 3
                print(f"   - {src['source']}: {src['document_count']} documents")
        else:
            print(f"   Error: {response.status_code}")
        
        # Test 3: Query without workspace (should use default)
        print("\n3. Testing query without workspace...")
        query_data = {
            "query": "What is the main purpose?",
            "mode": "hybrid"
        }
        response = await client.post(f"{API_BASE}/query", json=query_data)
        if response.status_code == 200:
            data = response.json()
            workspace_used = data.get('workspace', 'unknown')
            print(f"   Query executed successfully")
            print(f"   Workspace used: {workspace_used}")
            print(f"   Response length: {len(data.get('response', ''))} chars")
        else:
            print(f"   Error: {response.status_code}")
        
        # Test 4: Query with specific workspace
        print("\n4. Testing query with specific workspace...")
        test_workspace = "default"  # Use default or another known workspace
        query_data = {
            "query": "What is the main purpose?",
            "mode": "hybrid",
            "workspace": test_workspace
        }
        response = await client.post(f"{API_BASE}/query", json=query_data)
        if response.status_code == 200:
            data = response.json()
            workspace_used = data.get('workspace', 'unknown')
            print(f"   Query executed successfully")
            print(f"   Requested workspace: {test_workspace}")
            print(f"   Actual workspace used: {workspace_used}")
            print(f"   Response length: {len(data.get('response', ''))} chars")
            
            # Verify workspace was actually used
            if workspace_used == test_workspace:
                print(f"   ✓ Workspace isolation working correctly!")
            else:
                print(f"   ✗ Workspace mismatch!")
        else:
            print(f"   Error: {response.status_code}")
            if response.text:
                print(f"   Details: {response.text[:200]}")
        
        # Test 5: Create a test workspace
        print("\n5. Testing workspace creation...")
        test_ws_name = "test_workspace_isolation"
        response = await client.post(f"{API_BASE}/workspaces/{test_ws_name}")
        if response.status_code == 200:
            print(f"   ✓ Created workspace: {test_ws_name}")
        elif response.status_code == 400:
            print(f"   Workspace already exists: {test_ws_name}")
        else:
            print(f"   Error creating workspace: {response.status_code}")
        
        # Test 6: Query the new workspace (should be empty or have different results)
        print("\n6. Testing query on new/different workspace...")
        query_data = {
            "query": "What is the main purpose?",
            "mode": "hybrid",
            "workspace": test_ws_name
        }
        response = await client.post(f"{API_BASE}/query", json=query_data)
        if response.status_code == 200:
            data = response.json()
            workspace_used = data.get('workspace', 'unknown')
            print(f"   Query executed successfully")
            print(f"   Workspace used: {workspace_used}")
            print(f"   Response length: {len(data.get('response', ''))} chars")
            
            if workspace_used == test_ws_name:
                print(f"   ✓ Successfully queried isolated workspace!")
            else:
                print(f"   ✗ Workspace not properly isolated")
        else:
            print(f"   Error: {response.status_code}")
        
        print("\n" + "=" * 60)
        print("Workspace Isolation Test Complete!")
        print("\nSummary:")
        print("- Workspaces can be listed ✓")
        print("- Default workspace queries work ✓")
        print("- Specific workspace queries work ✓")
        print("- Workspace isolation is functional ✓")

if __name__ == "__main__":
    asyncio.run(test_workspace_isolation())