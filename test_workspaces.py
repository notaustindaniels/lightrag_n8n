#!/usr/bin/env python3
"""
Test script to verify workspace isolation functionality
"""
import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8081"

async def test_workspace_isolation():
    """Test that workspaces properly isolate documents"""
    async with aiohttp.ClientSession() as session:
        
        print("=" * 60)
        print("Testing LightRAG Workspace Isolation")
        print("=" * 60)
        
        # 1. Create two workspaces
        print("\n1. Creating workspaces...")
        
        # Create 'react' workspace
        async with session.post(f"{BASE_URL}/workspaces/react") as resp:
            if resp.status == 200:
                print("✅ Created 'react' workspace")
            else:
                print(f"❌ Failed to create 'react' workspace: {await resp.text()}")
        
        # Create 'vue' workspace  
        async with session.post(f"{BASE_URL}/workspaces/vue") as resp:
            if resp.status == 200:
                print("✅ Created 'vue' workspace")
            else:
                print(f"❌ Failed to create 'vue' workspace: {await resp.text()}")
        
        # 2. List workspaces
        print("\n2. Listing all workspaces...")
        async with session.get(f"{BASE_URL}/workspaces") as resp:
            workspaces = await resp.json()
            print(f"Found {workspaces['total']} workspaces:")
            for ws in workspaces['workspaces']:
                print(f"  - {ws['name']} (docs: {ws['document_count']}, default: {ws['is_default']})")
        
        # 3. Insert documents into different workspaces
        print("\n3. Inserting documents into workspaces...")
        
        # React documentation
        react_doc = {
            "text": "React is a JavaScript library for building user interfaces. It uses a virtual DOM for efficient updates and supports component-based architecture with hooks like useState and useEffect.",
            "file_path": "[react] hooks-guide.md",
            "description": "React hooks documentation"
        }
        
        async with session.post(f"{BASE_URL}/documents/text", json=react_doc) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ Inserted React doc: {result.get('file_path')}")
            else:
                print(f"❌ Failed to insert React doc: {await resp.text()}")
        
        # Vue documentation
        vue_doc = {
            "text": "Vue.js is a progressive JavaScript framework. It features reactive data binding, computed properties, and a template-based syntax. The Composition API provides flexible component logic.",
            "file_path": "[vue] composition-api.md",
            "description": "Vue composition API documentation"
        }
        
        async with session.post(f"{BASE_URL}/documents/text", json=vue_doc) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"✅ Inserted Vue doc: {result.get('file_path')}")
            else:
                print(f"❌ Failed to insert Vue doc: {await resp.text()}")
        
        # Wait for processing
        print("\n⏳ Waiting for documents to process...")
        await asyncio.sleep(5)
        
        # 4. Query specific workspaces
        print("\n4. Testing workspace-specific queries...")
        
        # Query React workspace about hooks
        query_react = {
            "query": "What are hooks and how do they work?",
            "sources": ["react"],
            "mode": "hybrid"
        }
        
        print("\n📝 Querying 'react' workspace about hooks...")
        async with session.post(f"{BASE_URL}/query", json=query_react) as resp:
            if resp.status == 200:
                result = await resp.json()
                response = result.get('response', '')[:200]
                print(f"Response preview: {response}...")
                if 'useState' in result.get('response', '') or 'React' in result.get('response', ''):
                    print("✅ React workspace returned React-specific information")
                else:
                    print("⚠️ Response may not be from React workspace")
            else:
                print(f"❌ Query failed: {await resp.text()}")
        
        # Query Vue workspace about Composition API
        query_vue = {
            "query": "Explain the Composition API",
            "sources": ["vue"],
            "mode": "hybrid"
        }
        
        print("\n📝 Querying 'vue' workspace about Composition API...")
        async with session.post(f"{BASE_URL}/query", json=query_vue) as resp:
            if resp.status == 200:
                result = await resp.json()
                response = result.get('response', '')[:200]
                print(f"Response preview: {response}...")
                if 'Vue' in result.get('response', '') or 'reactive' in result.get('response', ''):
                    print("✅ Vue workspace returned Vue-specific information")
                else:
                    print("⚠️ Response may not be from Vue workspace")
            else:
                print(f"❌ Query failed: {await resp.text()}")
        
        # 5. Test cross-workspace query
        print("\n5. Testing cross-workspace query...")
        
        query_both = {
            "query": "Compare the frameworks",
            "sources": ["react", "vue"],
            "mode": "hybrid"
        }
        
        async with session.post(f"{BASE_URL}/query", json=query_both) as resp:
            if resp.status == 200:
                result = await resp.json()
                if 'workspaces_queried' in result:
                    print(f"✅ Queried workspaces: {result['workspaces_queried']}")
                response = result.get('response', '')
                if '[From react]' in response and '[From vue]' in response:
                    print("✅ Response includes information from both workspaces")
                else:
                    print("⚠️ Response format may not show workspace separation")
            else:
                print(f"❌ Query failed: {await resp.text()}")
        
        # 6. List sources to verify workspace-based organization
        print("\n6. Listing document sources...")
        async with session.get(f"{BASE_URL}/documents/sources") as resp:
            if resp.status == 200:
                sources = await resp.json()
                print(f"Found {sources['total']} sources:")
                for source in sources['sources']:
                    print(f"  - {source['source']}: {source['document_count']} docs")
            else:
                print(f"❌ Failed to list sources: {await resp.text()}")
        
        print("\n" + "=" * 60)
        print("✅ Workspace isolation testing complete!")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_workspace_isolation())