#!/usr/bin/env python3
"""
Test script for Enhanced LightRAG API
"""
import requests
import json
import time
from urllib.parse import quote

BASE_URL = "http://localhost:9621"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_enhanced_insert():
    """Test enhanced document insertion"""
    print("Testing enhanced document insertion...")
    
    test_doc = {
        "text": "This is a test document about Python programming. Python is a high-level programming language.",
        "description": "Test document for LightRAG",
        "source_url": "https://example.com/python-tutorial",
        "sitemap_url": "https://example.com/sitemap.xml",
        "doc_index": 1,
        "total_docs": 1
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/text/enhanced",
        json=test_doc
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        return response.json().get("doc_id")
    return None

def test_get_documents():
    """Test getting all documents"""
    print("\nTesting document retrieval...")
    response = requests.get(f"{BASE_URL}/documents")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total documents: {data.get('total', 0)}")
        
        # Show first document if available
        if data.get('statuses', {}).get('processed'):
            first_doc = data['statuses']['processed'][0]
            print(f"\nFirst document:")
            print(f"  ID: {first_doc.get('id')}")
            print(f"  File Path: {first_doc.get('file_path')}")
            print(f"  Source URL: {first_doc.get('metadata', {}).get('source_url')}")
    else:
        print(f"Error: {response.text}")
    print()

def test_get_by_sitemap():
    """Test getting documents by sitemap"""
    print("Testing get documents by sitemap...")
    sitemap_url = "https://example.com/sitemap.xml"
    
    response = requests.get(
        f"{BASE_URL}/documents/by-sitemap/{quote(sitemap_url, safe='')}"
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_query():
    """Test querying the RAG system"""
    print("Testing RAG query...")
    
    query_data = {
        "query": "What is Python?",
        "mode": "hybrid"
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=query_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Query: {result.get('query')}")
        print(f"Response: {result.get('response')[:200]}...")
    else:
        print(f"Error: {response.text}")
    print()

def test_delete_by_sitemap():
    """Test deleting documents by sitemap"""
    print("Testing delete by sitemap...")
    sitemap_url = "https://example.com/sitemap.xml"
    
    response = requests.delete(
        f"{BASE_URL}/documents/by-sitemap/{quote(sitemap_url, safe='')}"
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_debug_storage():
    """Test debug storage endpoint"""
    print("Testing debug storage endpoint...")
    response = requests.get(f"{BASE_URL}/debug/storage")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Debug info:")
        print(f"  Metadata store count: {data.get('metadata_store_count', 0)}")
        print(f"  LightRAG storage: {data.get('lightrag_storage', 'Unknown')}")
        print(f"  RAG instance type: {data.get('rag_instance_type', 'Unknown')}")
    print()

def test_status_endpoints():
    """Test various status endpoints"""
    print("Testing status endpoints...")
    
    # Test /status
    print("  Testing /status...")
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        print(f"    System status: {response.json().get('status')}")
        print(f"    Document count: {response.json().get('documents', {}).get('total', 0)}")
    
    # Test /documents/status
    print("  Testing /documents/status...")
    response = requests.get(f"{BASE_URL}/documents/status")
    if response.status_code == 200:
        print(f"    Total documents: {response.json().get('total', 0)}")
    print()

def test_webui():
    """Test WebUI availability"""
    print("Testing WebUI endpoint...")
    
    # Test root redirect
    print("  Testing root redirect...")
    response = requests.get(f"{BASE_URL}/", allow_redirects=False)
    if response.status_code == 307:
        print(f"  Root redirects to: {response.headers.get('location')}")
    
    # Test webui endpoint
    print("  Testing /webui endpoint...")
    response = requests.get(f"{BASE_URL}/webui")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print("  WebUI is available!")
        print(f"  Content length: {len(response.content)} bytes")
        # Check if it's the full UI or fallback
        if b"LightRAG Extended API" in response.content:
            print("  Using fallback WebUI")
        else:
            print("  Full LightRAG WebUI detected")
    else:
        print("  WebUI not available")
    print()

if __name__ == "__main__":
    print("Enhanced LightRAG API Test Suite")
    print("=" * 50)
    
    # Run tests
    test_health()
    
    # Test status endpoints
    test_status_endpoints()
    
    # Test WebUI
    test_webui()
    
    # Insert a document
    doc_id = test_enhanced_insert()
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Get documents
    test_get_documents()
    
    # Test debug endpoint
    test_debug_storage()
    
    # Get by sitemap
    test_get_by_sitemap()
    
    # Query
    test_query()
    
    # Clean up - delete by sitemap
    test_delete_by_sitemap()
    
    print("\nTest suite completed!")