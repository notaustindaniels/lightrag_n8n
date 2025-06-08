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

def test_webui():
    """Test WebUI availability"""
    print("Testing WebUI endpoint...")
    response = requests.get(f"{BASE_URL}/webui/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("WebUI is available!")
    else:
        print("WebUI not available")
    
    # Test root redirect
    response = requests.get(f"{BASE_URL}/", allow_redirects=False)
    if response.status_code in [301, 302, 307, 308]:
        print(f"Root redirects to: {response.headers.get('Location')}")
    print()

def test_enhanced_insert():
    """Test enhanced document insertion"""
    print("Testing enhanced document insertion...")
    
    # Test with simple URL
    test_doc1 = {
        "text": "This is a test document about Python programming. Python is a high-level programming language.",
        "description": "Test document for LightRAG",
        "source_url": "https://example.com/python-tutorial",
        "sitemap_url": "https://example.com/sitemap.xml",
        "doc_index": 1,
        "total_docs": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/text/enhanced",
        json=test_doc1
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    doc_id1 = response.json().get("doc_id") if response.status_code == 200 else None
    
    # Test with deep path URL
    test_doc2 = {
        "text": "This is another test document about advanced Python features like decorators and generators.",
        "description": "Advanced Python document",
        "source_url": "https://docs.python.org/3/library/functions/decorators/advanced-usage",
        "sitemap_url": "https://example.com/sitemap.xml",
        "doc_index": 2,
        "total_docs": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/text/enhanced",
        json=test_doc2
    )
    
    print(f"\nDeep path document:")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return doc_id1

def test_get_documents():
    """Test getting all documents"""
    print("\nTesting document retrieval...")
    response = requests.get(f"{BASE_URL}/documents")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total documents: {data.get('total', 0)}")
        
        # Show all documents if available
        if data.get('statuses', {}).get('processed'):
            print(f"\nDocuments:")
            for i, doc in enumerate(data['statuses']['processed'][:5], 1):  # Show first 5
                print(f"\n  Document {i}:")
                print(f"    ID: {doc.get('id')}")
                print(f"    Display Name: {doc.get('display_name')}")
                print(f"    File Path: {doc.get('file_path')}")
                print(f"    Source URL: {doc.get('metadata', {}).get('source_url')}")
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

if __name__ == "__main__":
    print("Enhanced LightRAG API Test Suite")
    print("=" * 50)
    
    # Run tests
    test_health()
    test_webui()
    
    # Insert a document
    doc_id = test_enhanced_insert()
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Get documents
    test_get_documents()
    
    # Get by sitemap
    test_get_by_sitemap()
    
    # Query
    test_query()
    
    # Clean up - delete by sitemap
    test_delete_by_sitemap()
    
    print("\nTest suite completed!")