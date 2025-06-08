#!/usr/bin/env python3
"""
Test script to verify document ID display behavior
"""
import requests
import json
import time

BASE_URL = "http://localhost:9621"

def test_n8n_style_insert():
    """Test inserting a document the way n8n does it"""
    print("Testing n8n-style document insertion...")
    
    # Simulate what n8n sends
    test_doc = {
        "text": "This is a test document about advanced Python decorators and their usage.",
        "description": "Test document from n8n workflow",
        "source_url": "https://docs.python.org/3/library/functions/decorators/advanced-usage",
        "sitemap_url": "https://docs.python.org/sitemap.xml",
        "doc_index": 1,
        "total_docs": 1,
        "document_id": "[docs.python.org] 3/library/functions/decorators/advanced-usage"
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/text/enhanced",
        json=test_doc
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json().get("doc_id") if response.status_code == 200 else None

def test_get_documents():
    """Get all documents and check their display"""
    print("\nGetting all documents...")
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total documents: {data.get('total', 0)}")
        
        if data.get('statuses', {}).get('processed'):
            print("\nDocument details:")
            for doc in data['statuses']['processed']:
                print(f"\n  ID: {doc.get('id')}")
                print(f"  file_path: {doc.get('file_path')}")
                print(f"  display_name: {doc.get('display_name')}")
                
                metadata = doc.get('metadata', {})
                print(f"  source_url: {metadata.get('source_url')}")
                print(f"  full_path: {metadata.get('full_path', 'N/A')}")
                
                # This is what the WebUI sees
                print(f"  >>> WebUI displays: {doc.get('file_path')}")
    else:
        print(f"Error: {response.text}")

def test_without_document_id():
    """Test inserting without document_id to see the difference"""
    print("\n\nTesting without document_id (API generates it)...")
    
    test_doc = {
        "text": "Another test document about Python generators.",
        "description": "Test without document_id",
        "source_url": "https://realpython.com/introduction-to-python-generators",
        "sitemap_url": "https://realpython.com/sitemap.xml",
        "doc_index": 2,
        "total_docs": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/text/enhanced",
        json=test_doc
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("=== Document Display Test ===\n")
    
    # Test with document_id (n8n style)
    doc_id = test_n8n_style_insert()
    
    # Wait a bit
    time.sleep(1)
    
    # Test without document_id
    test_without_document_id()
    
    # Wait a bit
    time.sleep(1)
    
    # Get all documents to see how they're displayed
    test_get_documents()
    
    print("\n=== Test Complete ===")