#!/usr/bin/env python3
"""
Test script to verify WebUI display behavior is fixed
"""
import requests
import json
import time

BASE_URL = "http://localhost:9621"

def insert_test_documents():
    """Insert test documents with different ID formats"""
    print("=== Inserting Test Documents ===\n")
    
    # Test 1: Document with n8n-style document_id
    test_doc1 = {
        "text": "This is a test document about Python decorators.",
        "description": "Test with n8n document_id",
        "source_url": "https://docs.python.org/3/library/functions/decorators/advanced-usage",
        "sitemap_url": "https://docs.python.org/sitemap.xml",
        "doc_index": 1,
        "total_docs": 3,
        "document_id": "[docs.python.org] 3/library/functions/decorators/advanced-usage"
    }
    
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc1)
    print(f"Document 1 - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
    
    # Test 2: Document without document_id (API generates path)
    test_doc2 = {
        "text": "This is a test document about Python generators.",
        "description": "Test without document_id",
        "source_url": "https://realpython.com/introduction-to-python-generators",
        "sitemap_url": "https://realpython.com/sitemap.xml",
        "doc_index": 2,
        "total_docs": 3
    }
    
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc2)
    print(f"\nDocument 2 - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
    
    # Test 3: Document with root domain
    test_doc3 = {
        "text": "This is the homepage content.",
        "description": "Test with root domain",
        "source_url": "https://example.com/",
        "sitemap_url": "https://example.com/sitemap.xml",
        "doc_index": 3,
        "total_docs": 3,
        "document_id": "[example.com] index"
    }
    
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc3)
    print(f"\nDocument 3 - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")

def check_document_display():
    """Check how documents are displayed"""
    print("\n\n=== Checking Document Display ===\n")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        print(f"Total documents: {len(docs)}\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"  ID (WebUI displays this): {doc.get('id')}")
            print(f"  doc_id (internal hash): {doc.get('doc_id', 'N/A')}")
            print(f"  file_path: {doc.get('file_path')}")
            print(f"  display_name: {doc.get('display_name')}")
            
            # Check what the WebUI would show
            display_id = doc.get('id')
            if display_id.startswith('doc-'):
                print(f"  ❌ WebUI shows hash ID: {display_id}")
            elif display_id.startswith('[') and ']' in display_id:
                print(f"  ✅ WebUI shows formatted ID: {display_id}")
            else:
                print(f"  ⚠️  WebUI shows: {display_id}")
            
            print()
    else:
        print(f"Error getting documents: {response.status_code}")
        print(response.text)

def test_by_sitemap():
    """Test the by-sitemap endpoint"""
    print("\n=== Testing By-Sitemap Endpoint ===\n")
    
    sitemap_url = "https://docs.python.org/sitemap.xml"
    response = requests.get(f"{BASE_URL}/documents/by-sitemap/{sitemap_url}")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('documents', [])
        
        print(f"Documents for sitemap {sitemap_url}: {len(docs)}\n")
        
        for doc in docs:
            print(f"- ID: {doc.get('id')}")
            print(f"  doc_id: {doc.get('doc_id', 'N/A')}")
            print(f"  source_url: {doc.get('source_url')}")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    print("WebUI Display Test\n")
    print("This test verifies that the WebUI displays the correct document IDs\n")
    
    # Insert test documents
    insert_test_documents()
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Check how they're displayed
    check_document_display()
    
    # Test by-sitemap
    test_by_sitemap()
    
    print("\n=== Test Complete ===")
    print("\nExpected behavior:")
    print("- Documents with document_id from n8n should display as '[domain] path'")
    print("- Documents without document_id should display as '[domain] path' (generated)")
    print("- The WebUI should show these formatted IDs, not the hash IDs")