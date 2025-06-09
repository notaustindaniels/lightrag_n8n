#!/usr/bin/env python3
"""
Test script to verify [unknown] URL handling
"""
import requests
import json
import time

BASE_URL = "http://localhost:9621"

def test_unknown_url_handling():
    """Test how the API handles [unknown] URL format from n8n"""
    print("=== Testing [unknown] URL Handling ===\n")
    
    # Test 1: Document with [unknown] prefix (simulating n8n failure)
    test_doc1 = {
        "text": "Test content about Python generators.",
        "description": "Test document with [unknown] URL",
        "source_url": "https://realpython.com/introduction-to-python-generators",
        "sitemap_url": "https://realpython.com/sitemap.xml",
        "doc_index": 1,
        "total_docs": 2,
        "document_id": "[unknown] https://realpython.com/introduction-to-python-generators"
    }
    
    print("Test 1: Inserting document with [unknown] prefix...")
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc1)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Document inserted")
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
        print(f"  display_name: {result.get('display_name')}")
    else:
        print(f"✗ Failed: {response.status_code}")
    
    # Test 2: Document with proper format
    test_doc2 = {
        "text": "Test content about Python decorators.",
        "description": "Test document with proper format",
        "source_url": "https://docs.python.org/3/library/functions/decorators",
        "sitemap_url": "https://docs.python.org/sitemap.xml",
        "doc_index": 2,
        "total_docs": 2,
        "document_id": "[docs.python.org] 3/library/functions/decorators"
    }
    
    print("\nTest 2: Inserting document with proper format...")
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc2)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Document inserted")
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
        print(f"  display_name: {result.get('display_name')}")
    else:
        print(f"✗ Failed: {response.status_code}")
    
    # Wait for processing
    time.sleep(2)
    
    # Get documents and check display
    print("\n=== Checking Document Display ===")
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        print(f"\nTotal documents: {len(docs)}\n")
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            source_url = metadata.get('source_url', '')
            
            # Check our test documents
            if source_url in [test_doc1['source_url'], test_doc2['source_url']]:
                print(f"Document from: {source_url}")
                print(f"  id: {doc.get('id')}")
                print(f"  file_path: {doc.get('file_path')}")
                print(f"  display_name: {doc.get('display_name')}")
                
                # Check for [unknown]
                if '[unknown]' in str(doc.get('id', '')):
                    print(f"  ✗ ID still contains [unknown]")
                else:
                    print(f"  ✓ No [unknown] in ID")
                
                # Check display consistency
                if doc.get('file_path', '').startswith('[') and doc.get('display_name', '').startswith('['):
                    print(f"  ✓ Consistent format")
                else:
                    print(f"  ✗ Inconsistent format")
                
                print()
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_unknown_url_handling()