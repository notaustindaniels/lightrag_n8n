#!/usr/bin/env python3
"""
Test script to verify File Name and hover display
"""
import requests
import json
import time

BASE_URL = "http://localhost:9621"

def test_document_display():
    """Test document display fields"""
    print("=== Testing Document Display Fields ===\n")
    
    # Test document with full path
    test_doc = {
        "text": "Test content about Python decorators.",
        "description": "Test document",
        "source_url": "https://docs.python.org/3/library/functions/decorators/advanced-usage",
        "sitemap_url": "https://docs.python.org/sitemap.xml",
        "doc_index": 1,
        "total_docs": 1,
        "document_id": "[docs.python.org] 3/library/functions/decorators/advanced-usage"
    }
    
    print("Inserting test document...")
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Document inserted")
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
        print(f"  display_name: {result.get('display_name')}")
    else:
        print(f"✗ Failed: {response.status_code}")
        return
    
    # Wait for processing
    time.sleep(2)
    
    # Get documents
    print("\n=== Checking Document Display ===")
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        for doc in docs:
            source_url = doc.get('metadata', {}).get('source_url', '')
            if source_url == test_doc['source_url']:
                print(f"\nFound our test document:")
                print(f"  id field: {doc.get('id')}")
                print(f"  file_path: {doc.get('file_path')}")
                print(f"  display_name: {doc.get('display_name')}")
                print(f"  source_url: {source_url}")
                
                print(f"\nWebUI Display:")
                print(f"  ID shows: {source_url} (full URL - cool!)")
                print(f"  File Name shows: {doc.get('display_name')}")
                print(f"  Hover shows: {doc.get('file_path')}")
                
                # Check if display is correct
                expected_display = "[docs.python.org] advanced-usage"
                expected_hover = "[docs.python.org] 3/library/functions/decorators/advanced-usage"
                
                if doc.get('display_name') == expected_display:
                    print(f"  ✓ File Name correct: shows slug only")
                else:
                    print(f"  ✗ File Name incorrect: expected '{expected_display}'")
                
                if doc.get('file_path') == expected_hover:
                    print(f"  ✓ Hover correct: shows full path")
                else:
                    print(f"  ✗ Hover incorrect: expected '{expected_hover}'")
                
                break
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_document_display()