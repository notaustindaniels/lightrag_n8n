#!/usr/bin/env python3
"""
Test script to verify custom document IDs work correctly with LightRAG
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:9621"

def test_custom_id_insertion():
    """Test inserting documents with custom IDs"""
    print("=== Testing Custom ID Insertion ===\n")
    
    # Test 1: Document with n8n-style custom ID
    test_doc1 = {
        "text": "This is a test document about Python decorators and their advanced usage patterns.",
        "description": "Test with custom document_id",
        "source_url": "https://docs.python.org/3/library/functions/decorators/advanced-usage",
        "sitemap_url": "https://docs.python.org/sitemap.xml",
        "doc_index": 1,
        "total_docs": 2,
        "document_id": "[docs.python.org] 3/library/functions/decorators/advanced-usage"
    }
    
    print("Inserting document with custom ID...")
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc1)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  Custom ID should be: {test_doc1['document_id']}")
        print(f"  file_path: {result.get('file_path')}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return False
    
    # Test 2: Document without custom ID (should use hash)
    test_doc2 = {
        "text": "This is another test document about Python generators and yield expressions.",
        "description": "Test without custom ID",
        "source_url": "https://realpython.com/introduction-to-python-generators",
        "sitemap_url": "https://realpython.com/sitemap.xml",
        "doc_index": 2,
        "total_docs": 2
    }
    
    print("\nInserting document without custom ID...")
    response = requests.post(f"{BASE_URL}/documents/text/enhanced", json=test_doc2)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  doc_id: {result.get('doc_id')}")
        print(f"  file_path: {result.get('file_path')}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return False
    
    return True

def check_document_retrieval():
    """Check how documents are retrieved and displayed"""
    print("\n\n=== Checking Document Retrieval ===\n")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        print(f"Total documents found: {len(docs)}\n")
        
        custom_id_found = False
        hash_id_found = False
        
        for doc in docs:
            doc_id = doc.get('id')
            file_path = doc.get('file_path')
            display_name = doc.get('display_name')
            
            print(f"Document:")
            print(f"  ID (WebUI displays this): {doc_id}")
            print(f"  file_path: {file_path}")
            print(f"  display_name: {display_name}")
            
            # Check if this is displaying correctly
            if doc_id and doc_id.startswith('[') and ']' in doc_id:
                print(f"  ✓ Custom ID displayed correctly!")
                custom_id_found = True
            elif doc_id and doc_id.startswith('doc-'):
                print(f"  ✗ Still showing hash ID")
                hash_id_found = True
            
            # Check metadata
            metadata = doc.get('metadata', {})
            if metadata:
                print(f"  source_url: {metadata.get('source_url', 'N/A')}")
            
            print()
        
        # Summary
        print("\n=== Summary ===")
        if custom_id_found and not hash_id_found:
            print("✓ All documents with custom IDs are displaying correctly!")
            return True
        elif custom_id_found and hash_id_found:
            print("⚠ Mixed results: Some documents show custom IDs, others show hash IDs")
            return False
        else:
            print("✗ No custom IDs found in display")
            return False
    else:
        print(f"Error getting documents: {response.status_code}")
        print(response.text)
        return False

def test_hover_behavior():
    """Test what the hover behavior would show"""
    print("\n=== Testing Hover Behavior ===\n")
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        for doc in docs:
            doc_id = doc.get('id')
            file_path = doc.get('file_path')
            
            if doc_id and doc_id.startswith('[') and ']' in doc_id:
                # Parse the custom ID to simulate hover behavior
                parts = doc_id.split('] ', 1)
                if len(parts) == 2:
                    domain_part = parts[0] + ']'
                    path_part = parts[1]
                    
                    print(f"Document ID: {doc_id}")
                    print(f"  Display: {domain_part} {path_part.split('/')[-1] if '/' in path_part else path_part}")
                    print(f"  Hover: {doc_id}")
                    print(f"  ✓ Hover would show full path correctly\n")
                else:
                    print(f"Document ID: {doc_id}")
                    print(f"  ⚠ Unexpected format\n")
            elif doc_id and doc_id.startswith('doc-'):
                print(f"Document ID: {doc_id}")
                print(f"  Display: {doc_id}")
                print(f"  Hover: [unknown] {file_path}")
                print(f"  ✗ Hover would show [unknown]\n")
    
    return True

def main():
    """Run all tests"""
    print("Custom Document ID Test Suite\n")
    print("This test verifies that custom document IDs work correctly\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("✗ Server is not healthy")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running at", BASE_URL)
        print("Please start the server with: docker-compose up -d")
        sys.exit(1)
    
    # Run tests
    success = True
    
    if not test_custom_id_insertion():
        success = False
    
    # Wait a bit for processing
    time.sleep(2)
    
    if not check_document_retrieval():
        success = False
    
    test_hover_behavior()
    
    # Final result
    print("\n=== Test Results ===")
    if success:
        print("✓ All tests passed! Custom IDs are working correctly.")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())