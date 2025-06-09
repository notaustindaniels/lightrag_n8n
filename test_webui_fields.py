#!/usr/bin/env python3
"""
Test to see what fields the WebUI receives and how they're displayed
"""
import requests
import json

BASE_URL = "http://localhost:9621"

def test_webui_fields():
    """Check what fields are returned for documents"""
    print("=== Testing WebUI Field Display ===\n")
    
    # Get documents
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get('statuses', {}).get('processed', [])
        
        if docs:
            # Show first document in detail
            doc = docs[0]
            print("First document fields:")
            print(f"  id: {doc.get('id')}")
            print(f"  doc_id: {doc.get('doc_id')}")
            print(f"  file_path: {doc.get('file_path')}")
            print(f"  display_name: {doc.get('display_name')}")
            
            metadata = doc.get('metadata', {})
            print(f"\nMetadata fields:")
            print(f"  source_url: {metadata.get('source_url')}")
            print(f"  file_path (in metadata): {metadata.get('file_path')}")
            print(f"  display_name (in metadata): {metadata.get('display_name')}")
            print(f"  id (in metadata): {metadata.get('id')}")
            
            print(f"\n=== Analysis ===")
            print(f"If ID shows '[domain] <full-url>', the WebUI might be:")
            print(f"1. Using 'id' field: {doc.get('id')}")
            print(f"2. Constructing from metadata")
            print(f"3. Using some other field")
            
            print(f"\nIf File Name shows just slug, it might be using:")
            print(f"1. A field we're not seeing")
            print(f"2. Parsing one of these fields")
            print(f"3. Using display_name but parsing it")
        else:
            print("No documents found")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_webui_fields()