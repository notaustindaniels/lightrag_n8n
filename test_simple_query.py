#!/usr/bin/env python3
"""
Simple test to check if LightRAG can retrieve any data
"""
import requests
import json

# Test the API endpoint directly
BASE_URL = "http://localhost:8020"

def test_query_api():
    """Test the query API with the test workspace"""
    
    print("Testing Query API")
    print("="*60)
    
    # Query parameters
    query_data = {
        "query": "What is LightRAG?",
        "mode": "hybrid",
        "workspace": "test"
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result.get('status')}")
        print(f"Mode: {result.get('mode')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"\nResponse:\n{'-'*40}")
        print(result.get('response', 'No response'))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    # Also test different modes
    print("\n" + "="*60)
    print("Testing Different Modes")
    print("="*60)
    
    modes = ["local", "global", "hybrid", "naive"]
    for mode in modes:
        print(f"\n### {mode.upper()} Mode ###")
        query_data["mode"] = mode
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'No response')
            # Show first 200 chars
            preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(preview)
        else:
            print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_query_api()