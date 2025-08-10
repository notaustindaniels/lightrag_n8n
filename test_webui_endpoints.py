#!/usr/bin/env python3
"""Test script to verify all WebUI required endpoints are working"""

import requests
import json
import sys

def test_endpoints(base_url="https://sea-turtle-app-6kjrs.ondigitalocean.app"):
    """Test all endpoints required by the WebUI"""
    
    print("=" * 60)
    print("TESTING WEBUI REQUIRED ENDPOINTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: GET /health
    print("\n1. Testing GET /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        results["GET /health"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["GET /health"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 2: GET /documents
    print("\n2. Testing GET /documents...")
    try:
        response = requests.get(f"{base_url}/documents", timeout=10)
        results["GET /documents"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["GET /documents"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 3: POST /documents/upload (with test file)
    print("\n3. Testing POST /documents/upload...")
    try:
        # Create a test text file
        test_content = "Test document content for WebUI endpoint testing"
        files = {'file': ('test.txt', test_content, 'text/plain')}
        response = requests.post(f"{base_url}/documents/upload", files=files, timeout=10)
        results["POST /documents/upload"] = {
            "status_code": response.status_code,
            "success": response.status_code in [200, 201]
        }
        if response.status_code == 500:
            print(f"   ✗ Status: {response.status_code} - {response.text}")
        else:
            print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["POST /documents/upload"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 4: POST /query
    print("\n4. Testing POST /query...")
    try:
        query_data = {
            "query": "What is LightRAG?",
            "mode": "hybrid",
            "stream": False
        }
        response = requests.post(
            f"{base_url}/query", 
            json=query_data,
            timeout=10
        )
        results["POST /query"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        if response.status_code == 500:
            print(f"   ✗ Status: {response.status_code} - {response.text}")
        else:
            print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["POST /query"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 5: POST /query/stream
    print("\n5. Testing POST /query/stream...")
    try:
        query_data = {
            "query": "What is LightRAG?",
            "mode": "hybrid",
            "stream": True
        }
        response = requests.post(
            f"{base_url}/query/stream", 
            json=query_data,
            timeout=10,
            stream=True
        )
        results["POST /query/stream"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        if response.status_code == 500:
            print(f"   ✗ Status: {response.status_code}")
        else:
            print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["POST /query/stream"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 6: DELETE /documents
    print("\n6. Testing DELETE /documents...")
    try:
        response = requests.delete(f"{base_url}/documents", timeout=10)
        results["DELETE /documents"] = {
            "status_code": response.status_code,
            "success": response.status_code in [200, 204]
        }
        if response.status_code == 405:
            print(f"   ✗ Status: {response.status_code} - Method Not Allowed")
        elif response.status_code == 500:
            print(f"   ✗ Status: {response.status_code} - {response.text}")
        else:
            print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["DELETE /documents"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 7: POST /documents/scan
    print("\n7. Testing POST /documents/scan...")
    try:
        response = requests.post(f"{base_url}/documents/scan", timeout=10)
        results["POST /documents/scan"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        if response.status_code == 500:
            print(f"   ✗ Status: {response.status_code} - {response.text}")
        else:
            print(f"   ✓ Status: {response.status_code}")
    except Exception as e:
        results["POST /documents/scan"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Test 8: GET /graphs with wildcard
    print("\n8. Testing GET /graphs?label=*...")
    try:
        response = requests.get(f"{base_url}/graphs?label=*&max_depth=3&max_nodes=10", timeout=10)
        data = response.json()
        has_correct_format = (
            "nodes" in data and 
            "edges" in data and
            (len(data["edges"]) == 0 or "id" in data["edges"][0])
        )
        results["GET /graphs"] = {
            "status_code": response.status_code,
            "success": response.status_code == 200 and has_correct_format,
            "has_edge_ids": len(data["edges"]) == 0 or "id" in data["edges"][0]
        }
        print(f"   ✓ Status: {response.status_code}, Edge IDs: {'Yes' if has_correct_format else 'No'}")
    except Exception as e:
        results["GET /graphs"] = {"error": str(e), "success": False}
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working_endpoints = sum(1 for r in results.values() if r.get("success", False))
    total_endpoints = len(results)
    
    print(f"\nWorking endpoints: {working_endpoints}/{total_endpoints}")
    
    for endpoint, result in results.items():
        status = "✓" if result.get("success", False) else "✗"
        print(f"  {status} {endpoint}")
    
    if working_endpoints == total_endpoints:
        print("\n🎉 All WebUI required endpoints are working!")
    else:
        print(f"\n⚠️  {total_endpoints - working_endpoints} endpoints need attention")
    
    return working_endpoints == total_endpoints

if __name__ == "__main__":
    # Allow specifying a different server URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://sea-turtle-app-6kjrs.ondigitalocean.app"
    
    print(f"Testing against: {base_url}\n")
    success = test_endpoints(base_url)
    sys.exit(0 if success else 1)