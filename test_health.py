#!/usr/bin/env python3
"""
Test script to verify health check endpoint
"""
import requests
import time
import sys

def test_health_endpoint(host="localhost", port=9621, max_attempts=30):
    """Test the health endpoint"""
    url = f"http://{host}:{port}/health"
    print(f"Testing health endpoint: {url}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=10)
            print(f"Attempt {attempt + 1}: Status {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("✅ Health check passed!")
                return True
            elif response.status_code == 503:
                print("⏳ Service still starting...")
            else:
                print(f"❌ Unexpected status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt + 1}: Connection refused")
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1}: Request timeout")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
        
        if attempt < max_attempts - 1:
            time.sleep(5)
    
    print("❌ Health check failed after all attempts")
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test health endpoint")
    parser.add_argument("--host", default="localhost", help="Host to test")
    parser.add_argument("--port", type=int, default=9621, help="Port to test")
    parser.add_argument("--attempts", type=int, default=30, help="Max attempts")
    
    args = parser.parse_args()
    
    success = test_health_endpoint(args.host, args.port, args.attempts)
    sys.exit(0 if success else 1) 