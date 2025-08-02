#!/usr/bin/env python3
"""
Debug script to help troubleshoot deployment issues
"""
import os
import sys
import time
import signal
import subprocess

# Optional dependency - install with: pip install psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: psutil not available. Install with 'pip install psutil' for detailed system info.")

def check_environment():
    """Check environment variables and system info"""
    print("=== Environment Check ===")
    env_vars = [
        "WORKING_DIR", "HOST", "PORT", "OPENAI_API_KEY", 
        "OPENAI_BASE_URL", "LLM_MODEL", "EMBEDDING_MODEL"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        if "API_KEY" in var and value != "NOT SET":
            value = f"{value[:10]}..." if len(value) > 10 else "***"
        print(f"{var}: {value}")
    
    print(f"\nPython: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    if HAS_PSUTIL:
        print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        print(f"CPU count: {psutil.cpu_count()}")
    else:
        print("System info unavailable (psutil not installed)")

def check_port_usage():
    """Check what's using the configured port"""
    port = int(os.getenv("PORT", "9621"))
    print(f"\n=== Port {port} Usage ===")
    
    if not HAS_PSUTIL:
        print("psutil not available - cannot check port usage")
        print("You can check manually with: netstat -tlnp | grep :9621")
        return True  # Assume available if we can't check
    
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                try:
                    process = psutil.Process(conn.pid)
                    print(f"Port {port} is used by PID {conn.pid}: {process.name()}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"Port {port} is used by PID {conn.pid}: <unknown>")
                return False
        print(f"Port {port} is available")
        return True
    except Exception as e:
        print(f"Error checking port usage: {e}")
        return False

def check_file_permissions():
    """Check file permissions"""
    print("\n=== File Permissions ===")
    files_to_check = [
        "/app/lightrag_extended_api.py",
        "/app/startup_wrapper.py", 
        "/app/data/rag_storage"
    ]
    
    for file_path in files_to_check:
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                print(f"{file_path}: {oct(stat.st_mode)[-3:]} (readable: {os.access(file_path, os.R_OK)})")
            else:
                print(f"{file_path}: NOT FOUND")
        except Exception as e:
            print(f"{file_path}: ERROR - {e}")

def test_import():
    """Test importing the main application"""
    print("\n=== Import Test ===")
    try:
        print("Testing FastAPI import...")
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
        
        print("Testing uvicorn import...")
        import uvicorn
        print(f"✅ Uvicorn {uvicorn.__version__}")
        
        print("Testing LightRAG import...")
        import lightrag
        print(f"✅ LightRAG imported")
        
        print("Testing main application import...")
        # This is the critical test - this is where it usually fails
        sys.path.insert(0, '/app')
        from lightrag_extended_api import app
        print("✅ Main application imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_server_test():
    """Run a quick server test"""
    print("\n=== Quick Server Test ===")
    try:
        import uvicorn
        from fastapi import FastAPI
        
        test_app = FastAPI()
        
        @test_app.get("/test")
        async def test_endpoint():
            return {"status": "ok", "message": "test server works"}
        
        port = int(os.getenv("PORT", "9621"))
        
        print(f"Starting test server on port {port}...")
        
        # Start server in background
        config = uvicorn.Config(test_app, host="0.0.0.0", port=port, log_level="error")
        server = uvicorn.Server(config)
        
        import asyncio
        import threading
        
        def run_server():
            asyncio.run(server.serve())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give it time to start
        time.sleep(3)
        
        # Test the endpoint
        import requests
        response = requests.get(f"http://localhost:{port}/test", timeout=5)
        
        if response.status_code == 200:
            print("✅ Test server works!")
            return True
        else:
            print(f"❌ Test server returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test server failed: {e}")
        return False

def main():
    """Main debug function"""
    print("LightRAG Deployment Debug Tool")
    print("=" * 50)
    
    check_environment()
    check_file_permissions()
    port_available = check_port_usage()
    import_ok = test_import()
    
    if port_available and import_ok:
        print("\n=== Running Quick Server Test ===")
        server_ok = run_quick_server_test()
        
        if server_ok:
            print("\n✅ All checks passed! The application should work.")
        else:
            print("\n❌ Server test failed.")
    else:
        print("\n❌ Basic checks failed. Fix these issues first:")
        if not port_available:
            print("- Port is not available")
        if not import_ok:
            print("- Import failed")

    print("\n=== Recommendations ===")
    print("1. Check that all environment variables are set correctly")
    print("2. Ensure OpenAI API key is valid")
    print("3. Check available memory and disk space")
    print("4. Review application logs for specific errors")
    if not HAS_PSUTIL:
        print("5. Install psutil for detailed system monitoring: pip install psutil")

if __name__ == "__main__":
    main() 