#!/usr/bin/env python3
"""
Startup wrapper for LightRAG Extended API Server
Handles initialization and provides better health check support
"""
import os
import sys
import time
import asyncio
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Simple health check server to respond during startup
startup_app = FastAPI()
is_ready = False

@startup_app.get("/health")
async def startup_health():
    if is_ready:
        return JSONResponse({"status": "ready", "service": "lightrag"}, status_code=200)
    else:
        return JSONResponse({"status": "starting", "service": "lightrag"}, status_code=503)

def run_startup_server():
    """Run a minimal server that responds to health checks during startup"""
    uvicorn.run(
        startup_app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "9621")),
        log_level="error"
    )

def main():
    global is_ready
    
    print("Starting LightRAG Extended API Server...")
    print(f"Port: {os.getenv('PORT', '9621')}")
    print(f"Working directory: {os.getenv('WORKING_DIR', '/app/data/rag_storage')}")
    
    # Start the temporary health check server in a separate thread
    startup_thread = Thread(target=run_startup_server, daemon=True)
    startup_thread.start()
    print("Health check server started, initializing main application...")
    
    # Give the startup server a moment to start
    time.sleep(2)
    
    try:
        # Import the main application (this is where the heavy initialization happens)
        print("Importing LightRAG application...")
        from lightrag_extended_api import app, rag_instance
        
        print("LightRAG imported successfully")
        
        # Mark as ready
        is_ready = True
        print("Application initialized and ready")
        
        # Stop the startup server by letting it naturally exit
        # The main server will take over the port
        time.sleep(1)
        
        # Run the main application
        port = int(os.getenv("PORT", "9621"))
        host = os.getenv("HOST", "0.0.0.0")
        
        print(f"Starting main server on {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()