#!/usr/bin/env python3
"""
Startup wrapper for LightRAG Extended API Server
Handles initialization and provides better health check support
"""
import os
import sys
import time
import asyncio
import signal
from threading import Thread, Event
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Simple health check server to respond during startup
startup_app = FastAPI()
is_ready = False
startup_server = None
shutdown_event = Event()

@startup_app.get("/health")
async def startup_health():
    if is_ready:
        return JSONResponse({"status": "ready", "service": "lightrag"}, status_code=200)
    else:
        return JSONResponse({"status": "starting", "service": "lightrag"}, status_code=503)

@startup_app.get("/")
async def startup_root():
    if is_ready:
        return JSONResponse({"message": "LightRAG Extended API", "status": "ready"}, status_code=200)
    else:
        return JSONResponse({"message": "LightRAG Extended API", "status": "starting"}, status_code=503)

def run_startup_server():
    """Run a minimal server that responds to health checks during startup"""
    global startup_server
    config = uvicorn.Config(
        startup_app,
        host="0.0.0.0", 
        port=int(os.getenv("PORT", "9621")),
        log_level="error",
        access_log=False
    )
    startup_server = uvicorn.Server(config)
    
    # Run until shutdown event is set
    try:
        asyncio.run(startup_server.serve())
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"Startup server error: {e}")

def shutdown_startup_server():
    """Properly shutdown the startup server"""
    global startup_server
    if startup_server:
        print("Shutting down startup server...")
        shutdown_event.set()
        if hasattr(startup_server, 'should_exit'):
            startup_server.should_exit = True
        if hasattr(startup_server, 'force_exit'):
            startup_server.force_exit = True
        time.sleep(1)  # Give it time to shutdown gracefully

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, shutting down...")
    shutdown_startup_server()
    sys.exit(0)

def main():
    global is_ready
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting LightRAG Extended API Server...")
    print(f"Port: {os.getenv('PORT', '9621')}")
    print(f"Working directory: {os.getenv('WORKING_DIR', '/app/data/rag_storage')}")
    
    # Start the temporary health check server in a separate thread
    startup_thread = Thread(target=run_startup_server, daemon=True)
    startup_thread.start()
    print("Health check server started, initializing main application...")
    
    # Give the startup server a moment to start
    time.sleep(3)
    
    try:
        # Import the main application (this is where the heavy initialization happens)
        print("Importing LightRAG application...")
        from lightrag_extended_api import app, rag_instance
        
        print("LightRAG imported successfully")
        
        # Mark as ready
        is_ready = True
        print("Application initialized and ready")
        
        # Give health checks a moment to see the ready state
        time.sleep(2)
        
        # Properly shutdown the startup server
        shutdown_startup_server()
        
        # Wait a bit for the port to be fully released
        time.sleep(3)
        
        # Run the main application
        port = int(os.getenv("PORT", "9621"))
        host = os.getenv("HOST", "0.0.0.0")
        
        print(f"Starting main server on {host}:{port}")
        
        # Use uvicorn.Config and Server for better control
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        
        # Run the server
        asyncio.run(server.serve())
        
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        shutdown_startup_server()
        sys.exit(1)

if __name__ == "__main__":
    main()