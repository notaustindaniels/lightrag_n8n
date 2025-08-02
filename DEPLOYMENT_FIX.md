# Digital Ocean Deployment Fix

## Issue
The deployment to Digital Ocean was failing with the error:
```
ERROR failed health checks after 8 attempts with error Readiness probe failed: dial tcp 10.244.20.50:9621: connect: connection refused
```

## Root Cause
The issue was in the `startup_wrapper.py` file which had a port conflict:
1. Started a temporary health check server on port 9621
2. Tried to start the main server on the same port 9621
3. This caused a port binding conflict, preventing the main server from starting

## Fix Applied

### 1. Fixed Port Conflict in startup_wrapper.py
- Added proper shutdown mechanism for the startup server
- Added signal handlers for graceful shutdown
- Added timing to ensure port is released before main server starts
- Improved error handling and logging

### 2. Updated Health Check Configuration
- Increased `initial_delay_seconds` from 60 to 120 seconds
- Increased `timeout_seconds` from 10 to 15 seconds  
- Increased `failure_threshold` from 5 to 8 attempts
- This gives more time for LightRAG initialization

### 3. Added Debugging Tools
- `test_health.py`: Script to test the health endpoint
- `debug_deployment.py`: Comprehensive debugging script
- Debug script optionally uses psutil if available for system monitoring

### 4. Improved Consistency
- Updated Dockerfile to use `startup_wrapper.py` for consistency
- Added fallback startup code to `lightrag_extended_api.py`

## Key Changes Made

### startup_wrapper.py
- Fixed port conflict by properly shutting down startup server
- Added signal handlers for SIGTERM/SIGINT
- Improved error handling and logging
- Added graceful shutdown mechanisms

### .do/app.yaml
- Increased health check timeouts
- More tolerant failure thresholds

### Dockerfile
- Now uses startup_wrapper.py for consistency
- Same startup process in all environments

## Testing
Run the debug script to verify deployment:
```bash
python debug_deployment.py
```

Run health check test:
```bash
python test_health.py --host your-app-url.ondigitalocean.app --port 443
```

## Expected Behavior
1. Startup server starts on port 9621 with health endpoint returning 503 "starting"
2. Main application imports and initializes (takes 30-60 seconds)
3. Startup server shows "ready" status
4. Startup server shuts down gracefully
5. Main server starts on port 9621
6. Health checks pass with 200 "ready" status

## Troubleshooting
If deployment still fails:
1. Check Digital Ocean app logs for specific error messages
2. Verify all environment variables are set (especially OPENAI_API_KEY)
3. Check memory usage - LightRAG requires adequate memory
4. Ensure the application has time to fully initialize before health checks fail 