# Deployment Changes Summary - Graph Display Fix

## Problem
Knowledge graph not displaying in WebUI on DigitalOcean deployment despite:
- Documents showing in documents tab  
- Graph data existing on server (confirmed via API)
- API returning graph data correctly

## Root Cause
Nested workspace directory structure created by passing `workspace` parameter to LightRAG, causing:
- Files stored in `/workspaces/[name]/[name]/` instead of `/workspaces/[name]/`
- NetworkXStorage looking for GraphML files in wrong location
- Graphs not loading properly on initialization

## Changes Made

### 1. **lightrag_extended_api.py** - Main fixes

#### Workspace Creation Fix (Line ~181)
- Removed `workspace` parameter when creating LightRAG instances
- Prevents double-nesting of directories
```python
# Before: LightRAG(working_dir=workspace_dir, workspace=workspace, ...)
# After: LightRAG(working_dir=workspace_dir, ...)
```

#### Graph Loading Improvements (Line ~193-235)
- Added fallback to check both flat and nested directory structures
- Manually loads and sets graph if NetworkXStorage returns empty
- Checks multiple GraphML paths:
  - `workspace_dir/graph_chunk_entity_relation.graphml`
  - `workspace_dir/workspace_name/graph_chunk_entity_relation.graphml`

#### Empty Label Handling (Line ~1269-1274)
- Converts empty string labels to None in `/graphs` endpoint
- Ensures combined graph approach is used for empty queries

#### Combined Graph Improvements (Line ~396-423)
- Checks both flat and nested structures when loading GraphML files
- Better error handling and fallback mechanisms

#### New Verification Endpoint (Line ~1978-2070)
- Added `/verify` endpoint for deployment health checks
- Reports workspace status, graph data, file structure issues
- Identifies nested workspace directories needing migration

### 2. **migrate_workspace_data.py** - New migration script
- Moves data from nested to flat structure
- Handles GraphML file merging (keeps file with more nodes)
- Merges JSON vector databases
- Creates backups before modifications
- Supports dry-run mode for safety

### 3. **Test Scripts Created**
- `test_graph_fix.py` - Tests basic graph loading
- `test_deployment_fix.py` - Tests deployment-specific issues

## Deployment Instructions

1. **Push changes to master branch**
   ```bash
   git add lightrag_extended_api.py migrate_workspace_data.py
   git commit -m "Fix graph display issue with nested workspace directories"
   git push origin master
   ```

2. **After DigitalOcean redeploys, SSH into server and run migration**
   ```bash
   # Check current structure
   curl https://your-server.com/verify
   
   # If nested workspaces detected, run migration
   python migrate_workspace_data.py --working-dir /app/data/rag_storage
   
   # Verify migration
   curl https://your-server.com/verify
   ```

3. **Verify graph display**
   ```bash
   # Check graph data availability
   curl "https://your-server.com/graphs?label=&max_depth=3"
   
   # Check debug info
   curl https://your-server.com/graph/debug
   ```

4. **Test WebUI**
   - Navigate to https://your-server.com/webui/
   - Go to Knowledge Graph tab
   - Should now see nodes and edges

## Key Endpoints for Verification

- `/verify` - Comprehensive health check
- `/graph/debug` - Detailed graph storage info
- `/graph/reload` - Force reload graphs from disk
- `/graphs?label=` - Get combined graph data

## What This Fixes

1. ✅ Graphs load correctly from nested directories
2. ✅ Empty string labels handled properly
3. ✅ Fallback loading when NetworkXStorage fails
4. ✅ Migration path for existing deployments
5. ✅ Verification tools for troubleshooting

## Important Notes

- The fix is backward compatible
- Migration script creates backups
- No data loss expected
- GraphML files are source of truth

## If Issues Persist

1. Check `/verify` endpoint for warnings/errors
2. Run migration script if nested directories exist
3. Use `/graph/reload` to force refresh
4. Check server logs for graph loading messages
5. Verify GraphML files exist in workspace directories