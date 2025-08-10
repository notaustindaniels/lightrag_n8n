#!/usr/bin/env python3
"""
Migration script to fix nested workspace directories created by incorrect configuration.
This moves data from workspaces/[name]/[name]/ to workspaces/[name]/
"""

import os
import shutil
import json
from pathlib import Path
import argparse

def migrate_workspace(base_dir: str, dry_run: bool = False):
    """Migrate workspace data from nested to flat structure"""
    workspaces_dir = os.path.join(base_dir, "workspaces")
    
    if not os.path.exists(workspaces_dir):
        print(f"No workspaces directory found at {workspaces_dir}")
        return
    
    migrations_performed = []
    
    # Iterate through each workspace
    for workspace_name in os.listdir(workspaces_dir):
        workspace_path = os.path.join(workspaces_dir, workspace_name)
        
        if not os.path.isdir(workspace_path):
            continue
            
        # Check if there's a nested directory with the same name
        nested_path = os.path.join(workspace_path, workspace_name)
        
        if os.path.exists(nested_path) and os.path.isdir(nested_path):
            print(f"\nFound nested workspace: {workspace_name}")
            
            # List files in nested directory
            nested_files = os.listdir(nested_path)
            print(f"  Files in nested directory: {nested_files}")
            
            if not dry_run:
                # Move files from nested to parent
                for filename in nested_files:
                    src = os.path.join(nested_path, filename)
                    dst = os.path.join(workspace_path, filename)
                    
                    # Handle existing files
                    if os.path.exists(dst):
                        # For GraphML files, use the one with more nodes
                        if filename.endswith('.graphml'):
                            try:
                                import networkx as nx
                                src_graph = nx.read_graphml(src)
                                dst_graph = nx.read_graphml(dst)
                                
                                if src_graph.number_of_nodes() > dst_graph.number_of_nodes():
                                    print(f"    Replacing {filename} (src has more nodes: {src_graph.number_of_nodes()} > {dst_graph.number_of_nodes()})")
                                    shutil.move(src, dst)
                                else:
                                    print(f"    Keeping existing {filename} (dst has more nodes: {dst_graph.number_of_nodes()} >= {src_graph.number_of_nodes()})")
                                    os.remove(src)
                            except Exception as e:
                                print(f"    Error comparing GraphML files: {e}")
                                print(f"    Backing up existing {filename} to {filename}.backup")
                                shutil.move(dst, f"{dst}.backup")
                                shutil.move(src, dst)
                        # For JSON files, merge if possible
                        elif filename.endswith('.json'):
                            try:
                                with open(src, 'r') as f:
                                    src_data = json.load(f)
                                with open(dst, 'r') as f:
                                    dst_data = json.load(f)
                                
                                # Simple merge strategy - combine if both are dicts
                                if isinstance(src_data, dict) and isinstance(dst_data, dict):
                                    # For vector DB files, merge the data
                                    if 'data' in src_data and 'data' in dst_data:
                                        if isinstance(src_data['data'], list) and isinstance(dst_data['data'], list):
                                            # Merge lists, avoiding duplicates based on ID
                                            existing_ids = {item.get('id') for item in dst_data['data'] if isinstance(item, dict) and 'id' in item}
                                            for item in src_data['data']:
                                                if isinstance(item, dict) and 'id' in item and item['id'] not in existing_ids:
                                                    dst_data['data'].append(item)
                                            
                                            with open(dst, 'w') as f:
                                                json.dump(dst_data, f, indent=2)
                                            print(f"    Merged {filename}")
                                        else:
                                            # Can't merge, backup and replace
                                            print(f"    Backing up existing {filename} to {filename}.backup")
                                            shutil.move(dst, f"{dst}.backup")
                                            shutil.move(src, dst)
                                    else:
                                        # Merge top-level keys
                                        dst_data.update(src_data)
                                        with open(dst, 'w') as f:
                                            json.dump(dst_data, f, indent=2)
                                        print(f"    Merged {filename}")
                                else:
                                    # Can't merge, backup and replace
                                    print(f"    Backing up existing {filename} to {filename}.backup")
                                    shutil.move(dst, f"{dst}.backup")
                                    shutil.move(src, dst)
                                
                                os.remove(src) if os.path.exists(src) else None
                            except Exception as e:
                                print(f"    Error merging JSON files: {e}")
                                print(f"    Backing up existing {filename} to {filename}.backup")
                                shutil.move(dst, f"{dst}.backup")
                                shutil.move(src, dst)
                        else:
                            # For other files, backup existing and move new
                            print(f"    Backing up existing {filename} to {filename}.backup")
                            shutil.move(dst, f"{dst}.backup")
                            shutil.move(src, dst)
                    else:
                        # No conflict, just move
                        print(f"    Moving {filename}")
                        shutil.move(src, dst)
                
                # Remove the now-empty nested directory
                try:
                    os.rmdir(nested_path)
                    print(f"  Removed empty nested directory")
                except OSError as e:
                    print(f"  Warning: Could not remove nested directory: {e}")
                
                migrations_performed.append(workspace_name)
            else:
                print(f"  [DRY RUN] Would migrate files from nested directory")
                migrations_performed.append(workspace_name)
    
    if migrations_performed:
        print(f"\n{'Would migrate' if dry_run else 'Migrated'} {len(migrations_performed)} workspace(s): {', '.join(migrations_performed)}")
    else:
        print("\nNo nested workspaces found. Directory structure is correct.")
    
    return migrations_performed

def main():
    parser = argparse.ArgumentParser(description='Migrate nested workspace directories to flat structure')
    parser.add_argument('--working-dir', '-d', 
                        default=os.getenv('WORKING_DIR', '/app/data/rag_storage'),
                        help='Base working directory (default: $WORKING_DIR or /app/data/rag_storage)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    print(f"Migration Script for LightRAG Workspace Data")
    print(f"Working directory: {args.working_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    
    migrate_workspace(args.working_dir, args.dry_run)
    
    print("\nMigration complete!")

if __name__ == "__main__":
    main()