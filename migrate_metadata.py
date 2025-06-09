#!/usr/bin/env python3
"""
Migration script to update metadata from sitemap_identifier to sitemap_url
and add display_name fields for better UI display
"""
import json
import os
import re

def generate_display_name_from_file_path(file_path: str, doc_id: str) -> str:
    """Generate a display name from a file path for legacy documents"""
    if "[" in file_path and "]" in file_path:
        parts = file_path.split("] ", 1)
        if len(parts) == 2:
            domain_part = parts[0] + "]"
            path_part = parts[1]
            if "/" in path_part:
                last_part = path_part.split("/")[-1]
                return f"{domain_part} {last_part}"
            else:
                return file_path
        else:
            return file_path
    else:
        return f"text/{doc_id[:8]}..."

def migrate_metadata(metadata_file_path):
    """Migrate metadata from old format to new format"""
    
    if not os.path.exists(metadata_file_path):
        print(f"Metadata file not found: {metadata_file_path}")
        return
    
    print(f"Loading metadata from: {metadata_file_path}")
    
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    
    migrated_count = 0
    display_name_count = 0
    
    for doc_id, doc_metadata in metadata.items():
        # Check if we have an old sitemap_identifier
        if 'sitemap_identifier' in doc_metadata and 'sitemap_url' not in doc_metadata:
            # Extract URL from [SITEMAP: url] format
            sitemap_identifier = doc_metadata['sitemap_identifier']
            match = re.match(r'\[SITEMAP: (.+)\]', sitemap_identifier)
            
            if match:
                sitemap_url = match.group(1)
                doc_metadata['sitemap_url'] = sitemap_url
                migrated_count += 1
                print(f"Migrated {doc_id}: {sitemap_identifier} -> {sitemap_url}")
        
        # Also check content_summary for sitemap identifiers
        if 'content_summary' in doc_metadata:
            content = doc_metadata['content_summary']
            # Update [SITEMAP: url] format in content to use sitemap_url from metadata
            if 'sitemap_url' in doc_metadata and '[SITEMAP:' in content:
                old_pattern = r'\[SITEMAP: [^\]]+\]'
                new_sitemap = f"[SITEMAP: {doc_metadata['sitemap_url']}]"
                doc_metadata['content_summary'] = re.sub(old_pattern, new_sitemap, content, count=1)
        
        # Add display_name if it doesn't exist
        if 'display_name' not in doc_metadata and 'file_path' in doc_metadata:
            file_path = doc_metadata['file_path']
            doc_metadata['display_name'] = generate_display_name_from_file_path(file_path, doc_id)
            display_name_count += 1
            print(f"Added display_name for {doc_id}: {doc_metadata['display_name']}")
        
        # Add original_doc_id for hash-based IDs to support new custom ID system
        if 'original_doc_id' not in doc_metadata and doc_id.startswith('doc-'):
            doc_metadata['original_doc_id'] = doc_id
    
    # Save updated metadata
    backup_path = metadata_file_path + '.backup'
    print(f"Creating backup at: {backup_path}")
    
    with open(backup_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saving migrated metadata to: {metadata_file_path}")
    
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMigration complete!")
    print(f"- Migrated {migrated_count} documents from sitemap_identifier to sitemap_url")
    print(f"- Added display_name to {display_name_count} documents")
    print(f"Backup saved to: {backup_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
    else:
        # Default path
        metadata_path = "/app/data/rag_storage/document_metadata.json"
    
    migrate_metadata(metadata_path)