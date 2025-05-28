#!/usr/bin/env python3
"""
Migration script to update metadata from sitemap_identifier to sitemap_url
"""
import json
import os
import re

def migrate_metadata(metadata_file_path):
    """Migrate metadata from old format to new format"""
    
    if not os.path.exists(metadata_file_path):
        print(f"Metadata file not found: {metadata_file_path}")
        return
    
    print(f"Loading metadata from: {metadata_file_path}")
    
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    
    migrated_count = 0
    
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
    
    # Save updated metadata
    backup_path = metadata_file_path + '.backup'
    print(f"Creating backup at: {backup_path}")
    
    with open(backup_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saving migrated metadata to: {metadata_file_path}")
    
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMigration complete! Migrated {migrated_count} documents.")
    print(f"Backup saved to: {backup_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
    else:
        # Default path
        metadata_path = "/app/data/rag_storage/document_metadata.json"
    
    migrate_metadata(metadata_path)