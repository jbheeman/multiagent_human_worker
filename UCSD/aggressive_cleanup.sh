#!/bin/bash

# More aggressive cleanup - removes entire dataset cache if needed
# Use this if you're sure you won't need to re-download

set -e

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if [ ! -d "$HF_CACHE_DIR" ]; then
    HF_CACHE_DIR="$HOME/.cache/huggingface"
fi

if [ ! -d "$HF_CACHE_DIR" ]; then
    echo "HuggingFace cache directory not found"
    exit 1
fi

echo "WARNING: This will delete the entire Amazon-Reviews-2023 dataset cache!"
echo "Cache directory: $HF_CACHE_DIR"
echo ""
read -p "Are you sure? Type 'yes' to continue: " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

# Find and remove Amazon Reviews dataset
AMAZON_DIR=$(find "$HF_CACHE_DIR" -type d -name "*Amazon-Reviews-2023*" 2>/dev/null | head -1)

if [ -n "$AMAZON_DIR" ]; then
    echo "Found Amazon Reviews cache at: $AMAZON_DIR"
    SIZE_BEFORE=$(du -sh "$AMAZON_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Size: $SIZE_BEFORE"
    echo "Removing..."
    rm -rf "$AMAZON_DIR"
    echo "Removed!"
else
    echo "Amazon Reviews dataset cache not found in expected location"
    echo "Searching for large files..."
    
    # Find large files in cache
    find "$HF_CACHE_DIR" -type f -size +1G -exec ls -lh {} \; 2>/dev/null | head -10
fi

# Clean up empty directories
find "$HF_CACHE_DIR" -type d -empty -delete 2>/dev/null || true

SPACE_AFTER=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo "Cache size after cleanup: $SPACE_AFTER"
echo "Done!"

