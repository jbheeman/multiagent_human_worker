#!/bin/bash

# Script to run test_splits.py for each category sequentially
# This ensures each category runs in its own process, freeing memory between runs

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# List of all categories
CATEGORIES=(
    "All_Beauty"
    "Toys_and_Games"
    "Cell_Phones_and_Accessories"
    "Industrial_and_Scientific"
    "Gift_Cards"
    "Musical_Instruments"
    "Electronics"
    "Handmade_Products"
    "Arts_Crafts_and_Sewing"
    "Baby_Products"
    "Health_and_Household"
    "Office_Products"
    "Digital_Music"
    "Grocery_and_Gourmet_Food"
    "Sports_and_Outdoors"
    "Home_and_Kitchen"
    "Subscription_Boxes"
    "Tools_and_Home_Improvement"
    "Pet_Supplies"
    "Video_Games"
    "Kindle_Store"
    "Clothing_Shoes_and_Jewelry"
    "Patio_Lawn_and_Garden"
    "Unknown"
    "Books"
    "Automotive"
    "CDs_and_Vinyl"
    "Beauty_and_Personal_Care"
    "Amazon_Fashion"
    "Magazine_Subscriptions"
    "Software"
    "Health_and_Personal_Care"
    "Appliances"
    "Movies_and_TV"
)

# Function to check if category is already processed
is_category_processed() {
    local category=$1
    # Check if category appears in any of the output files
    if grep -q "\"category\": \"$category\"" training.jsonl validation.jsonl test.jsonl 2>/dev/null; then
        return 0  # Category is processed
    else
        return 1  # Category is not processed
    fi
}

# Process each category
for category in "${CATEGORIES[@]}"; do
    echo "=========================================="
    echo "Processing category: $category"
    echo "=========================================="
    
    # Check if already processed
    if is_category_processed "$category"; then
        echo "[$category] Already processed, skipping..."
        continue
    fi
    
    # Run the script for this category
    echo "[$category] Starting processing..."
    python test_splits.py --category "$category"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "[$category] ✓ Successfully completed"
    else
        echo "[$category] ✗ Failed with error code $?"
        echo "Continuing with next category..."
    fi
    
    echo ""
done

echo "=========================================="
echo "All categories processed!"
echo "=========================================="

