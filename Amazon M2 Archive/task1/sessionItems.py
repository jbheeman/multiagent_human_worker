"""
Quick example: Extract items from a single session and look up product details
Uses O(1) dictionary lookup for fast product searches
"""

import pandas as pd
import re

# Example 1: Parse prev_items from test sessions
def parse_prev_items(prev_items_str):
    """Parse the prev_items string into a list of ASINs (space-separated within list)"""
    # Extract all ASINs (10 alphanumeric characters)
    # This handles space-separated format: "['B08V12CT4C' 'B08V1KXBQD' ...]"
    asins = re.findall(r'\b([A-Z0-9]{10})\b', prev_items_str)
    return asins

test_file = "sessions_test_task1_phase1.csv"
df = pd.read_csv(test_file)

print("Sample session:")
print(df.head(3))

# Get items from first session
first_session = df.iloc[311675]
items = parse_prev_items(first_session['prev_items'])
print(f"\nItems from first session: {items}")
print(f"Number of items: {len(items)}")

# Example 3: Look up product details for these ASINs in products_train.csv
# Use O(1) dictionary lookup for fast searches
products_file = "../products_train.csv"
print(f"\nLoading products into dictionary for O(1) lookup...")


# Build dictionary keyed by ASIN for O(1) lookups
products_dict = {}
for chunk in pd.read_csv(products_file, chunksize=500000):
    for _, row in chunk.iterrows():
        products_dict[row['id']] = row.to_dict()
    
    # Progress indicator
    print(f"  Loaded {len(products_dict):,} products...", end='\r')

print(f"\nTotal products loaded: {len(products_dict):,}")

# O(1) lookups
print(f"\nLooking up {len(items)} items...")
found_products = []
for asin in items:
    if asin in products_dict:
        found_products.append(products_dict[asin])

if found_products:
    product_df = pd.DataFrame(found_products)
    print(f"\nFound {len(found_products)} products (including duplicates):")
    print(product_df[['id', 'title', 'price', 'brand', 'locale']].head(10))
    
    # Filter by UK locale (check for both 'UK' and 'GB' codes)
    uk_products = product_df[product_df['locale'].isin(['UK', 'GB'])]
    print(f"\nProducts with UK locale: {len(uk_products)}")
    
    if len(uk_products) > 0:
        # Remove duplicates based on ID (keep first occurrence)
        uk_products_unique = uk_products.drop_duplicates(subset='id', keep='first')
        print(f"Unique UK products: {len(uk_products_unique)}")
        
        # Select only the required fields: title, price, brand, color, size, model
        required_fields = ['id', 'title', 'price', 'brand', 'color', 'size', 'model']
        # Only include fields that exist in the dataframe
        available_fields = [field for field in required_fields if field in uk_products_unique.columns]
        uk_products_filtered = uk_products_unique[available_fields]
        
        # Create JSON file with ASIN as key (format: {"ASIN": {"title": ..., "price": ..., ...}})
        # Set ID as index and use 'index' orient
        uk_products_indexed = uk_products_filtered.set_index('id')
        uk_products_indexed.to_json('products.json', orient='index')
        
        # This will create: {"B08V12CT4C": {"title": "...", "price": "...", ...}, ...}
        print("\nSaved to products.json with UK locale items only!")
        print(f"Fields included: {', '.join(available_fields[1:])}")  # Exclude 'id' from display
    else:
        print("\nNo products found with UK locale.")
else:
    print("\nNo products found.")

