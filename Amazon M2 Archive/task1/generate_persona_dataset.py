"""
Generate dataset for persona generation:
- Loads products from Kaggle using kagglehub
- Parses sessions and converts ASINs to product details
- Outputs CSV with: session_id, prev_items_asin, prev_items_details, persona (empty)
"""

import pandas as pd
import re
import kagglehub
from kagglehub import KaggleDatasetAdapter

print("=" * 80)
print("STEP 1: Downloading products_train.csv from Kaggle...")
print("=" * 80)

# Download products_train.csv using kagglehub
products_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "marquis03/amazon-m2",
    "products_train.csv"
)

print(f"Loaded {len(products_df):,} products")
print(f"Columns: {list(products_df.columns)}")

print("\n" + "=" * 80)
print("STEP 2: Building product lookup dictionary (ASIN -> product details)...")
print("=" * 80)

# Build dictionary keyed by ASIN for O(1) lookups
products_dict = {}
for _, row in products_df.iterrows():
    products_dict[row['id']] = row.to_dict()

print(f"Product dictionary ready with {len(products_dict):,} products")

print("\n" + "=" * 80)
print("STEP 3: Loading sessions...")
print("=" * 80)

# Load sessions
sessions_file = "sessions_test_task1_phase1.csv"
sessions_df = pd.read_csv(sessions_file)
print(f"Loaded {len(sessions_df):,} sessions")

def parse_prev_items(prev_items_str):
    """Parse the prev_items string into a list of ASINs"""
    # Extract all ASINs (10 alphanumeric characters)
    asins = re.findall(r'\b([A-Z0-9]{10})\b', str(prev_items_str))
    return asins

def format_product_details(asins, products_dict):
    """
    Format product details as bullet points:
    - Title (Brand)
    
    Returns formatted string and list of found products
    """
    found_products = []
    for asin in asins:
        if asin in products_dict:
            product = products_dict[asin]
            title = product.get('title', 'Unknown Title')
            brand = product.get('brand', 'None')
            
            # Format as: - Title (Brand)
            formatted = f"- {title} ({brand})"
            found_products.append(formatted)
    
    # Join with newlines
    return "\n".join(found_products), len(found_products)

print("\n" + "=" * 80)
print("STEP 4: Processing sessions and converting ASINs to product details...")
print("=" * 80)

# Prepare output data
output_data = []
sessions_processed = 0
total_sessions = len(sessions_df)

for idx, row in sessions_df.iterrows():
    # Parse ASINs
    asins = parse_prev_items(row['prev_items'])
    
    # Format product details
    product_details, num_found = format_product_details(asins, products_dict)
    
    # Create output row
    output_row = {
        'session_id': idx,
        'prev_items_asin': row['prev_items'],
        'prev_items_details': product_details,
        'persona': ''  # Empty for now, to be filled by persona generation
    }
    output_data.append(output_row)
    
    sessions_processed += 1
    if sessions_processed % 10000 == 0:
        print(f"Processed {sessions_processed:,} / {total_sessions:,} sessions ({sessions_processed/total_sessions*100:.1f}%)")

print(f"\nProcessed all {sessions_processed:,} sessions")

print("\n" + "=" * 80)
print("STEP 5: Saving to CSV...")
print("=" * 80)

# Create output DataFrame
output_df = pd.DataFrame(output_data)
output_file = "persona_dataset.csv"
output_df.to_csv(output_file, index=False)

print(f"✓ Saved to {output_file}")
print(f"  Total sessions: {len(output_df):,}")
print(f"  Columns: {list(output_df.columns)}")

# Show sample
print("\n" + "=" * 80)
print("SAMPLE OUTPUT (first 2 sessions):")
print("=" * 80)
for idx in range(min(2, len(output_df))):
    row = output_df.iloc[idx]
    print(f"\nSession {row['session_id']}:")
    print(f"ASINs: {row['prev_items_asin'][:100]}...")
    print(f"Product Details:\n{row['prev_items_details'][:500]}...")
    print("-" * 40)

print("\n✓ Dataset generation complete!")

