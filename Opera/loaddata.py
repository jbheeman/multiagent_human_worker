import pandas as pd
import json
import argparse
import os
import re


def filter_for_shopping_information(persona_text):
    """
    Filter for shopping information in the persona text.
    """
    match = re.search(r"### Shopping Preference\s*\n(.*)", persona_text)
    if match:
        return match.group(1)
    return ""



def format_purchases(purchases_df):
    return purchases_df[["title", "price", "options"]].head(10).to_string(index=False)


def save_user_json(user_id, persona_text, user_sessions, purchases_df, out_path=None):
    """
    Assemble a simple JSON for the selected user including persona and purchases
    and save it to disk.
    """
    # Build purchases list (safe if empty)
    purchase_fields = ["session_id", "asin", "title", "price", "options"]
    purchases_list = (
        purchases_df[purchase_fields].to_dict(orient="records") if len(purchases_df) > 0 else []
    )

    data = {
        "user_id": str(user_id),
        "persona": persona_text,
        "num_sessions": len(user_sessions),
        "num_purchases": len(purchases_list),
        "purchases": purchases_list,
    }

    # Determine output path
    if out_path is None or len(str(out_path).strip()) == 0:
        # Default alongside this script inside Opera/
        default_dir = os.path.dirname(__file__)
        out_path = os.path.join(default_dir, f"user_{user_id}.json")

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path

# --- Step 7: Expand product JSONs ---
def extract_products(row):
    """
    Extract products from various sources depending on click type:
    - 'products' column for purchase and cart_side_bar clicks
    - page_meta for product_link clicks (as fallback)
    """
    products = []
    click_type = row.get("click_type", "")
    
    # Method 1: Try 'products' column first (works for purchase and cart_side_bar)
    try:
        if pd.notna(row.get("products")):
            products_data = json.loads(row["products"]) if isinstance(row["products"], str) else row["products"]
            if isinstance(products_data, list):
                products.extend(products_data)
    except Exception:
        pass

    if click_type == "cart_side_bar" and len(products) == 0:
        try:
            if pd.notna(row.get("page_meta")):
                page_meta = json.loads(row["page_meta"]) if isinstance(row["page_meta"], str) else row["page_meta"]
                if isinstance(page_meta, dict) and "cart_items" in page_meta:
                    cart_items = page_meta["cart_items"]
                    if isinstance(cart_items, list):
                        products.extend(cart_items)
        except Exception:
            pass



    
    # Method 2: For product_link clicks, try page_meta as fallback
    if click_type == "product_link" and len(products) == 0:
        try:
            if pd.notna(row.get("page_meta")):
                page_meta = json.loads(row["page_meta"]) if isinstance(row["page_meta"], str) else row["page_meta"]
                if isinstance(page_meta, dict):
                    # Check for cart_items or search_results data
                    if "cart_items" in page_meta:
                        cart_items = page_meta["cart_items"]
                        if isinstance(cart_items, list):
                            products.extend(cart_items)
                    # Check for search_results format: {"name": "search_results", "data": "{\"title\":\"...\",\"asin\":\"...\"}"}
                    elif "search_results" in page_meta or "data" in page_meta:
                        data = page_meta.get("search_results") or page_meta.get("data")
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                            except:
                                pass
                        if isinstance(data, dict) and "asin" in data:
                            products.append(data)
                        elif isinstance(data, list):
                            products.extend([item for item in data if isinstance(item, dict) and "asin" in item])
        except Exception:
            pass
    
    # Format products consistently
    result = []
    for p in products:
        if isinstance(p, dict):
            result.append({
                "session_id": row["session_id"],
                "asin": p.get("asin") if "asin" in p else "0000000000", #default to a dummy ASIN if not present
                "title": p.get("title") if "title" in p else "Unknown Title", #default to a dummy title if not present
                "price": p.get("price") if "price" in p else "Unknown Price", #default to a dummy price if not present
                "options": p.get("options") if "options" in p else None #default to a dummy options if not present
            })
    
    return result






def build_dataset_for_user(user_df, session_df, action_df ):
    
    # Collect all users' data
    all_data = []
    
    total_users = len(user_df)
    skipped_no_interview = 0
    
    for user_id in user_df["user_id"]:
        user_row = user_df[user_df["user_id"] == user_id]
        if len(user_row) == 0:
            skipped_no_interview += 1
            print(f"Skipping user {user_id} - user not found in dataframe")
            continue
            
        persona_info = user_row["interview_transcript_processed"].values[0]
        
        # Skip users without interview (empty or null persona_info)
        # Check multiple ways to ensure we catch all cases
        has_interview = False
        if pd.notna(persona_info):
            persona_str = str(persona_info).strip()
            if persona_str and persona_str.lower() not in ["", "nan", "none", "null"]:
                has_interview = True
        
        if not has_interview:
            skipped_no_interview += 1
            print(f"Skipping user {user_id} - no interview found")
            continue
        
        gold_persona = filter_for_shopping_information(persona_info)
        
        # Use full persona_info if shopping preference extraction failed
        if gold_persona == "":
            gold_persona = str(persona_info).strip()
            
        user_sessions = session_df[session_df["user_id"] == user_id]["session_id"].tolist()
        user_actions = action_df[action_df["session_id"].isin(user_sessions)]
        
        # Include purchases, cart items, and clicked products for more data
        # This gives us more items to work with for GEPA prompt refinement
        interaction_actions = user_actions[
            (user_actions["action_type"] == "click") &
            ((user_actions["click_type"] == "purchase") | 
             (user_actions["click_type"] == "cart_side_bar") | 
             (user_actions["click_type"] == "product_link"))
        ].copy()

        interactions = []
        for _, row in interaction_actions.iterrows():
            click_type = row["click_type"]
            # Map click_type to interaction type
            if click_type == "purchase":
                interaction_type = "purchase"
            elif click_type == "cart_side_bar":
                interaction_type = "cart"
            elif click_type == "product_link":
                interaction_type = "click"
            else:
                interaction_type = "unknown"
            
            # Extract products and add type field
            products = extract_products(row)
            for product in products:
                product["type"] = interaction_type
                interactions.append(product)
        
        # Deduplicate by (session_id, asin, type) to avoid counting same product multiple times
        # but allow same product with different types (e.g., clicked then purchased)
        seen = set()
        unique_interactions = []
        for item in interactions:
            key = (item["session_id"], item["asin"], item["type"])
            if key not in seen:
                seen.add(key)
                unique_interactions.append(item)
        
        # Skip users with no interactions
        if len(unique_interactions) == 0:
            print(f"Skipping user {user_id} - no interactions found")
            continue
        
        # Store as dictionary with interactions as a list
        all_data.append({
            "user_id": str(user_id),
            "gold_persona": gold_persona,
            "interactions": unique_interactions  # List of dicts with type field
        })

        purchase_count = sum(1 for item in unique_interactions if item["type"] == "purchase")
        cart_count = sum(1 for item in unique_interactions if item["type"] == "cart")
        click_count = sum(1 for item in unique_interactions if item["type"] == "click")
        print(f"✓ User {user_id}: {len(unique_interactions)} interactions ({purchase_count} purchases, {cart_count} cart, {click_count} clicks)")
    

    return all_data

if __name__ == "__main__":
    user_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/user/train/train.parquet")
    session_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/session/train/train.parquet")
    action_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/action/train/train.parquet")


    #testing stuff
    user_test_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/user/test/test.parquet")
    test_session_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/session/test/test.parquet")
    test_action_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/action/test/test.parquet")



    # Collect all users' data
    all_training_data = build_dataset_for_user(user_df, session_df, action_df)
    all_test_data = build_dataset_for_user(user_test_df, test_session_df, test_action_df)
        
        # Count by type for reporting
   
    # Now split the data
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Shuffle and split: 70% train, 15% val, 15% test
    train_data, val_data = train_test_split(
        all_training_data,
        test_size=0.1,
        random_state=42,
    )
    test_data = all_test_data

    
    # Save as JSON (parquet doesn't handle nested lists well)
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("data/val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open("data/test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n📊 Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"📈 Total users processed: {len(all_training_data)} for training")
    print(f"📈 Total users in test dataset: {len(all_test_data)} for testing")
