import pandas as pd
import json
import argparse
import os


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

# --- CLI args (optional: allow overriding output path) ---
parser = argparse.ArgumentParser(description="Create a JSON of user persona and purchases from OPeRA")
parser.add_argument("--out", type=str, default=None, help="Output JSON file path")
args = parser.parse_args()

# --- Step 1: Load all 3 tables (from the FULL version) ---
user_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/user/train/train.parquet")
session_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/session/train/train.parquet")
action_df = pd.read_parquet("hf://datasets/NEU-HAI/OPeRA/OPeRA_full/action/train/train.parquet")

# --- Step 2: Pick one user (you can change this) ---
user_id = user_df["user_id"].iloc[1]

# --- Step 3: Display their persona summary ---
persona_info = user_df.loc[user_df["user_id"] == user_id, "interview_transcript_processed"].values[0]
print(f"\n🧠 Persona Summary for {user_id}:\n")
print(persona_info)

# --- Step 4: Get this user's sessions ---
user_sessions = session_df[session_df["user_id"] == user_id]["session_id"].tolist()
print(f"\n🛒 Found {len(user_sessions)} sessions for this user.\n")

# --- Step 5: Filter actions belonging to those sessions ---
user_actions = action_df[action_df["session_id"].isin(user_sessions)]

# --- Step 6: Extract purchase clicks ---
purchase_actions = user_actions[
    (user_actions["action_type"] == "click") &
    (user_actions["click_type"] == "purchase")
].copy()

# --- Step 7: Expand product JSONs ---
def extract_products(row):
    try:
        products = json.loads(row["products"]) if pd.notna(row["products"]) else []
        return [
            {
                "session_id": row["session_id"],
                "asin": p.get("asin"),
                "title": p.get("title"),
                "price": p.get("price"),
                "options": p.get("options")
            }
            for p in products
        ]
    except Exception:
        return []

# Flatten all purchases into a single list
purchased_items = []
for _, row in purchase_actions.iterrows():
    purchased_items.extend(extract_products(row))

purchases_df = pd.DataFrame(purchased_items)




# --- Step 8: Display results ---
if len(purchases_df) == 0:
    print(f"No recorded purchases for user {user_id}.")
else:
    print(f"\n🛍️ Purchases made by {user_id}:\n")
    print(format_purchases(purchases_df))

# --- Step 9: Save consolidated JSON ---
output_path = save_user_json(
    user_id=user_id,
    persona_text=persona_info,
    user_sessions=user_sessions,
    purchases_df=purchases_df,
    out_path=args.out,
)
print(f"\n💾 Saved user JSON to: {output_path}")
