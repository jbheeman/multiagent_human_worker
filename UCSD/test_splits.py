import json
import csv
import gc
import sys
import argparse
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import re
from typing import List

SIGNAL_PATTERNS = [
    # justification / contrast
    r"\b(because|since|so|therefore|however|but|although|though|despite|yet)\b",
    # decision / outcome
    r"\b(returned|return|refund|kept|repurchase|buy again|won't buy|wouldn't buy|recommend|recommended)\b",
    # value / price
    r"\b(price|expensive|cheap|worth|value|overpriced)\b",
    # common product aspects (expand per category later)
    r"\b(quality|smell|scent|texture|fit|size|battery|durable|comfortable|easy|hard)\b",
    # strong sentiment
    r"\b(love|hate|amazing|terrible|awful|great|perfect|disappointed)\b",
    # numbers often indicate specifics
    r"\b\d+(\.\d+)?\b",
]

COMPILED_SIGNALS = re.compile("|".join(SIGNAL_PATTERNS), re.IGNORECASE)

def split_sentences(text: str) -> List[str]:
    # Simple sentence split on ., !, ? while keeping content
    # (Not perfect, but much better than split("."))
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def score_sentence(s: str) -> int:
    s_low = s.lower()
    matches = COMPILED_SIGNALS.findall(s_low)
    score = len(matches) * 2
    score += min(len(s), 200) // 80  # 0..2
    return score

def generate_high_signal_excerpt(review_text: str, max_chars: int = 900, min_sentences: int = 2) -> str:
    if not review_text or not review_text.strip():
        return ""

    sents = split_sentences(review_text)
    if len(sents) <= min_sentences:
        return review_text[:max_chars]

    scored = [(score_sentence(s), i, s) for i, s in enumerate(sents)]
    scored.sort(reverse=True)

    chosen = []
    used_chars = 0

    # pick top sentences, then output in original order for coherence
    for _, _, s in scored:
        if used_chars + len(s) + 1 > max_chars:
            continue
        chosen.append(s)
        used_chars += len(s) + 1
        if used_chars >= max_chars * 0.7 and len(chosen) >= min_sentences:
            break

    if not chosen:
        return review_text[:max_chars]

    # preserve original order
    chosen_set = set(chosen)
    ordered = [s for s in sents if s in chosen_set]
    excerpt = " ".join(ordered).strip()
    return excerpt[:max_chars]

categories = [
"All_Beauty", 
"Toys_and_Games",
"Cell_Phones_and_Accessories",
"Industrial_and_Scientific",
"Gift_Cards",
"Musical_Instruments",
"Electronics",
"Handmade_Products",
"Arts_Crafts_and_Sewing",
"Baby_Products",
"Health_and_Household",
"Office_Products",
"Digital_Music",
"Grocery_and_Gourmet_Food",
"Sports_and_Outdoors",
# "Home_and_Kitchen",
"Subscription_Boxes",
"Tools_and_Home_Improvement",
"Pet_Supplies",
"Video_Games",
"Kindle_Store",
"Clothing_Shoes_and_Jewelry",
"Patio_Lawn_and_Garden",
# "Unknown",
# "Books",
"Automotive",
"CDs_and_Vinyl",
"Beauty_and_Personal_Care",
"Amazon_Fashion",
"Magazine_Subscriptions",
"Software",
"Health_and_Personal_Care",
"Appliances",
"Movies_and_TV"]
TARGET_N = 30
MIN_REVIEWS = 5
MAX_REVIEWS = 8

# Files are opened in "a" mode (append) to resume from where we left off
training_f = open("training.jsonl", "a")
val_f = open("validation.jsonl", "a")
test_f = open("test.jsonl", "a")

# Track which categories have already been processed
def get_processed_categories():
    """Read existing output files to determine which categories are already done."""
    processed = set()
    for filename in ["training.jsonl", "validation.jsonl", "test.jsonl"]:
        try:
            with open(filename, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed.add(data.get("category"))
        except FileNotFoundError:
            pass
    return processed

processed_categories = get_processed_categories()
if processed_categories:
    print(f"Already processed categories: {sorted(processed_categories)}")
    print(f"Skipping these categories and continuing with the rest...")


def pick_users_from_5core(category: str) -> list[str]:
    """Stream the 5-core CSV and collect TARGET_N unique user_ids."""
    csv_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"benchmark/5core/rating_only/{category}.csv",
        repo_type="dataset",
    )
    users = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            users.append(uid)
            if len(users) >= TARGET_N:
                break
    return users


def collect_reviews_for_users(review_path: str, target_users: list[str]) -> dict[str, list[dict]]:
    """Stream review JSONL and collect up to MIN_REVIEWS reviews per target user, keeping only needed fields."""
    target = set(target_users)
    user_reviews = {u: [] for u in target_users}
    
    # Read directly from file instead of using load_dataset to avoid pyarrow buffering
    try:
        # Download the review file once
        category_from_path = review_path.split("/")[-1].replace(".jsonl", "")
        review_local_path = hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename=f"raw/review_categories/{category_from_path}.jsonl",
            repo_type="dataset",
        )
        
        # Read line-by-line to minimize memory usage
        line_count = 0
        with open(review_local_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                # Check memory every 100k lines
                if line_count % 100000 == 0:
                    check_memory_and_exit_if_high()
                
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    uid = r.get("user_id")
                    if uid in target and len(user_reviews[uid]) < MIN_REVIEWS:
                        # Keep only the fields you actually need (saves RAM)
                        user_reviews[uid].append({
                            "user_id": uid,
                            "parent_asin": r.get("parent_asin"),
                            "timestamp": r.get("timestamp", 0),
                            "rating": r.get("rating"),
                            "title": r.get("title"),
                            "text": r.get("text", ""),
                        })
                    # Stop once all users have MIN_REVIEWS reviews
                    if all(len(v) >= MIN_REVIEWS for v in user_reviews.values()):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"WARNING: Could not read reviews directly: {e}, falling back to load_dataset")
        # Fallback to load_dataset
        reviews_ds = load_dataset("json", data_files=review_path, split="train", streaming=True)
        for r in reviews_ds:
            uid = r.get("user_id")
            if uid in target and len(user_reviews[uid]) < MIN_REVIEWS:
                user_reviews[uid].append({
                    "user_id": uid,
                    "parent_asin": r.get("parent_asin"),
                    "timestamp": r.get("timestamp", 0),
                    "rating": r.get("rating"),
                    "title": r.get("title"),
                    "text": r.get("text", ""),
                })
            if all(len(v) >= MIN_REVIEWS for v in user_reviews.values()):
                break
    
    # Filter to only users with at least MIN_REVIEWS reviews
    return {uid: revs for uid, revs in user_reviews.items() if len(revs) >= MIN_REVIEWS}


def process_category(category: str):
    """Stream reviews and metadata for a single category, then append splits."""
    review_path = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{category}.jsonl"
    meta_path   = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_{category}.jsonl"
    
    # Step A: Stream the 5-core CSV and collect TARGET_N unique user_ids
    try:
        target_users = pick_users_from_5core(category)
        print(f"[{category}] Collected {len(target_users)} unique users from 5core CSV")
    except Exception as e:
        print(f"[{category}] WARNING: Could not load 5core CSV: {e}. Skipping category.")
        return
    
    if not target_users:
        print(f"[{category}] No users found in 5core CSV. Skipping category.")
        return
    
    # Step B: Stream the review JSONL and collect up to MIN_REVIEWS reviews per target user
    user_reviews = collect_reviews_for_users(review_path, target_users)
    
    # Filter to only users with at least MIN_REVIEWS reviews
    user_reviews = {uid: revs for uid, revs in user_reviews.items() if len(revs) >= MIN_REVIEWS}
    
    # Limit to TARGET_N users if we have more
    if len(user_reviews) > TARGET_N:
        # Sort by user_id for deterministic selection
        sorted_users = sorted(user_reviews.keys())[:TARGET_N]
        user_reviews = {uid: user_reviews[uid] for uid in sorted_users}

    print(f"[{category}] Selected {len(user_reviews)} users with at least {MIN_REVIEWS} reviews")
    if not user_reviews:
        return

    # Step C: Build needed ASINs for this category (meta only for ASINs you actually need)
    needed_asins = set()
    for _, revs in user_reviews.items():
        for r in revs:
            pa = r.get("parent_asin")
            if pa:
                needed_asins.add(pa)

    # Stream metadata once; keep only needed ASINs
    # Read JSONL line-by-line to avoid pyarrow type strictness issues (e.g., price field type inconsistency)
    meta_dict = {}
    try:
        # Download and read the metadata file directly (avoids pyarrow type strictness)
        # Note: hf_hub_download will cache the file, so subsequent calls are fast
        meta_local_path = hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            filename=f"raw/meta_categories/meta_{category}.jsonl",
            repo_type="dataset",
        )
        
        # Read line-by-line to minimize memory usage
        line_count = 0
        with open(meta_local_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                # Check memory every 100k lines
                if line_count % 100000 == 0:
                    check_memory_and_exit_if_high()
                
                if not line.strip():
                    continue
                try:
                    m = json.loads(line)
                    asin = m.get("parent_asin") or m.get("asin")
                    if asin in needed_asins:
                        # Handle price field - it can be number or string
                        price = m.get("price")
                        if price is not None and isinstance(price, str):
                            try:
                                price = float(price)
                            except (ValueError, TypeError):
                                price = None
                        
                        meta_dict[asin] = {
                            "title": m.get("title"),
                            "brand": m.get("Brand") or m.get("brand"),
                            "price": price,
                            "main_category": m.get("main_category"),
                        }
                        if len(meta_dict) == len(needed_asins):
                            break
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
    except Exception as e:
        print(f"[{category}] WARNING: Could not load metadata via direct file read: {e}")
        print(f"[{category}] Attempting fallback to datasets library...")
        # Fallback to datasets library (may fail on type inconsistencies)
        try:
            meta_ds = load_dataset("json", data_files=meta_path, split="train", streaming=True)
            for m in meta_ds:
                asin = m.get("parent_asin") or m.get("asin")
                if asin in needed_asins:
                    meta_dict[asin] = {
                        "title": m.get("title"),
                        "brand": m.get("Brand") or m.get("brand"),
                        "price": m.get("price"),
                        "main_category": m.get("main_category"),
                    }
                    if len(meta_dict) == len(needed_asins):
                        break
        except Exception as e2:
            print(f"[{category}] ERROR: Could not load metadata: {e2}")
            # Continue without metadata - products will have empty product dicts

    def enrich(review):
        asin = review.get("parent_asin")
        prod = meta_dict.get(asin, {})
        return {
            "parent_asin": asin,
            "rating": review.get("rating"),
            "timestamp_ms": review.get("timestamp"),
            "review_excerpt": generate_high_signal_excerpt(review.get("text", "")),
            "review_full": review.get("text"),
            "review_title": review.get("title"),
            "product": prod,
        }

    # Deterministic ordering so splits are reproducible per category
    for idx, uid in enumerate(sorted(user_reviews.keys())):
        revs = user_reviews[uid]
        revs_sorted = sorted(revs, key=lambda x: x.get("timestamp", 0))
        if len(revs_sorted) < 2:
            continue
        history = revs_sorted[:-1]
        heldout = revs_sorted[-1]
        out = {
            "user_id": uid,
            "category": category,
            "history": [enrich(r) for r in history],
            "heldout": enrich(heldout),
        }
        if idx < 26:
            training_f.write(json.dumps(out) + "\n")
        elif idx < 29:
            val_f.write(json.dumps(out) + "\n")
        else:
            test_f.write(json.dumps(out) + "\n")
    
    # Flush files after each category to ensure data is persisted
    training_f.flush()
    val_f.flush()
    test_f.flush()
    
    # Explicit cleanup of large objects
    del user_reviews, needed_asins, meta_dict


try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process Amazon review categories')
parser.add_argument('--category', type=str, help='Process a single category (if not provided, processes all)')
parser.add_argument('--max-memory-mb', type=int, default=6000, help='Maximum memory in MB before early exit (default: 6000)')
args = parser.parse_args()

# Memory monitoring function (defined after args so it can access args.max_memory_mb)
def check_memory_and_exit_if_high():
    """Check memory usage and exit if too high."""
    if not HAS_PSUTIL:
        return
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        if mem_mb > args.max_memory_mb:
            print(f"WARNING: Memory usage ({mem_mb:.1f} MB) exceeds threshold ({args.max_memory_mb} MB)")
            print("Exiting early to prevent OOM kill...")
            sys.exit(1)
    except Exception:
        pass  # If we can't check memory, continue anyway

# Determine which categories to process
if args.category:
    # Process only the specified category
    if args.category not in categories:
        print(f"ERROR: Category '{args.category}' not found in categories list")
        sys.exit(1)
    categories_to_process = [args.category]
    print(f"Processing single category: {args.category}")
else:
    # Process all categories, skipping already processed ones
    categories_to_process = [cat for cat in categories if cat not in processed_categories]
    print(f"Processing {len(categories_to_process)} categories (skipping {len(processed_categories)} already processed)")

for cat in categories_to_process:
    try:
        # Check memory before starting
        check_memory_and_exit_if_high()
        
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[{cat}] Memory before: {mem_before:.1f} MB (max: {args.max_memory_mb} MB)")
        
        process_category(cat)
        
        # Check memory after processing
        check_memory_and_exit_if_high()
        
        if HAS_PSUTIL:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[{cat}] Memory after: {mem_after:.1f} MB (delta: {mem_after - mem_before:.1f} MB)")
        gc.collect()  # Force garbage collection between categories
        
        if HAS_PSUTIL:
            mem_after_gc = process.memory_info().rss / 1024 / 1024  # MB
            print(f"[{cat}] Memory after GC: {mem_after_gc:.1f} MB")
        
        # Aggressive cleanup between categories
        gc.collect()
        gc.collect()  # Call twice to handle cyclic references
    except Exception as e:
        # Log and continue with next category to avoid losing progress
        print(f"[{cat}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()  # Clean up even on error
        gc.collect()  # Call twice
        if args.category:
            # If processing single category, exit on error
            sys.exit(1)
        continue


training_f.close()
val_f.close()
test_f.close()

print("Wrote training.jsonl, validation.jsonl, test.jsonl")