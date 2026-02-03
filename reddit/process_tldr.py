import json
import os
from collections import defaultdict
import random
import re
import httpx
from smolagents.models import OpenAIServerModel
from openai import OpenAI
from openai import http_client

# --- CONFIGURATION ---
INPUT_FILE = "corpus-webis-tldr-17.json" 
OUTPUT_FILE = "gold_users.jsonl"          


# The "Chameleon" Standard
MIN_SUBREDDITS = 3         # User must be active in 3+ distinct communities
POSTS_PER_SUBREDDIT = 5    # User must have 5+ posts in EACH of those communities

# Split Sizes (How many users we actually need for the experiment)
TRAIN_SIZE = 50  # For GEPA Optimization
VAL_SIZE = 20    # For Validation
TEST_SIZE = 10   # For Final Evaluation

def split_into_train_val_test():
    """Reads the gold users file and creates the final datasets."""
    print(f"\nSplitting {OUTPUT_FILE} into train/val/test...")
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_users = [json.loads(line) for line in f]
    
    # Shuffle to ensure random sampling
    random.seed(42)
    random.shuffle(all_users)
    
    # Check if we have enough data
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(all_users) < total_needed:
        print(f"Warning: Only found {len(all_users)} gold users. Using all of them.")
        # If short on data, just split mostly into train
        train_end = int(len(all_users) * 0.7)
        val_end = int(len(all_users) * 0.9)
    else:
        train_end = TRAIN_SIZE
        val_end = TRAIN_SIZE + VAL_SIZE
        # We slice strictly to keep the dataset sizes manageable
    
    train_users = all_users[:train_end]
    val_users = all_users[train_end:val_end]
    test_users = all_users[val_end:val_end + TEST_SIZE]
    
    # Helper to write files
    def write_set(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            for user in data:
                f.write(json.dumps(user) + "\n")
        print(f"Saved {len(data)} users to {filename}")

    write_set("train_reddit.jsonl", train_users)
    write_set("val_reddit.jsonl", val_users)
    write_set("test_reddit.jsonl", test_users)

def process_dataset():
    print(f"Scanning {INPUT_FILE}... This might take a minute.")
    
    # Pass 1: Identify Candidates (Map Author -> Set of Subreddits)
    author_subreddits = defaultdict(set)
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % 100000 == 0: print(f"Scanned {i} posts...", end="\r")
                
                try:
                    data = json.loads(line)
                    author = data.get("author")
                    subreddit = data.get("subreddit")
                    
                    if author and subreddit and author != "[deleted]":
                        author_subreddits[author].add(subreddit)
                except:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Did you run the download script?")
        return

    # Filter for Gold Authors
    gold_authors = {
        auth for auth, subs in author_subreddits.items() 
        if len(subs) >= MIN_SUBREDDITS
    }
    
    print(f"\nFound {len(gold_authors)} authors active in {MIN_SUBREDDITS}+ subreddits.")
    
    # Pass 2: Extract Content for Gold Authors
    print("Extracting content for Gold Authors...")
    
    user_data = defaultdict(lambda: defaultdict(list))
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0: print(f"Processing {i} posts...", end="\r")
            
            try:
                data = json.loads(line)
                author = data.get("author")
                
                if author in gold_authors:
                    subreddit = data.get("subreddit")
                    content = data.get("content", "") or data.get("body", "")
                    
                    if len(user_data[author][subreddit]) < POSTS_PER_SUBREDDIT:
                        if len(content) > 50: 
                            user_data[author][subreddit].append(content)
            except:
                continue

    # Write to Gold File (Intermediate Step)
    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        count = 0
        for author, sub_dict in user_data.items():
            # Double check constraint: Must have 5 posts in at least 3 distinct subs
            # (Pass 1 only checked if they posted *once* in 3 subs, Pass 2 ensures volume)
            valid_subs = [sub for sub, posts in sub_dict.items() if len(posts) >= POSTS_PER_SUBREDDIT]
            
            if len(valid_subs) >= MIN_SUBREDDITS:
                # Filter sub_dict to ONLY include the valid subreddits
                final_history = []
                final_sub_list = []
                
                for sub in valid_subs:
                    final_sub_list.append(sub)
                    final_history.append({"subreddit": sub, "posts": sub_dict[sub]})

                record = {
                    "user_id": author,
                    "subreddits": final_sub_list,
                    "history": final_history
                }
                f.write(json.dumps(record) + "\n")
                count += 1
                
    print(f"Done! Saved {count} high-quality users to {OUTPUT_FILE}.")
    print(f"You can now safely delete {INPUT_FILE}.")
    
    # Run the split
    split_into_train_val_test()

# teacher_model_raw= OpenAIServerModel( # Still used for persona agent
#         model_id="qwen3",
#         api_base="https://ellm.nrp-nautilus.io/v1",
#         api_key=os.getenv("NAUT_API_KEY"),
#         client_kwargs={"http_client": http_client}
#     )
http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=os.getenv("NAUT_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1",
    http_client=http_client
)
model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.getenv("NAUT_API_KEY"),
)

def extract_json(raw_text):
    """Robust JSON extractor."""
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None
    
def get_demographics(all_text):

  
    prompt = f"""
    Analyze this user's post history and extract their demographics.
    
    USER HISTORY:
    {all_text[:4000]}
    
    CRITICAL INSTRUCTION:
    - For 'age', estimate the MOST LIKELY single integer. Do not give ranges. If they say "I was in high school in 2014", calculate the age (approx 26-27) and pick ONE number (e.g., "27").
    - For 'gender', pick 'male', 'female', or 'non-binary' based on the strongest cues.    
    
    Required JSON Structure:
    {{
        "age": "integer",
        "gender": "string",
        "occupation": "string",
        "location": "string"
    }}
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
            model="gpt-oss",
            messages=messages
        )
    content = extract_json(response.choices[0].message.content)
    print(f"Model response: {content}")
    return content


def get_schwartz_vector(context_text, subreddit):
    prompt = f"""
    Analyze the user's behavior in the subreddit r/{subreddit}.
    Infer their Schwartz Values for THIS CONTEXT ONLY.
    
    POSTS:
    {context_text[:2000]}
    
    Return JSON only (values 0.0 to 1.0): 
    {{ "POWER": 0.0, "ACHIEVEMENT": 0.0, "HEDONISM": 0.0, "STIMULATION": 0.0, "SELF_DIRECTION": 0.0, "UNIVERSALISM": 0.0, "BENEVOLENCE": 0.0, "TRADITION": 0.0, "CONFORMITY": 0.0, "SECURITY": 0.0 }}
    """
    try:
        resp = client.chat.completions.create(model="gpt-oss", messages=[{"role": "user", "content": prompt}])
        return extract_json(resp.choices[0].message.content)
    except:
        return None


def two_pass_enrichment(inputfile, output_file):
    """Performs two-pass enrichment on the dataset."""
    print(f"Starting Two-Pass Enrichment on {inputfile}...")
    
    
    with open(inputfile, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            data = json.loads(line)
            user_id = data['user_id']
            print(f"Processing User {i}: {user_id}...", end="\r")

            all_posts = []
            for item in data['history']:
                all_posts.extend(item['posts'])
            all_text = " ".join(all_posts)

            demographics = get_demographics(all_text)

            if not demographics:
                demographics = {"age": "unknown", "gender": "unknown", "occupation": "unknown", "location": "unknown"}
            
            for item in data['history']:
                subreddit = item['subreddit']
                posts = item['posts']
                sub_text = " ".join(posts)

                vector = get_schwartz_vector(sub_text, subreddit)


                if vector:
                        # Create the Flattened Instance
                    instance = {
                        "user_id": user_id,
                        "subreddit": subreddit,
                        "posts": posts,                 # Input for agent
                        "anchor_demographics": demographics, # Global Identity
                        "shift_vector": vector,         # Local Motivation
                        "target_vector": vector         # For Evaluation
                    }
                fout.write(json.dumps(instance) + "\n")
            

            print(f"\nDone! Saved flat dataset to {output_file}")

if __name__ == "__main__":
    inputfile = "test_reddit.jsonl"
    print(f"Starting Two-Pass Enrichment on {inputfile}...")
    output_file = "test_reddit_enriched.jsonl"
    two_pass_enrichment(inputfile, output_file)