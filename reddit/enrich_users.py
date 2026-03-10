import json
import os
import re
import httpx
from openai import OpenAI

# Make sure to set NAUT_API_KEY in your environment before running
INPUT_FILE = "thousand_users_raw.jsonl"
OUTPUT_FILE = "thousand_reddit_enriched.jsonl"

http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=os.getenv("NAUT_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1",
    http_client=http_client
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

def get_schwartz_vector_chameleon(user_id, contextual_posts):
    # Format the posts: "Subreddit R/X: Post 1, Post 2... Subreddit R/Y: ..."
    history_str = ""
    for sub, posts in contextual_posts.items():
        history_str += f"\n[SUBREDDIT: r/{sub}]\n" + "\n".join(posts[:3])

    # Shortening definitions to save tokens while maintaining accuracy
    schwartz_defs = """
    1. POWER: Social status, prestige, control/dominance over people/resources.
    2. ACHIEVEMENT: Personal success through demonstrating competence.
    3. HEDONISM: Pleasure and sensuous gratification for oneself.
    4. STIMULATION: Excitement, novelty, and challenge in life.
    5. SELF-DIRECTION: Independent thought and action, choosing, creating.
    6. UNIVERSALISM: Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.
    7. BENEVOLENCE: Preservation and enhancement of the welfare of people with whom one is in frequent personal contact.
    8. TRADITION: Respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide.
    9. CONFORMITY: Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations.
    10. SECURITY: Safety, harmony, and stability of society, of relationships, and of self.
    """

    prompt = f"""
    You are a Quantitative Psychologist. Analyze the provided Reddit post history to infer the user's stable personal values according to the Schwartz Theory of Basic Human Values.
    Identify the STABLE underlying Schwartz Values that persist across these different social contexts.
    
    DEFINITIONS:
    {schwartz_defs}

    POST HISTORY FOR USER {user_id}:
    {history_str[:6000]} 

    TASK:
    Based on the linguistic cues, tone, and stated beliefs in these posts, estimate the relative importance of each value to this individual on a scale of 0.0 (Not at all important/rejected) to 1.0 (Most important/central to identity).
    
    Return ONLY a raw JSON object:
    {{
      "POWER": 0.0, "ACHIEVEMENT": 0.0, "HEDONISM": 0.0, "STIMULATION": 0.0, "SELF_DIRECTION": 0.0, 
      "UNIVERSALISM": 0.0, "BENEVOLENCE": 0.0, "TRADITION": 0.0, "CONFORMITY": 0.0, "SECURITY": 0.0
    }}
    """
    try:
        resp = client.chat.completions.create(model="qwen3", messages=[{"role": "user", "content": prompt}])
        return extract_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error getting vector for {user_id}: {e}")
        return None

def main():
    print(f"Starting enrichment from {INPUT_FILE} to {OUTPUT_FILE}...")
    
    # Keep track of existing users so we can skip them
    existing_users = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "user_id" in data:
                        existing_users.add(data["user_id"])
                except:
                    pass
        print(f"Found {len(existing_users)} already processed users in {OUTPUT_FILE}. Skipping them.")
    else:
        print(f"No existing output file found at {OUTPUT_FILE}. Creating new.")
        
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as fin:
            all_lines = fin.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Make sure you generated it first.")
        return

    # Process each user
    count = 0
    skipped = 0
    # Open in 'append' mode so we don't overwrite previous runs
    with open(OUTPUT_FILE, "a+", encoding="utf-8") as fout:
        for i, line in enumerate(all_lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            user_id = data.get('user_id')
            if not user_id:
                continue
                
            if user_id in existing_users:
                skipped += 1
                continue
                
            print(f"[{i+1}/{len(all_lines)}] Processing User: {user_id}...")

            contextual_posts = {item['subreddit']: item['posts'] for item in data.get('history', [])}
            vector = get_schwartz_vector_chameleon(user_id, contextual_posts)

            if vector:
                instance = {
                    "user_id": user_id,
                    "target_vector": vector         # For Evaluation
                }
                
                # Copy history and any other attributes except 'user_id' over
                for key, val in data.items():
                    if key != 'user_id':
                        instance[key] = val
                        
                fout.write(json.dumps(instance) + "\n")
                fout.flush() # Force write to file continuously so Ctrl+C is safe
                count += 1
                
                print(f"  -> Successfully generated vector for {user_id}")
            else:
                print(f"  -> Failed to generate vector for {user_id}")
                
    print(f"\nDone! Processed {count} new users and skipped {skipped} existing users.")

if __name__ == "__main__":
    main()
