import os
import json
import re
import time
from functools import wraps
from smolagents.models import OpenAIServerModel

# --- 1. CONFIGURATION ---
OUTPUT_FILE = "amazon_ready_for_pipeline.jsonl"
NUM_SAMPLES = 200

# --- 2. PROMPT & MODEL SETUP ---
SCHWARTZ_GENERATION_PROMPT = """
You are an expert Psychological Profiler. Read the following behavioral profile of a retail shopper. 
Based on their described behaviors and priorities, score them on the 10 basic Schwartz Personal Values on a scale from 0.0 to 1.0.

Output ONLY a valid JSON block, nothing else.

=== SHOPPER PROFILE ===
{shopper_paragraph}

=== EXPECTED JSON FORMAT ===
```json
{{
  "Self-Direction": 0.8,
  "Stimulation": 0.4,
  "Hedonism": 0.3,
  "Achievement": 0.6,
  "Power": 0.2,
  "Security": 0.9,
  "Conformity": 0.7,
  "Tradition": 0.5,
  "Benevolence": 0.6,
  "Universalism": 0.5
}}
```
"""

# Same model configuration from your pipeline.py
persona_model = OpenAIServerModel(
    model_id="gpt-oss",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.getenv("NAUT_API_KEY"),
    max_tokens=1024,
    client_kwargs={"timeout": 120.0}
)

# --- 3. HELPER FUNCTIONS ---
def retry_with_backoff(max_retries=3, initial_delay=2.0):
    """Simple retry decorator so the script doesn't crash if the API blips on item #199."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"    [API Error] Retrying in {delay}s... ({e})")
                    time.sleep(delay)
                    delay *= 2
            return None
        return wrapper
    return decorator

def extract_schwartz_json(persona_text: str) -> dict:
    """Safely extracts the JSON block from the LLM output."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback for bare JSON
    match = re.search(r'\{[^{}]*"[A-Za-z\-]+":\s*[\d.]+.*?\}', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

@retry_with_backoff(max_retries=3)
def generate_schwartz_values(shopper_paragraph: str) -> dict:
    """Calls the model and parses the Schwartz values."""
    prompt = SCHWARTZ_GENERATION_PROMPT.format(shopper_paragraph=shopper_paragraph)
    response_message = persona_model([{"role": "user", "content": prompt}])
    
    # Handle smolagents response object correctly
    content = getattr(response_message, "content", str(response_message))
    return extract_schwartz_json(content)

import random

def load_personas(path: str) -> list:
    """
    Load persona records from JSON array, JSONL, or concatenated JSON objects.
    Per-line json.loads fails when a line contains multiple objects ('Extra data').
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.strip()
    if not raw:
        return []
    decoder = json.JSONDecoder()
    # Single value: array, object, or scalar
    try:
        val, end = decoder.raw_decode(raw)
        if end == len(raw):
            if isinstance(val, list):
                return val
            return [val]
    except json.JSONDecodeError:
        pass
    # Multiple concatenated objects (JSONL, or comma-separated objects without [...])
    out = []
    i = 0
    n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(raw, i)
        out.append(obj)
        i = end
    return out

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    input_file_path = "/home/pgen/personagen/multiagent_human_worker/reddit/amazonm2personas/rawpersonas.json"
    print(f"Loading data from local file: {input_file_path}...")
    all_personas = load_personas(input_file_path)

    print(f"Successfully loaded {len(all_personas)} total rows.")

    # 2. Randomly sample 200 rows (ensures we don't try to sample more than exist)
    random.seed(42) # Keeps the same 200 if you need to restart the script
    sampled_rows = random.sample(all_personas, min(NUM_SAMPLES, len(all_personas)))
    
    print(f"Sampled {len(sampled_rows)} rows. Starting generation loop...\n")

    successful = 0
    failed = 0

    # 3. Open output file and process the sampled rows
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i, row in enumerate(sampled_rows):
            # Extract data safely 
            session_id = row.get("session_id", f"unknown_{i}")
            persona_paragraph = row.get("persona", "")

            if not persona_paragraph:
                print(f"[{i+1}/{NUM_SAMPLES}] Skipping session {session_id}: No persona text found.")
                continue

            print(f"[{i+1}/{NUM_SAMPLES}] Generating Schwartz values for User ID: amazon_{session_id}...", flush=True)
            
            try:
                # Call the LLM (make sure `generate_schwartz_values` is defined above this block)
                schwartz_values = generate_schwartz_values(persona_paragraph)
                
                # If the LLM failed to output valid JSON, give it a default fallback
                if not schwartz_values:
                    print("    WARNING: Failed to parse JSON, using default neutral values.")
                    schwartz_values = {"Self-Direction": 0.5, "Stimulation": 0.5, "Security": 0.5}

                # Format exactly like your existing pipeline expects
                formatted_record = {
                    "user_id": f"amazon_{session_id}",
                    "persona": {
                        "core_identity": {
                            "worldview_and_personality": persona_paragraph
                        },
                        "schwartz_values": schwartz_values
                    }
                }
                
                # Write and flush to disk
                f_out.write(json.dumps(formatted_record, ensure_ascii=False) + "\n")
                f_out.flush()
                
                successful += 1
                
            except Exception as e:
                print(f"    ✗ Failed on session {session_id}: {e}")
                failed += 1

    print("\n" + "="*50)
    print(f"Generation Complete!")
    print(f"Successfully generated: {successful}/{len(sampled_rows)}")
    print(f"Failed: {failed}/{len(sampled_rows)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("="*50)