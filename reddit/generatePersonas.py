from smolagents.models import OpenAIServerModel
import os
from typing import Any
import json
import re
import time
from functools import wraps
from dataclasses import dataclass

persona_model = OpenAIServerModel(
        model_id="gpt-oss",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator to retry a function with exponential backoff on timeout or connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, Exception) as e:
                    error_str = str(e).lower()
                    is_timeout = "timeout" in error_str or "timed out" in error_str
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        raise
                    
                    if is_timeout or "connection" in error_str:
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Non-timeout error, don't retry
                        raise
            return None
        return wrapper
    return decorator


REDDIT_PROMPT = """
You are an expert Psychological Profiler.
Your goal is to write a **System Instruction** that will force an AI Agent to authentically embody a specific user.

=== INPUT DATA ===
1. DEMOGRAPHIC ANCHOR:
{anchor_demographics}

2. PSYCHOLOGICAL SHIFT (Context: r/{subreddit}):
{psych_vector_str}

3. BEHAVIORAL SAMPLES:
{history_str}

=== YOUR TASK ===
Write a cohesive, first-person **System Prompt** for an AI agent.
The prompt must:
1. Define the agent's specific demographic identity (Age, Gender, Role).
2. Explicitly encode the Psychological Values (Schwartz Vectors) as behavioral rules.
3. Synthesize the "Shift" (Values) with the "Anchor" (Identity) to resolve conflicts.

=== OUTPUT FORMAT ===
Return ONLY the System Prompt text. Start with "You are..."

"""


@dataclass
class PersonaDataInst:
    user_id: str             # "lumenation"
    subreddit: str           # "r/KotakuInAction" (The Context)
    
    # INPUTS FOR THE AGENT
    posts: list[str]         # The 5 posts from THIS subreddit only
    anchor_demographics: str # "28M, Developer, St. Louis" (extracted globally)
    shift_vector: dict       # {"POWER": 0.8, ...} (extracted locally from these posts)
    
    # GROUND TRUTH (For Evaluation)
    # We test if the agent matches THIS vector, not the global average
    target_vector: dict      # Same as shift_vector



def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            try:
                data = json.loads(row)
                examples.append(PersonaDataInst(user_id=data["user_id"], subreddit=data["subreddit"], posts=data["posts"], anchor_demographics=data["anchor_demographics"], shift_vector=data["shift_vector"], target_vector=data["target_vector"]))
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_persona_model(prompt: str) -> str:
    response_message = persona_model([{"role": "user", "content": prompt}])
    return response_message.content or ""

if __name__ == "__main__":

    #validation_personas.jsonl - write userID+generated persona as {userID: persona}

    # trainset = load_persona_dataset("train_gdelt_enriched.jsonl")
    testset = load_persona_dataset("test_reddit_enriched.jsonl")
    total_users = len(testset)
    print(f"Loaded {testset} users from test.jsonl")
    
    # Load existing user_ids from output file to skip already processed users
    output_file = "reddit_personas.jsonl"
    
    
    print(f"Starting persona generation...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    with open(output_file, "a") as f:
        for i in range(len(testset)):
            user_id = testset[i].user_id
            
          
            try:
                print(f"[{i+1}/{total_users}] Processing user: {user_id}...", end=" ", flush=True)

                anchor_demographics = testset[i].anchor_demographics
                subreddit = testset[i].subreddit
                psych_vector_str = testset[i].shift_vector
                posts = testset[i].posts
                prompt = REDDIT_PROMPT.format(history_str=posts, anchor_demographics=anchor_demographics, subreddit=subreddit, psych_vector_str=psych_vector_str)
                response_message = call_persona_model(prompt)
                persona_description = response_message

                print(f"Persona description: {persona_description}")
                if not persona_description:
                    print(f"WARNING: Empty persona description for user {user_id}")
                    continue
                print(f"--------------------------------")
                # print(f"Persona description: {persona_description}")
                # print(f"Traits: {traits}")
                # print(f"Psych vector: {psych_vector_str}")
                # print(f"Product list: {product_list_str}")
                # print(f"Prompt: {prompt}")
                # print(f"Response message: {response_message}")
                # print(f"--------------------------------")
                
                
                
                f.write(json.dumps({"user_id": user_id, "persona": persona_description}) + "\n")
                f.flush()  # Ensure data is written immediately
                
                successful += 1
                if i == 10:
                    break
                print(f"✓ Success")
                
            except Exception as e:
                failed += 1
                print(f"✗ Failed: {e}")
                # Continue to next user instead of crashing
                continue
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Successful: {successful}/{total_users}")
    print(f"Skipped: {skipped}/{total_users}")
    print(f"Failed: {failed}/{total_users}")
    print(f"{'='*50}")
