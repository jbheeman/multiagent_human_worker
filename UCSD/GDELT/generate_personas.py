from smolagents.models import OpenAIServerModel
import os
from typing import Any
import json
import re
import time
from functools import wraps
from dataclasses import dataclass
persona_model = OpenAIServerModel(
        model_id="gemma3",
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


GDELT_PERSONA_PROMPT = """
You are an expert Cognitive Profiler at the FBI. Your goal is to infer a user\'s **Decision-Making Style** and **Risk Profile** by synthesizing their history with their core psychological drivers. \n\n1. USER BEHAVIOR TRACE (Reviews & Purchases):\n{history_str}\n(Look for evidence of: Attention to detail, patience, impulse, skepticism, expertise level, reliance on brand vs. specs).\n\n2. PSYCHOLOGICAL CONTEXT (Schwartz Values):\n{psych_vector_str}\n**CRITICAL NOTE: These values represent the user\'s ABSOLUTE TRUE INTERNAL MOTIVATION. They are the non-negotiable foundation for all interpretation. When behavior contradicts the values (e.g., anxious user buys motorcycle), the values explain WHY the behavior occurs – never vice versa. High scores (top 2 values) dictate core drivers. IGNORING THIS HIERARCHY INVALIDATES THE ENTIRE ANALYSIS.**\n\n=== YOUR TASK ===\nSynthesize these inputs into a **Unified Cognitive Persona** where Psychological Context is the sole foundation for explaining behavior. You are analyzing **HOW THEY THINK** to resolve contradictions – NOT describing what they buy.\n\n**CONFLICT RESOLUTION PROTOCOL:**\n- **Hierarchy Rule (Non-Negotiable):** Schwartz Values are the CAUSE; behavior is the EFFECT. Always start with values to explain behavior. If behavior seems contradictory (e.g., "Anxious User buys motorcycle"), explicitly state: "This purchase manifests [Value] because [causal explanation revealing the value-driven purpose]."\n- **Dominant Value Priority:** Only the top 2 Schwartz Values (by score) may drive the persona. All behavior must be interpreted through these values first.\n- **Synthesis Requirement:** For every trait, the explanation MUST follow: "This [behavior] is a strategic manifestation of [Schwartz Value] because [causal explanation showing how the behavior serves the value]." Never describe behavior without anchoring to values.\n- **Evidence Mandate:** Every trait requires BOTH sources with explicit causal linkage. No trait may exist without value-behavior synthesis.\n\n=== CRITICAL GROUNDING RULES ===\n- **Source A (Behavior):** Quote verbatim text from reviews. Use correct `source_index` (0,1,2...). \n- **Source B (Psychology):** Cite Schwartz Value **with source_index: -1** (e.g., "BENEVOLENCE: 0.137").\n- **Mandatory Synthesis (25-30 words):** Every evidence pair MUST show value-first causation: \n  `[Behavior quote] is a strategic manifestation of [Value] because [explanation linking behavior to value purpose]` \n  (e.g., "I bought the motorcycle" is a strategic manifestation of POWER: 0.8 because "controlling high-risk machinery satisfies the need for dominance over unpredictable environments").\n- **Contradiction Handling Protocol:** When behavior seems value-inconsistent: \n  1. Identify the dominant value driving the action \n  2. Explain the hidden value purpose (e.g., "Gaming headset purchase demonstrates BENEVOLENCE because sharing VR experiences strengthens community bonds")\n  3. NEVER prioritize surface behavior over values\n- **NO EXCEPTIONS:** If behavior lacks direct evidence for a value, infer the value-driven strategy (e.g., "Controller charger purchase demonstrates POWER because maintaining device control enables competitive dominance").\n\n=== OUTPUT FORMAT (JSON ONLY) ===\n{{\n  "traits": [\n    {{\n      "trait": "Abstract behavioral trait (e.g., \'Control-seeking through precision tools\')",\n      "confidence": 0.8,\n      "evidence_quotes": [\n        {{"quote": "exact behavior quote", "source_index": 0}},\n        {{"quote": "Schwartz Value: [VALUE]: [SCORE]", "source_index": -1}}\n      ],\n      "synthesis_explanation": "[Behavior] is a strategic manifestation of [Value] because [25-30 word causal explanation showing how behavior serves value]"\n    }}\n  ],\n  "persona_description": "This person\'s decision-making is fundamentally driven by [Top Value 1] and [Top Value 2]. They strategically use [behavior pattern] to [achieve value-specific outcome], as seen in [causal example linking behavior to value]. This reveals [3-5 sentence summary of value-driven cognitive strategy]."\n}}'

"""


def calibrate_psych_vector(raw_vector):
        """
        Balances the Schwartz Vector by penalizing 'loud' dictionaries (General Inquirer)
        and boosting 'quiet' dictionaries (Moral Foundations).

        Args:
            raw_vector: Dict of trait scores from GDELT
        """

        # 1. Define AGGRESSIVE Multipliers to counter GDELT bias
        POPULATION_MEANS = {
        'POWER': 0.3815,
        'ACHIEVEMENT': 0.2319,
        'HEDONISM': 0.0735,
        'STIMULATION': 0.0883,
        'UNIVERSALISM': 0.0226,  # Very low baseline!
        'BENEVOLENCE': 0.0331,
        'TRADITION': 0.0387,
        'CONFORMITY': 0.0427,
        'SECURITY': 0.0875
        }
        relative_scores = {}

        for trait, score in raw_vector.items():
            # Get the average for this trait
            avg = POPULATION_MEANS.get(trait, 0.01) # Default to 0.01 to avoid div/0
            
            # 2. Calculate the Ratio (User Score / Average Score)
            # Example:
            # - User has 0.04 Universalism (Tiny number!)
            # - Average is 0.02 (Even tinier!)
            # - Ratio = 2.0 (User is TWICE as Universalist as the average person)
            ratio = score / avg
            
            relative_scores[trait] = ratio

    # 3. Re-Normalize to sum to 1.0
        total = sum(relative_scores.values())
        if total == 0: return raw_vector
        
        normalized_vector = {k: round(v/total, 3) for k, v in relative_scores.items()}
        
        return str(normalized_vector)

@dataclass
class PersonaDataInst:
    user_id: str
    history: list[dict[str, Any]]  # rating, review_excerpt, product{title}
    heldout: dict[str, Any]         #same
    schwartz_vector: dict[str, float] | None = None  # optional field for psychological vector



def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            data = json.loads(row)
            examples.append(PersonaDataInst(user_id=data["user_id"], history=data["history"], heldout=data["heldout"], schwartz_vector=data.get("psych_vector", None)))
    return examples

def _build_product_list_str(history: list[dict[str, Any]]) -> str:
        """
        Build a string of reviews, titles, and ratings.
        
        Args:
            history: List of history dicts with 'rating', 'review_excerpt', 'product'
        """
        lines = []
        for item in history:
            title = item['product']['title']
            rating = item.get('rating', 'N/A')
            review = item.get('review_excerpt', '').strip()
            
            # Format: Product Title | Rating: X/5 | Review: [text or "No review"]
            if review:
                lines.append(f"- {title} | Rating: {rating}/5 | Review: {review}")
            else:
                lines.append(f"- {title} | Rating: {rating}/5 | Review: (No review provided)")
        
        return "\n".join(lines)

def parse_persona_response(response_content: str) -> tuple[dict, str]:
        """
        Parse the LLM response to extract JSON traits and persona description.
        
        Args:
            response_content: Raw response string from the LLM (may include markdown code blocks)
        
        Returns:
            Tuple of (traits_dict, persona_description_str)
        """
        json_match = re.search(r'\s*\n(.*?)\n```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()

        else:
            json_match = re.search(r'\{.*"traits".*"persona_description".*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)

            else:
                json_str = response_content.strip()
        try:
            parsed_json = json.loads(json_str)
            traits = parsed_json.get("traits", [])
            persona_description = parsed_json.get("persona_description", "")
            return traits, persona_description
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {json_str}")
            return [], ""


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_persona_model(prompt: str) -> str:
    response_message = persona_model([{"role": "user", "content": prompt}])
    return response_message.content or ""

if __name__ == "__main__":

    #validation_personas.jsonl - write userID+generated persona as {userID: persona}

    # trainset = load_persona_dataset("train_gdelt_enriched.jsonl")
    testset = load_persona_dataset("personasforjesh.jsonl")
    total_users = len(testset)
    print(f"Loaded {testset} users from test.jsonl")
    
    # Load existing user_ids from output file to skip already processed users
    existing_user_ids = set()
    output_file = "gdelt_personas.jsonl"
    try:
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_user_ids.add(data["user_id"])
        print(f"Found {len(existing_user_ids)} already processed users. Will skip them.\n")
    except FileNotFoundError:
        print(f"Output file doesn't exist yet. Starting fresh.\n")
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}. Starting fresh.\n")
    
    print(f"Starting persona generation...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    with open(output_file, "a") as f:
        for i in range(len(testset)):
            user_id = testset[i].user_id
            
          
            try:
                print(f"[{i+1}/{total_users}] Processing user: {user_id}...", end=" ", flush=True)

                psych_vector_str = calibrate_psych_vector(testset[i].schwartz_vector)
                product_list_str = _build_product_list_str(testset[i].history)
                prompt = GDELT_PERSONA_PROMPT.format(history_str=product_list_str, psych_vector_str=psych_vector_str)
                response_message = call_persona_model(prompt)
                traits, persona_description = parse_persona_response(response_message)
                
                if not persona_description:
                    print(f"WARNING: Empty persona description for user {user_id}")
                print(f"Persona description: {persona_description}")
                print(f"Traits: {traits}")
                print(f"Psych vector: {psych_vector_str}")
                print(f"Product list: {product_list_str}")
                print(f"Prompt: {prompt}")
                print(f"Response message: {response_message}")
                print(f"--------------------------------")
                
                
                
                f.write(json.dumps({"user_id": user_id, "persona": persona_description}) + "\n")
                f.flush()  # Ensure data is written immediately
                
                successful += 1
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
