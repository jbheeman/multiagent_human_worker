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



UCSD_PERSONA_PROMPT = """
You are an expert Consumer Psychologist. Your goal is to infer a user's detailed **Shopping Persona** using ONLY the provided user history.

The user history contains:
- Product metadata (title/category/brand/price if present)
- The user's star rating
- A short excerpt from the user's review text (written by the user)

You must use **Deductive Reasoning** to infer stable preferences and decision patterns. Use what is present and what is consistently absent, but do NOT make sweeping claims unless there is evidence in the provided text.

CRITICAL GROUNDING RULES:
- Every trait you output MUST be supported by 1–2 **verbatim** evidence quotes copied from the provided review excerpts.
- Evidence quotes MUST be exact spans from the review excerpts (no paraphrasing, no invented quotes).
- If you cannot find evidence for a trait, do not include that trait.

STYLE RULES FOR THE FINAL PARAGRAPH:
- Write 3–6 sentences.
- Do NOT mention specific product names, ASINs, or brands in the paragraph.
- Do NOT say "insufficient data." Prefer cautious but complete inference (e.g., “They often…” “They tend to…”).
- The paragraph should read like a clean persona summary, not bullet points.

=== EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===

USER HISTORY:
(0) Product: Natural Sea Mist Texturizing Spray | Category: Beauty | Brand: (unknown) | Price: (unknown)
    Rating: 5.0
    Review excerpt: "Such a lovely scent but not overpowering. This spray is really nice... I am comparing to other brands with yucky chemicals so I'm gonna stick with this."

(1) Product: Under-Eye Mask Patches | Category: Beauty | Brand: (unknown) | Price: 7.98
    Rating: 4.0
    Review excerpt: "These are terrific!... They feel soothing but I wish they stayed on a bit better. Still a good value."

PSYCHOLOGICAL ANALYSIS (Internal Reasoning):
- The user is scent-sensitive but prefers pleasant fragrance: evidence: "lovely scent but not overpowering."
- They care about ingredient/chemical concerns: evidence: "yucky chemicals."
- They evaluate tradeoffs (good overall but notes a flaw): evidence: "I wish they stayed on a bit better."
- They are value-aware: evidence: "Still a good value."

OUTPUT (JSON ONLY):
{{
  "traits": [
    {{
      "trait": "Prefers pleasant fragrance that is not overpowering",
      "confidence": 0.72,
      "evidence_quotes": [
        {{"quote": "lovely scent but not overpowering", "source_index": 0}}
      ]
    }},
    {{
      "trait": "Avoids products perceived as having harsh or undesirable chemicals",
      "confidence": 0.70,
      "evidence_quotes": [
        {{"quote": "yucky chemicals", "source_index": 0}}
      ]
    }},
    {{
      "trait": "Makes balanced decisions by noting both positives and drawbacks",
      "confidence": 0.66,
      "evidence_quotes": [
        {{"quote": "I wish they stayed on a bit better", "source_index": 1}}
      ]
    }},
    {{
      "trait": "Value-conscious: weighs price against performance",
      "confidence": 0.58,
      "evidence_quotes": [
        {{"quote": "Still a good value", "source_index": 1}}
      ]
    }}
  ],
  "persona_description": "[Participant] tends to evaluate products by balancing immediate sensory experience and practical performance, and they often articulate clear tradeoffs rather than giving purely one-sided feedback. They show sensitivity to ingredients and prefer options that feel cleaner or less harsh, even if that means tolerating minor downsides. They also pay attention to value, weighing whether a product's benefits justify its cost. Overall, they come across as a thoughtful, criteria-driven shopper who relies on personal experience and specific preferences when deciding what to keep using."
}}
=== END EXAMPLE ===

YOUR TASK:
Analyze the following USER HISTORY for the current user.

USER HISTORY:
{history_str}

OUTPUT REQUIREMENTS:
- Return JSON ONLY (no markdown, no extra commentary).
- Include 4–8 traits. “Each trait must have 1–2 evidence_quotes. Do not include more than 2.”
- Each evidence quote must be copied verbatim from the corresponding history review excerpt including punctuation.
- Use "source_index" to refer to which history entry (0-based index) the quote came from.
- Then provide "persona_description" as a clean 3–6 sentence paragraph with no specific product/brand mentions.
"""
@dataclass
class PersonaDataInst:
    user_id: str
    history: list[dict[str, Any]]  # rating, review_excerpt, product{title}
    heldout: dict[str, Any]         #same

def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            data = json.loads(row)
            examples.append(PersonaDataInst(user_id=data["user_id"], history=data["history"], heldout=data["heldout"]))
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

    trainset = load_persona_dataset("test.jsonl")
    total_users = len(trainset)
    print(f"Loaded {total_users} users from test.jsonl")
    
    # Load existing user_ids from output file to skip already processed users
    existing_user_ids = set()
    output_file = "util_score_model/test_personas.jsonl"
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
        for i in range(len(trainset)):
            user_id = trainset[i].user_id
            
            # Skip if already processed
            if user_id in existing_user_ids:
                skipped += 1
                print(f"[{i+1}/{total_users}] Skipping user: {user_id} (already processed)")
                continue
            
            try:
                print(f"[{i+1}/{total_users}] Processing user: {user_id}...", end=" ", flush=True)
                
                product_list_str = _build_product_list_str(trainset[i].history)
                prompt = UCSD_PERSONA_PROMPT.format(history_str=product_list_str)
                response_message = call_persona_model(prompt)
                traits, persona_description = parse_persona_response(response_message)
                
                if not persona_description:
                    print(f"WARNING: Empty persona description for user {user_id}")
                
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
