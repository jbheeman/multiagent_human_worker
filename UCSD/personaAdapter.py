# https://github.com/gepa-ai/gepa

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
import json
import re
from gepa.core.adapter import GEPAAdapter, EvaluationBatch
import gepa

from smolagents.models import OpenAIServerModel
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import textwrap

import time
from functools import wraps

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


class GEPACompatibleModel:
    """Wrapper to make OpenAIServerModel compatible with GEPA's string-based calls."""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
    
    def __call__(self, prompt: str | list, **kwargs):
        """Handle both string prompts (from GEPA) and message lists (normal usage)."""
        if isinstance(prompt, str):
            # Convert string to message format for GEPA compatibility
            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages, **kwargs)
            # Return just the content as a string for GEPA
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        else:
            # Normal message list format
            return self.model(prompt, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Delegate to wrapped model's generate method."""
        return self.model.generate(*args, **kwargs)



eval_model = SentenceTransformer("all-mpnet-base-v2")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

#This model generates the persona description
persona_model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

#This model evaluates the persona description   
teacher_model_raw= OpenAIServerModel( # Still used for persona agent
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

teacher_model = GEPACompatibleModel(teacher_model_raw)

@dataclass
class PersonaDataInst:
    user_id: str
    history: list[dict[str, Any]]  # rating, review_excerpt, product{title}
    heldout: dict[str, Any]         #same

@dataclass
class PersonaTrajectory:
    user_id: str
    history_str: str #reviews + titles + ratings
    heldout_item_str: str
    traits_json: dict[str, Any]
    grounding_score: float
    alignment_score: float
    total_score: float
    # you can add extra fields if you like, e.g. error messages



RolloutOutput = TypeVar("RolloutOutput") # the generated persona description
Trajectory = PersonaTrajectory
DataInst = PersonaDataInst
Candidate = dict[str, str]
EvaluatorFn = Callable[[list[DataInst], Candidate], tuple[list[RolloutOutput], list[float]]] # the evaluator function

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
  "persona_description": "Description paragraph here"
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







# === EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===

# USER HISTORY:
# (0) Product: Natural Sea Mist Texturizing Spray | Category: Beauty | Brand: (unknown) | Price: (unknown)
#     Rating: 5.0
#     Review excerpt: "Such a lovely scent but not overpowering. This spray is really nice... I am comparing to other brands with yucky chemicals so I'm gonna stick with this."

# (1) Product: Under-Eye Mask Patches | Category: Beauty | Brand: (unknown) | Price: 7.98
#     Rating: 4.0
#     Review excerpt: "These are terrific!... They feel soothing but I wish they stayed on a bit better. Still a good value."

# PSYCHOLOGICAL ANALYSIS (Internal Reasoning):
# - The user is scent-sensitive but prefers pleasant fragrance: evidence: "lovely scent but not overpowering."
# - They care about ingredient/chemical concerns: evidence: "yucky chemicals."
# - They evaluate tradeoffs (good overall but notes a flaw): evidence: "I wish they stayed on a bit better."
# - They are value-aware: evidence: "Still a good value."

# OUTPUT (JSON ONLY):
# {{
#   "traits": [
#     {{
#       "trait": "Prefers pleasant fragrance that is not overpowering",
#       "confidence": 0.72,
#       "evidence_quotes": [
#         {{"quote": "lovely scent but not overpowering", "source_index": 0}}
#       ]
#     }},
#     {{
#       "trait": "Avoids products perceived as having harsh or undesirable chemicals",
#       "confidence": 0.70,
#       "evidence_quotes": [
#         {{"quote": "yucky chemicals", "source_index": 0}}
#       ]
#     }},
#     {{
#       "trait": "Makes balanced decisions by noting both positives and drawbacks",
#       "confidence": 0.66,
#       "evidence_quotes": [
#         {{"quote": "I wish they stayed on a bit better", "source_index": 1}}
#       ]
#     }},
#     {{
#       "trait": "Value-conscious: weighs price against performance",
#       "confidence": 0.58,
#       "evidence_quotes": [
#         {{"quote": "Still a good value", "source_index": 1}}
#       ]
#     }}
#   ],
#   "persona_description": "[Participant] tends to evaluate products by balancing immediate sensory experience and practical performance, and they often articulate clear tradeoffs rather than giving purely one-sided feedback. They show sensitivity to ingredients and prefer options that feel cleaner or less harsh, even if that means tolerating minor downsides. They also pay attention to value, weighing whether a product's benefits justify its cost. Overall, they come across as a thoughtful, criteria-driven shopper who relies on personal experience and specific preferences when deciding what to keep using."
# }}
# === END EXAMPLE ===

# BASE_PROMPT_STRING = BASE_PROMPT_STRING.format(product_list_str=product_list_str)



# base_candidate = {
#     "persona_prompt": "Based on the items "  # your hand-written persona prompt
# }


def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            data = json.loads(row)
            examples.append(PersonaDataInst(user_id=data["user_id"], history=data["history"], heldout=data["heldout"]))
    return examples


class ProposalFn(Protocol):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        - Given the current `candidate`, a reflective dataset (as returned by
          `GEPAAdapter.make_reflective_dataset`), and a list of component names to update,
          return a mapping component_name -> new component text (str). This allows the user
          to implement their own instruction proposal logic. For example, the user can use
          a different LLM, implement DSPy signatures, etc. Another example can be situations
          where 2 or more components need to be updated together (coupled updates).

        Returns
        - Dict[str, str] mapping component names to newly proposed component texts.
        """
        ...

def _normalize(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, truncate."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
class PersonaGEPAAdapter(GEPAAdapter[PersonaDataInst, PersonaTrajectory, str]):
    def _build_product_list_str(self, history: list[dict[str, Any]]) -> str:
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
    
    def _build_heldout_str(self, heldout: dict[str, Any]) -> str:
        """
        Build a string for the heldout item.
        
        Args:
            heldout: Dict with 'rating', 'review_excerpt', 'product'
        """
        title = heldout['product']['title']
        rating = heldout.get('rating', 'N/A')
        review = heldout.get('review_excerpt', '').strip()
        
        if review:
            return f"- {title} | Rating: {rating}/5 | Review: {review}"
        else:
            return f"- {title} | Rating: {rating}/5 | Review: (No review provided)"
    
    def _grounding_score(self, output_json: dict[str, Any], history_excerpts: list[str]) -> float:
        traits = output_json.get("traits", []) or []
        if not traits:
            return 0.0

        norm_excerpts = [_normalize(x) for x in history_excerpts]
        total_quotes = 0
        valid_quotes = 0
        traits_with_valid = 0

        for t in traits:
            evs = t.get("evidence_quotes", []) or []
            trait_has_valid = False

            for ev in evs:
                q = _normalize(ev.get("quote", ""))
                idx = ev.get("source_index", None)

                if len(q) < 12:
                    continue

                total_quotes += 1

                if not (isinstance(idx, int) and 0 <= idx < len(norm_excerpts)):
                    continue  # invalid attribution => invalid quote

                if q in norm_excerpts[idx]:
                    valid_quotes += 1
                    trait_has_valid = True

            if trait_has_valid:
                traits_with_valid += 1

        if total_quotes == 0:
            return 0.0

        valid_quote_ratio = valid_quotes / total_quotes
        trait_coverage = traits_with_valid / max(len(traits), 1)
        return 0.5 * valid_quote_ratio + 0.5 * trait_coverage





            


        
    def _utility_score(self, generated: str, heldout: dict[str, Any]) -> float:
        """
        Score the utility score using LLM judge with PURCHASE-ALIGNED criteria.

        judge LLM gets the generated persona description and the heldout item (title, review excerpt, rating).

        """

        persona_description = generated
        heldout_product_str = self._build_heldout_str(heldout)
        # print("Heldout product str:", heldout_product_str)
        judge_prompt = """
        You are a strict evaluator that predicts how a specific user would rate a product.
        You must follow the instructions exactly and output valid JSON only.
        You will be given:
        (A) a persona description grounded in a user’s prior reviews
        (B) a heldout product described only by metadata (title/category/brand/price).

        Your task: Predict the user’s star rating for the heldout product on the 1–5 scale.

        CRITICAL RULES:
        - Use ONLY the persona description and the heldout product metadata. Do not use external knowledge.
        - Do NOT assume you have seen the heldout review text (you do not).
        - If information is insufficient, choose the most likely rating based on similar preferences implied by the persona, but do not invent new traits.
        - Prefer ordinal consistency: if the persona suggests strong fit, lean 4–5; if mismatch, lean 1–2; if unclear, lean 3–4 depending on cautiousness/value-sensitivity traits.
        - Output JSON ONLY in the exact schema below. No extra keys.

        SCHEMA:
        {{
        "predicted_rating": 1-5 integer,
        "confidence": 0.0-1.0,
        "reasoning": "2-4 sentences explaining the choice without referencing any external info."
        }}

        PERSONA_DESCRIPTION:
        {persona_description}

        HELDOUT_PRODUCT:
        {heldout_product_str}
        """
        judge_prompt = judge_prompt.format(persona_description=persona_description, heldout_product_str=heldout_product_str)

        raw_output = teacher_model(judge_prompt, temperature=0.0)

        print("Judge raw output:", raw_output)




        return 100.00 # TODO: Implement the scoring logic
    

    def paragraph_to_trait_alignment_score(self, persona_description: str, traits: list[dict[str, Any]]) -> float:
        """
        Score how well the persona description aligns with the extracted traits.
       
        """
        judge_prompt = """
        You are a strict evaluator. Output valid JSON only. No extra text.
        You will be given:
        (A) a list of allowed traits
        (B) a persona_description paragraph

        Your tasks:
        1) Trait Coverage: For each allowed trait, decide if the paragraph clearly expresses its meaning.
        2) No New Traits: Decide if the paragraph introduces any additional stable traits/behaviors/preferences that are NOT in the allowed list.

        Rules:
        - Paraphrases count as covered.
        - If a trait is only weakly implied or vague, mark NOT_COVERED.
        - "New traits" means any new stable preference/behavior claim (e.g., "shops offline", "avoids sponsored products", "only buys premium brands") not present in the allowed traits.
        - Ignore harmless filler words that do not add a stable behavioral claim (e.g., "thoughtful shopper") unless they imply a new behavior.
        - Output JSON ONLY with the schema below. No extra keys.

        Schema:
        {{
        "per_trait": [
            {{
            "trait": "<exact trait string>",
            "verdict": "COVERED" | "NOT_COVERED",
            "support": "<short phrase from the paragraph or 'NONE'>"
            }}
        ],
        "coverage_ratio": <float 0.0-1.0>,
        "has_new_traits": true | false,
        "new_traits": ["...","..."],
        "faithfulness_score": <float 0.0-1.0>
        }}

        Scoring instructions for faithfulness_score:
        - If has_new_traits=false, set faithfulness_score=1.0
        - Else set faithfulness_score=max(0.0, 1.0 - 0.25 * len(new_traits))

        ALLOWED_TRAITS (JSON array of strings):
        {traits_list_json}

        PERSONA_DESCRIPTION:
        {persona_description}
        """

        traits_list = [t.get("trait", "") for t in traits]
        traits_list_json = json.dumps(traits_list, indent=2)
        judge_prompt = judge_prompt.format(traits_list_json=traits_list_json, persona_description=persona_description)
        raw_output = teacher_model(judge_prompt, temperature=0.0)
        print("Paragraph-to-trait alignment judge output:", raw_output)
        #get faithfulness_score from the output JSON
        try:
            output_json = json.loads(raw_output)
            faithfulness_score = output_json.get("faithfulness_score", 0.0)
            coverage_ratio = output_json.get("coverage_ratio", 0.0)
            print("Faithfulness score:", faithfulness_score)
            print("Coverage ratio:", coverage_ratio)
            return faithfulness_score * 0.7 + coverage_ratio * 0.3
        except json.JSONDecodeError:
            print(f"Error parsing JSON from paragraph-to-trait alignment judge: {raw_output}")
            return 0.0





       
           
    def _score_persona(self, persona_description: str, traits: list[dict[str, Any]], history_excerpts: list[str], heldout: dict[str, Any]) -> float:
        """
        G = GroundingScore(traits, history_excerpts) + U = UtilityScore(persona_description, heldout)
        """
        grounding_score = self._grounding_score({"traits": traits}, history_excerpts)
        alignment_score = self.paragraph_to_trait_alignment_score(persona_description, traits)

        return grounding_score * 0.5 + alignment_score * 0.5
        #skipping utility score for now



    def parse_persona_response(self, response_content: str) -> tuple[list[dict[str, Any]], str]:
        """
        Parse the LLM response to extract JSON traits and persona description.
        
        Args:
            response_content: Raw response string from the LLM (may include markdown code blocks)
        
        Returns:
            Tuple of (traits_list, persona_description_str)
        """
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
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



    def load_persona_dataset(self, path: str) -> list[PersonaDataInst]:
       with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            data = json.loads(row)
            examples.append(PersonaDataInst(
                user_id=data["user_id"], 
                history=data["history"], 
                heldout=data["heldout"]))
        return examples

   
        
    
    def evaluate(
        self,
        batch: list[PersonaDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PersonaTrajectory, str]:
        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[PersonaTrajectory] | None = [] if capture_traces else None

        # use candidate["persona_prompt"], not hard-coded BASE_PROMPT_STRING
        prompt_template = candidate["persona_prompt"]
        for data_inst in batch:
            traits = []
            persona_text = ""
            score = 0.0
            grounding_score = 0.0
            alignment_score = 0.0
            
            try:
                history_str = self._build_product_list_str(data_inst.history)
                prompt = prompt_template.format(history_str=history_str)
                @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
                def call_persona_model():
                    response_message = persona_model([{"role": "user", "content": prompt}])
                    return response_message.content or ""
                raw_output = call_persona_model()
                traits, persona_text = self.parse_persona_response(raw_output)
                grounding_score = self._grounding_score({"traits": traits}, [h.get("review_excerpt", "") for h in data_inst.history])
                alignment_score = self.paragraph_to_trait_alignment_score(persona_text, traits)
                score = grounding_score * 0.5 + alignment_score * 0.5            
            except Exception as e:
                print(f"Error generating persona: {e}")
                persona_text = ""
                score = 0.0
                grounding_score = 0.0
                alignment_score = 0.0

            outputs.append(persona_text)
            scores.append(score)
            if capture_traces:
                history_str = self._build_product_list_str(data_inst.history)
                heldout_str = self._build_heldout_str(data_inst.heldout)
                trajectories.append(
                    PersonaTrajectory(
                        user_id=data_inst.user_id,
                        history_str=history_str,
                        heldout_item_str=heldout_str,
                        traits_json={"traits": traits, "persona_description": persona_text},
                        grounding_score=grounding_score,
                        alignment_score=alignment_score,
                        total_score=score,
                    )
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        

#     def make_reflective_dataset(
#         self,
#         candidate: dict[str, str],
#         eval_batch: EvaluationBatch[PersonaTrajectory, str],
#         components_to_update: list[str],
#     ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
#         datasets: dict[str, list[dict[str, Any]]] = {}

#         if "persona_prompt" not in components_to_update:
#             return datasets

#         trajectories = eval_batch.trajectories or []
#         records: list[dict[str, Any]] = []

#         # Sort by score (lowest score = needs most improvement)
#         sorted_trajs = sorted(trajectories, key=lambda t: t.score)
#         selected_trajs = sorted_trajs[:8] 

#         print(f"Generating dynamic critiques for {len(selected_trajs)} trajectories...")

#         for traj in selected_trajs:
#             # Filter the gold persona to only include purchase-inferable traits
#             filtered_gold = filter_gold_persona_for_purchases(traj.gold_persona)
            
#             if PURCHASE_ONLY_MODE:
#                 # PURCHASE-ALIGNED CRITIQUE PROMPT
#                 critique_prompt = f"""
#                 I am optimizing an AI to generate "Shopper Psychographics" from PURCHASE DATA ONLY.
                
#                 CRITICAL CONTEXT: The AI only sees what was PURCHASED. It has NO access to:
#                 - What items were viewed/clicked but not bought
#                 - How many items were compared
#                 - Whether the user read reviews
#                 - Time spent browsing
                
#                 User Purchases (THE ONLY DATA AVAILABLE):
#                 {traj.purchases_str}

#                 Purchase-Inferable Reference Traits:
#                 {filtered_gold}

#                 Generated Output:
#                 {traj.generated_persona}

#                 === CRITIQUE TASK (PURCHASE-ONLY MODE) ===
#                 Evaluate ONLY on traits that CAN be inferred from purchases:
                
#                 1. ABSTRACTION CHECK: Does it avoid listing specific products?
#                    - BAD: "Bought NADALY vacuum"
#                    - GOOD: "Prioritizes home automation"
                
#                 2. CATEGORY INFERENCE: Does it correctly identify category preferences from what WAS and WASN'T purchased?
#                    - If only electronics purchased: "Likely shops offline for groceries/clothes"
                
#                 3. BRAND TIER ANALYSIS: Does it infer traits from brand choices?
#                    - Niche brands → "Likely researches before buying"
#                    - Premium brands → "Values quality over price"
#                    - Budget brands → "Price-conscious"
                
#                 DO NOT CRITICIZE the model for failing to mention:
#                 - Review reading behavior (not observable from purchases)
#                 - Browsing patterns (not observable from purchases)
#                 - Sponsored product awareness (not observable from purchases)
#                 - Decision-making process details (not observable from purchases)
                
#                 Provide ACTIONABLE feedback focused on PURCHASE-INFERABLE improvements only.
#                 """
#             else:
#                 # Original critique prompt for full-data mode
#                 critique_prompt = f"""
#                 I am optimizing an AI to generate "Shopper Psychographics" from transaction logs.
#                 Compare the Generated Persona with the Gold Truth.

#                 User Purchases:
#                 {traj.purchases_str}

#                 Gold Truth (Psychology):
#                 {traj.gold_persona}

#                 Generated Output:
#                 {traj.generated_persona}

#                 === CRITIQUE TASK ===
#                 Analyze the gap between the Generated Output and the Gold Truth.
                
#                 1. ABSTRACTION CHECK: Does the output just list items or infer lifestyle?
#                 2. INFERENCE QUALITY: Did the model miss "Inference by Omission"?
#                 3. BRAND ANALYSIS: Does it correctly infer traits from Brand Choices?
                
#                 Provide specific, constructive feedback.
#                 """
                        
#             try:
#                 specific_critique = teacher_model(critique_prompt)
#             except Exception as e:
#                 print(f"Critique generation failed: {e}")
#                 specific_critique = "Improve alignment with purchase-inferable traits."

#             feedback = (
#                 f"Score: {traj.score:.3f}. "
#                 f"Critique: {specific_critique} "
#                 "Focus on improving purchase-inferable traits only."
#             )

#             rec = {
#                 "Inputs": {
#                     "purchases": traj.purchases_str,
#                     # Use filtered gold persona so GEPA doesn't chase impossible targets
#                     "gold_persona": filtered_gold if PURCHASE_ONLY_MODE else traj.gold_persona,
#                 },
#                 "Generated Outputs": traj.generated_persona,
#                 "Feedback": feedback,
#                 "score": traj.score,
#                 "user_id": traj.user_id,
#             }
#             records.append(rec)

#         datasets["persona_prompt"] = records
#         return datasets

#     propose_new_texts: ProposalFn | None = None


# def custom_proposal_function(
#     candidate: dict[str, str],
#     reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
#     components_to_update: list[str],
# ) -> dict[str, str]:
    
#     current_prompt = candidate["persona_prompt"]
#     failures = reflective_dataset.get("persona_prompt", [])
    
#     examples_str = ""
#     for i, fail in enumerate(failures):
#         examples_str += f"\n--- Example {i+1} ---\n"
#         examples_str += f"User Input (Purchases):\n{fail['Inputs']['purchases']}\n"
#         examples_str += f"Current AI Output:\n{fail['Generated Outputs']}\n"
#         examples_str += f"CRITIQUE (What went wrong):\n{fail['Feedback']}\n"

#     if PURCHASE_ONLY_MODE:
#         # PURCHASE-ALIGNED META-PROMPT
#         meta_prompt_core = """You are an expert Prompt Engineer for an e-commerce AI system.
# Your goal is to optimize a "System Instruction" that converts a user's *PURCHASE HISTORY ONLY* into a "Persona Description."

# CRITICAL CONSTRAINT: The AI ONLY has access to PURCHASED items. It has NO access to:
# - Viewed/clicked items that weren't purchased
# - Browsing time or session patterns  
# - Review reading behavior
# - Sponsored product awareness

# The prompt MUST NOT instruct the model to:
# - Claim how the user researches (e.g., "reads reviews thoroughly") - this cannot be observed
# - Reference "viewing patterns" or "click behavior" - this data doesn't exist
# - Make claims about decision-making process - only outcomes are visible

# The prompt SHOULD instruct the model to:
# 1. INFER FROM PURCHASES: What categories do they buy? What brand tiers?
# 2. INFER FROM OMISSIONS: What categories are ABSENT? (e.g., no food = shops offline for groceries)
# 3. INFER FROM BRAND CHOICES: Niche brands suggest research; premium brands suggest quality focus
# 4. SYNTHESIZE ABSTRACTLY: "Home automation enthusiast" not "bought vacuum and smart speaker"

# I will show you:
# 1. The CURRENT PROMPT being used.
# 2. FAILURE CASES with critiques focused on PURCHASE-INFERABLE improvements only.

# Your Task: Rewrite the prompt to better extract purchase-inferable insights while AVOIDING any instructions that assume click/view/browsing data exists.
# """
#     else:
#         meta_prompt_core = """You are an expert Prompt Engineer for an e-commerce AI system.
# Your goal is to optimize a "System Instruction" that converts a user's *Purchase History* into a specific "Persona Description."

# I will show you:
# 1. The CURRENT PROMPT being used.
# 2. A list of FAILURE CASES (User History -> Generated Persona -> Critique).

# Your Task:
# Analyze the Critical Feedback. Identify patterns in what the Current Prompt is missing.
# Then, write a NEW, IMPROVED PROMPT that addresses these specific weaknesses.

# CRITICAL GUIDELINES FOR THE NEW PROMPT:
# 1. BAN "ITEM LISTING": Forbid listing specific purchases.
# 2. INFER FROM OMISSION: Look for what is *missing*.
# 3. INFER FROM CONSISTENCY: Analyze Brand/Price Tiers.
# 4. SYNTHESIZE CATEGORIES: Abstract from items to patterns.
# """

#     meta_prompt_input = textwrap.dedent(f"""
#     {meta_prompt_core}

#     === CRITICAL CONSTRAINTS ===
#     {"The new prompt MUST NOT reference 'views', 'clicks', 'browsing', or 'review reading' as these are NOT observable from purchase data." if PURCHASE_ONLY_MODE else ""}
    
#     === HINT ===
#     Add a mandatory "Deductive Analysis Step" where the model must answer:
#     "What do the Brand Choices and Missing Categories tell us about the user's strategy?"
#     before writing the final description.
    
#     === CURRENT PROMPT ===
#     {current_prompt}
    
#     === FAILURE ANALYSIS (Critiques from the Judge) ===
#     {examples_str}
    
#     === TASK ===
#     Based on the critiques above, rewrite the "CURRENT PROMPT" to fix the recurring errors.
#     {"REMEMBER: Do NOT add instructions about analyzing views/clicks/reviews - this data is NOT available." if PURCHASE_ONLY_MODE else ""}
#     Return ONLY the new prompt text, ready to be pasted into the system.
#     """)

#     new_prompt_text = teacher_model(meta_prompt_input, temperature=0.7)
    
#     return {"persona_prompt": new_prompt_text}
    
if __name__ == "__main__":
    #test loading 1 user and their purchases

    
    
    
    
    base_candidate = {
    "persona_prompt": UCSD_PERSONA_PROMPT
}


  
    trainset = load_persona_dataset("training.jsonl")
    # print(trainset[0].history[0].get("rating"))
    # print(trainset[0].history[0].get("review_excerpt"))
    # print(trainset[0].history[0].get("product").get("title"))
    # print(trainset[0].heldout)
    valset   = load_persona_dataset("validation.jsonl")
    adapter = PersonaGEPAAdapter()
    product_list_str = adapter._build_product_list_str(trainset[0].history)
    # heldout_str = adapter.build_heldout_str(trainset[0].heldout)
    # print("Heldout str:", heldout_str)
    # print(product_list_str)
    # prompt = UCSD_PERSONA_PROMPT.format(history_str=product_list_str)
    # response_message = persona_model([{"role": "user", "content": prompt}])
    # print(response_message.content)
    #parse for json
    # traits, persona_description = adapter.parse_persona_response(response_message.content)
    # print(traits)
    # print(persona_description)
    # output_json = {"traits": traits}
    # history_excerpts = [item.get('review_excerpt', '') for item in trainset[0].history]


    # grounding_score = adapter._grounding_score(output_json, history_excerpts )
    # print("grounding_score:", grounding_score)

    # alignment_score  = adapter.paragraph_to_trait_alignment_score(persona_description, traits)
    # print("alignment_score:", alignment_score)
    # utility_score = adapter._utility_score(persona_description, trainset[0].heldout)
    # print("utility_score:", utility_score)
    # score = grounding_score + utility_score
    # print(score)


    

#     adapter.propose_new_texts = custom_proposal_function

    gepa_result = gepa.optimize(
    seed_candidate=base_candidate,
    trainset=trainset,
    valset=valset,
    max_metric_calls=50, # <-- Set a budget
    reflection_lm=teacher_model, # <-- Use a strong model to reflect on mistakes and propose better prompts
    adapter=adapter,
)

    best = gepa_result.best_candidate
    print("\n=== Best persona prompt ===")
    print(best)

    # batch = trainset[:2]
    # eval_batch = adapter.evaluate(batch, base_candidate, capture_traces=True)
    # print("Outputs:", eval_batch.outputs)
    # print("Scores:", eval_batch.scores)
    # print("First trajectory:", eval_batch.trajectories[0] if eval_batch.trajectories else None)

    # print(f"Loaded {len(trainset)} examples")
    


    # {"user_id": "AE3KLVXGZPANXE5XLXYKHTVAZ3FQ", "category": "All_Beauty", "history": [{"parent_asin": "B095RWJJB8", "rating": 4.0, "timestamp_ms": 1627679830425, "review_excerpt": "This is a pretty bow however $7 for one bow is pretty expensive considering I can get 10 of these bows for $8 from other sellers.", "review_full": "This is a pretty bow however $7 for one bow is pretty expensive considering I can get 10 of these bows for $8 from other sellers.", "review_title": "Pretty but overpriced", "product": {"title": "Summer Crystal Hair Clip Sparkling Sequins, Double-Layered Alligator Clip Hair Bow Accessory For Women and Girls, Made in Korea, Daily, Party, Cosplay (Holographic)", "brand": null, "price": null, "main_category": "All Beauty"}}, {"parent_asin": "B097JXPZ6D", "rating": 4.0, "timestamp_ms": 1627938153438, "review_excerpt": "This is a cute bow and is exactly what is advertised. I do believe the $10 price point is pretty high considering you can get 10 headbands for $12. It is well made and fits my 4 year old daughter\u2019s head nicely.", "review_full": "This is a cute bow and is exactly what is advertised. I do believe the $10 price point is pretty high considering you can get 10 headbands for $12. It is well made and fits my 4 year old daughter\u2019s head nicely.", "review_title": "Pretty headband", "product": {"title": "Summer Crystal Headband for Girls, 3D Large Glitter Top Bow, Hair Accessory for Girls and Women, Various Occasions, Holidays, Parties, Daily, Cosplay, Gift (Magenta)", "brand": null, "price": null, "main_category": "All Beauty"}}, {"parent_asin": "B08Q8NQMX2", "rating": 4.0, "timestamp_ms": 1628083724757, "review_excerpt": "These are cute and my 4 year old daughter loves them. They come in bright colors however a handful do them has creases wings and I\u2019m not really sure how to get the crease out.", "review_full": "These are cute and my 4 year old daughter loves them. They come in bright colors however a handful do them has creases wings and I\u2019m not really sure how to get the crease out.", "review_title": "Cute butterfly clips but some wings are creased", "product": {"title": "DARKLATER Butterfly Hair Clips for Girls,for Toddler Girls,Baby Girls and Women,Cute Hair Clips,Beautiful Hair Accessories,12 PCS", "brand": null, "price": null, "main_category": "All Beauty"}}, {"parent_asin": "B093JGCRWX", "rating": 3.0, "timestamp_ms": 1628722253112, "review_excerpt": "If this product was indeed EWG verified, it would not only be on the website but it would have the EWG logo on the product plus it wouldn\u2019t have linalool which is high on the allergy list.<br /><br />Other than the linalool, this has decent ingredients. I would stay away from this product if you have malassezia (fungal) acne as olive and japonica may be triggers and/or pore clogging.<br /><br />Like all natural bar shampoos, it won\u2019t lather like traditional synthetic shampoos but it does clean. It takes some getting use too and a period of detoxing for your hair to get use to the change in chemicals if you are switching from synthetic to natural but it is worth it!<br /><br />I would recommend this shampoo bar however I am rather concerned about the EWG verified claim.", "review_full": "I searched the EWG website for this company and product and in many spelling varieties and came up empty handed. If this product was indeed EWG verified, it would not only be on the website but it would have the EWG logo on the product plus it wouldn\u2019t have linalool which is high on the allergy list.<br /><br />Other than the linalool, this has decent ingredients. It is silicone free, paraben free, sulfate free and alcohol free. I would stay away from this product if you have malassezia (fungal) acne as olive and japonica may be triggers and/or pore clogging.<br /><br />Like all natural bar shampoos, it won\u2019t lather like traditional synthetic shampoos but it does clean. It takes some getting use too and a period of detoxing for your hair to get use to the change in chemicals if you are switching from synthetic to natural but it is worth it!<br /><br />I would recommend this shampoo bar however I am rather concerned about the EWG verified claim.", "review_title": "Paraben free, silicone free, sulfate free but not EWG verified", "product": {"title": "The Vegan Glow Quinoa Protein Shampoo Bar | EWG Verified | Vegetable proteins from Quinoa & Soybeans", "brand": null, "price": null, "main_category": "All Beauty"}}], "heldout": {"parent_asin": "B08Z7FQGW3", "rating": 4.0, "timestamp_ms": 1629826110674, "review_excerpt": "This is a beautiful dark purple leaf crown with rose gold metal. It fits my female adult head nicely and that was after I bent it to make it smaller. It wouldn\u2019t fit a small child. My 4 year old daughter was very disappointed that it didn\u2019t fit her. It came quickly and I\u2019m surprised it was damaged due to the lack of product protection. It is well made and a fun addition to anyone's dress up collection!", "review_full": "This is a beautiful dark purple leaf crown with rose gold metal. It fits my female adult head nicely and that was after I bent it to make it smaller. It wouldn\u2019t fit a small child. My 4 year old daughter was very disappointed that it didn\u2019t fit her. It came quickly and I\u2019m surprised it was damaged due to the lack of product protection. It is well made and a fun addition to anyone's dress up collection!", "review_title": "Beautiful crown for adults", "product": {"title": "S SNUOY Purple Crystal Vintage Queen Crowns Baroque Tiaras Wedding Bridal Queen Tiaras and Crowns for Women and Girls Party Headbands", "brand": null, "price": null, "main_category": "All Beauty"}}}
