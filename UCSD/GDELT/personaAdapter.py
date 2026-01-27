# https://github.com/gepa-ai/gepa

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import random
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
import httpx

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


# Custom HTTP client that skips SSL verification (for Nautilus SSL issues)
http_client = httpx.Client(verify=False)

#This model generates the persona description
persona_model = OpenAIServerModel(
        model_id="gemma3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
        client_kwargs={"http_client": http_client}
    )

#This model evaluates the persona description
teacher_model_raw= OpenAIServerModel( # Still used for persona agent
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
        client_kwargs={"http_client": http_client}
    )

teacher_model = GEPACompatibleModel(teacher_model_raw)

@dataclass
class PersonaDataInst:
    user_id: str
    history: list[dict[str, Any]]  # rating, review_excerpt, product{title}
    heldout: dict[str, Any]         #same
    schwartz_vector: dict[str, float] | None = None  # optional field for psychological vector

@dataclass
class PersonaTrajectory:
    user_id: str
    history_str: str #reviews + titles + ratings
    heldout_item_str: str
    traits_json: dict[str, Any]
    grounding_score: float
    schwartz_alignment_score: float
    total_score: float
    raw_pvq_score: float
    schwartz_vector: dict[str, float] | None = None  # optional field for psychological vector
    # you can add extra fields if you like, e.g. error messages



RolloutOutput = TypeVar("RolloutOutput") # the generated persona description
Trajectory = PersonaTrajectory
DataInst = PersonaDataInst
Candidate = dict[str, str]
EvaluatorFn = Callable[[list[DataInst], Candidate], tuple[list[RolloutOutput], list[float]]] # the evaluator function

UCSD_PERSONA_PROMPT = """
You are an expert Cognitive Profiler at the FBI.Your goal is to infer a user's **Decision-Making Style** and **Risk Profile** by analyzing their history and current psychological state.

1. USER BEHAVIOR TRACE (Reviews & Purchases):
{history_str}
(Look for evidence of: Attention to detail, patience, impulse, skepticism, expertise level, reliance on brand vs. specs).

2. PSYCHOLOGICAL CONTEXT (Schwartz Values):
{psych_vector_str}
(Note: These values drive their current motivation. High 'Security' = Risk Aversion. High 'Stimulation' = Novelty Seeking. etc.)


=== YOUR TASK ===
Synthesize these inputs into a **Unified Cognitive Persona**.
You are NOT analyzing *what* they buy. You are analyzing **HOW THEY THINK**.

=== CRITICAL GROUNDING RULES ===
- Every trait must be supported by evidence.
- **Source A (Behavior):** Quote verbatim text from reviews. Use the correct `source_index` (0, 1, 2...).
- **Source B (Psychology):** Cite the Schwartz Value. **MUST USE `source_index: -1`** for these citations.
- **Best Practice:** Ideally, find traits that are supported by **BOTH** behavior and psychology (e.g., Trait: "Risk Averse" -> Evidence: ["returned for safety defect", "SECURITY: 0.8"]).
=== OUTPUT FORMAT (JSON ONLY) ===
{{
  "traits": [
    {{
      "trait": "Abstract behavioral trait",
      "confidence": 0.8,
      "evidence_quotes": [
        {{"quote": "exact quote or value", "source_index": 0}}
      ]
    }}
  ],
  "persona_description": "A 3-5 sentence paragraph describing this person's **General Problem-Solving Style**. Do NOT mention specific products."
}}
"""


# Full PVQ-40 Items and Scoring Key
# Source: Schwartz Portrait Values Questionnaire (PVQ-40)
# Scale: 1 (Not like me at all) to 6 (Very much like me)

PVQ_DATA = {
    "items": {
        1: "Thinking up new ideas and being creative is important to him. He likes to do things in his own original way.",
        2: "It is important to him to be rich. He wants to have a lot of money and expensive things.",
        3: "He thinks it is important that every person in the world be treated equally. He believes everyone should have equal opportunities in life.",
        4: "It’s very important to him to show his abilities. He wants people to admire what he does.",
        5: "It is important to him to live in secure surroundings. He avoids anything that might endanger his safety.",
        6: "He thinks it is important to do lots of different things in life. He always looks for new things to try.",
        7: "He believes that people should do what they’re told. He thinks people should follow rules at all times, even when no one is watching.",
        8: "It is important to him to listen to people who are different from him. Even when he disagrees with them, he still wants to understand them.",
        9: "He thinks it’s important not to ask for more than what you have. He believes that people should be satisfied with what they have.",
        10: "He seeks every chance he can to have fun. It is important to him to do things that give him pleasure.",
        11: "It is important to him to make his own decisions about what he does. He likes to be free to plan and to choose his activities for himself.",
        12: "It’s very important to him to help the people around him. He wants to care for their well-being.",
        13: "Being very successful is important to him. He likes to impress other people.",
        14: "It is very important to him that his country be safe. He thinks the state must be on watch against threats from within and without.",
        15: "He likes to take risks. He is always looking for adventures.",
        16: "It is important to him to always behave properly. He wants to avoid doing anything people would say is wrong.",
        17: "It is important to him to be in charge and tell others what to do. He wants people to do what he says.",
        18: "It is important to him to be loyal to his friends. He wants to devote himself to people close to him.",
        19: "He strongly believes that people should care for nature. Looking after the environment is important to him.",
        20: "Religious belief is important to him. He tries hard to do what his religion requires.",
        21: "It is important to him that things be organized and clean. He really does not like things to be a mess.",
        22: "He thinks it’s important to be interested in things. He likes to be curious and to try to understand all sorts of things.",
        23: "He believes all the world’s people should live in harmony. Promoting peace among all groups in the world is important to him.",
        24: "He thinks it is important to be ambitious. He wants to show how capable he is.",
        25: "He thinks it is best to do things in traditional ways. It is important to him to keep up the customs he has learned.",
        26: "Enjoying life’s pleasures is important to him. He likes to spoil himself.",
        27: "It is important to him to respond to the needs of others. He tries to support those he knows.",
        28: "He believes he should always show respect to his parents and to older people. It is important to him to be obedient.",
        29: "He wants everyone to be treated justly, even people he doesn’t know. It is important to him to protect the weak in society.",
        30: "He likes surprises. It is important to him to have an exciting life.",
        31: "He tries hard to avoid getting sick. Staying healthy is very important to him.",
        32: "Getting ahead in life is important to him. He strives to do better than others.",
        33: "Forgiving people who have hurt him is important to him. He tries to see what is good in them and not to hold a grudge.",
        34: "It is important to him to be independent. He likes to rely on himself.",
        35: "Having a stable government is important to him. He is concerned that the social order be protected.",
        36: "It is important to him to be polite to other people all the time. He tries never to disturb or irritate others.",
        37: "He really wants to enjoy life. Having a good time is very important to him.",
        38: "It is important to him to be humble and modest. He tries not to draw attention to himself.",
        39: "He always wants to be the one who makes the decisions. He likes to be the leader.",
        40: "It is important to him to adapt to nature and to fit into it. He believes that people should not change nature."
    },
    "mapping": {
        "UNIVERSALISM": [3, 8, 19, 23, 29, 40],
        "BENEVOLENCE": [12, 18, 27, 33],
        "TRADITION": [9, 20, 25, 38],
        "CONFORMITY": [7, 16, 28, 36],
        "SECURITY": [5, 14, 21, 31, 35],
        "POWER": [2, 17, 39],
        "ACHIEVEMENT": [4, 13, 24, 32],
        "HEDONISM": [10, 26, 37],
        "STIMULATION": [6, 15, 30],
        "SELF_DIRECTION": [1, 11, 22, 34] # Note: SELF_DIRECTION keys to GDELT "SELF_DIRECTION" if present, otherwise maps to Creative/Free traits
    }
}




def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            data = json.loads(row)
            examples.append(PersonaDataInst(user_id=data["user_id"], history=data["history"], heldout=data["heldout"], schwartz_vector=data.get("psych_vector", None)))
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
            # if title:
            #     lines.append(f"- {title}")
        
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
    
    def calibrate_psych_vector(self, raw_vector):
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
        
        return normalized_vector



 

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

                #score Schwartz citations as valid if source_index == -1
                if (idx == -1 and q in ["security", "power", "achievement", "hedonism", "stimulation", "universalism", "benevolence", "tradition", "conformity"]):
                    valid_quotes += 1
                    trait_has_valid = True
                    total_quotes += 1
                    continue


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



    # PVQ-40 Items and Scoring Key
# Scale: 1 (Not like me at all) to 6 (Very much like me)
    
    def _administer_pvq_test(self, persona_text: str) -> dict[str, float]:
        """
        One-Shot PVQ Administration.
        1. Adopts the Persona.
        2. Reads all 40 items.
        3. Returns a single JSON object with ratings {1: 5, 2: 1, ...}.
        4. Calculates and returns the aggregated Value Scores.
        """
        
        # --- A. Format the Survey Sheet ---
        # "1. Thinking up new ideas... \n 2. It is important..."
        survey_text = "\n".join([f"{k}. {v}" for k, v in PVQ_DATA['items'].items()])
        
        # --- B. The Prompt ---
        system_prompt = f"""
        You are participating in a Psychology Study.
        
        YOUR IDENTITY:
        "{persona_text}"
        
        INSTRUCTIONS:
        Below are descriptions of people. For each one, ask yourself: "How much like me is this person?"
        
        Use this rating scale exactly:
        1 = Not like me at all
        2 = Not like me
        3 = A little like me
        4 = Somewhat like me
        5 = Like me
        6 = Very much like me
        
        OUTPUT FORMAT:
        You must return a valid JSON object mapping the Item Number (string) to your Score (integer).
        Example: {{"1": 5, "2": 2, ... "40": 6}}
        Do not include any other text.
        """
        
        user_prompt = f"""
        Please rate these 40 items based on your persona.
        
        {survey_text}
        """

        try:
            # --- C. The Call ---
            # (Using your existing model wrapper)
            response = persona_model([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            print(f"PVQ Response: {response}")
            
            # --- D. Parsing ---
            # Robust JSON extraction
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if not match: 
                print("PVQ Failed: No JSON found in response.")
                return {}
            
            item_scores = json.loads(match.group(0))
            print(f"PVQ Item Scores: {item_scores}")
            
            # --- E. Scoring (Aggregating Items into Values) ---
            final_trait_scores = {}
            
            for trait, item_ids in PVQ_DATA['mapping'].items():
                # Gather the scores for this trait (e.g., Security = Items 5, 14, 21...)
                raw_values = []
                for i in item_ids:
                    # Handle both string "1" and int 1 keys
                    val = item_scores.get(str(i)) or item_scores.get(i)
                    if val is not None:
                        raw_values.append(float(val))
                
                # Average them to get the Trait Score (1.0 - 6.0)
                if raw_values:
                    final_trait_scores[trait] = sum(raw_values) / len(raw_values)
                else:
                    final_trait_scores[trait] = 0.0
            
            
            print(f"Final Trait Scores: {final_trait_scores}")
            return final_trait_scores

        except Exception as e:
            print(f"PVQ Critical Failure: {e}")
            return {}
           
    
    def _score_schwartz_alignment(self, persona_text: str, target_vector: dict[str, float]) -> float:
        """
        Comparing GDELT Targets (Normalized 0-1) vs PVQ Results (Scale 1-6).
        """
        if not target_vector: return 0.5, 0.0 
        
        # 1. Run the Survey
        pvq_results = self._administer_pvq_test(persona_text) # Returns {SECURITY: 5.5, POWER: 2.1...}
        if not pvq_results: return 0.0, 0.0
        
        # 2. Identify the Dominant Target Trait
        # (The one we REALLY care about for this optimization)
        sorted_traits = sorted(target_vector.items(), key=lambda x: x[1], reverse=True)
        primary_trait, primary_val = sorted_traits[0] # e.g., SECURITY        
        measured_score = pvq_results.get(primary_trait, 0)
        print(f"DEBUG: Target {primary_trait} ({primary_val}) -> PVQ Score {measured_score}")
        
        # 3. Calculate Alignment Score
        # PVQ is 1-6. We expect High GDELT (>0.2) to map to High PVQ (>4.5).
        # We expect Low GDELT (<0.1) to map to Low PVQ (<3.0).
        
        # Option A: Simple Thresholding (Robust)
        if measured_score >= 4.5: grade = 1.0
        elif measured_score >= 3.5: grade = 0.5
        else: grade = 0.0
        
        return grade, measured_score
        # Option B: Judge LLM (User's request)
        # Pass the numbers to the Teacher Model for a nuanced critique
        judge_prompt = f"""
        EVALUATION TASK:
        Target Trait: {primary_trait} (High Priority)
        
        Psychometric Test Result (PVQ-40):
        The Persona scored {measured_score:.1f} on a scale of 1.0 to 6.0 for {primary_trait}.
        
        Did the Persona Generator successfully encode the target trait?
        - Score 1.0 if score is > 4.5
        - Score 0.5 if score is 3.5 - 4.5
        - Score 0.0 if score is < 3.5
        
        Return float only.
        """
        response = teacher_model(judge_prompt)
        # ... parse float ...
        return parsed_float
    
    def _score_persona(self, persona_description: str, traits: list[dict[str, Any]], history_excerpts: list[str], heldout: dict[str, Any]) -> float:
        """
        G = GroundingScore(traits, history_excerpts) + U = UtilityScore(persona_description, heldout)
        """
        grounding_score = self._grounding_score({"traits": traits}, history_excerpts)
        schwartz_alignment_score = self._score_schwartz_alignment(persona_description, traits)

        return grounding_score * 0.5 + schwartz_alignment_score * 0.5
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
                heldout=data["heldout"],
                schwartz_vector=data.get("psych_vector", None)
                ))
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
            schwartz_alignment_score = 0.0

            schwartz_vector = data_inst.schwartz_vector
            
            try:
                history_str = self._build_product_list_str(data_inst.history)
                psych_vector_str = ""
                if schwartz_vector:
                    calibrated_vector = self.calibrate_psych_vector(schwartz_vector)
                    psych_vector_str = str(calibrated_vector)
                prompt = prompt_template.format(history_str=history_str, psych_vector_str=psych_vector_str)
                @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
                def call_persona_model():
                    response_message = persona_model([{"role": "user", "content": prompt}])
                    return response_message.content or ""
                raw_output = call_persona_model()
                traits, persona_text = self.parse_persona_response(raw_output)
                grounding_score = self._grounding_score({"traits": traits}, [h.get("review_excerpt", "") for h in data_inst.history])
                alignment_grade, raw_pvq_val = self._score_schwartz_alignment(persona_text, data_inst.schwartz_vector)
                score = grounding_score * 0.5 + alignment_grade * 0.5            
            except Exception as e:
                print(f"Error generating persona: {e}")
                persona_text = ""
                score = 0.0
                grounding_score = 0.0
                alignment_grade = 0.0
                raw_pvq_val = 0.0

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
                        schwartz_vector=schwartz_vector,
                        schwartz_alignment_score=alignment_grade,
                        raw_pvq_score=raw_pvq_val,
                        total_score=score,
                    )
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

            
    def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch: EvaluationBatch[PersonaTrajectory, str],
            components_to_update: list[str],
        ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
            datasets: dict[str, list[dict[str, Any]]] = {}

            if "persona_prompt" not in components_to_update:
                return datasets

            trajectories = eval_batch.trajectories or []
            records: list[dict[str, Any]] = []

            # Sort by score (lowest score = needs most improvement)
            sorted_trajs = sorted(trajectories, key=lambda t: t.total_score)
            
            # Focus on the bottom 5 failures
            selected_trajs = sorted_trajs[:5] 

            print(f"Generating diagnostic critiques for {len(selected_trajs)} trajectories...")

            for traj in selected_trajs:
                # 1. Identify the Dominant Trait that we were testing for
                if traj.schwartz_vector:
                    sorted_vec = sorted(traj.schwartz_vector.items(), key=lambda x: x[1], reverse=True)
                    primary_trait, target_val = sorted_vec[0]
                    # Get the score the persona actually got on the test
                    # (Note: You need to make sure 'alignment_score' in trajectory is the PVQ score or store pvq results in traj)
                    # Assuming alignment_score tracks the success of the primary trait:
                    measured_val = traj.raw_pvq_score 
                else:
                    primary_trait = "N/A"
                    target_val = 0.0
                    measured_val = 0.0

                # 2. Construct the Failure Narrative
                # This is the raw data the Teacher needs to see to diagnose "Behavioral Overwrite"
                critique_prompt = f"""
                DIAGNOSIS TASK:
                Analyze why this AI Persona failed to align with its Psychological Profile.

                1. THE TARGET PSYCHOLOGY (Input):
                - Dominant Trait: {primary_trait} (Value: {target_val})
                - Context: {traj.schwartz_vector}

                2. THE SHOPPING BEHAVIOR (Input):
                - History Snippet: {traj.history_str[:400]}...

                3. THE GENERATED PERSONA (Output):
                - "{traj.traits_json.get("persona_description", "")}"

                4. THE FAILURE METRIC:
                - The generated persona took a Psychometric Test (PVQ).
                - Target Score for {primary_trait}: High
                - Actual Score for {primary_trait}: {measured_val:.2f} (Low/Mismatch)

                ANALYSIS QUESTION:
                Did the model ignore the {primary_trait} value? Did it let the shopping history override the psychological constraints?
                Write a 1-sentence diagnosis of the error.
                """

                try:
                    # Ask Teacher Model to diagnose the specific error
                    specific_critique = teacher_model(critique_prompt)
                except Exception as e:
                    print(f"Critique generation failed: {e}")
                    specific_critique = f"Failed to align {primary_trait}. Target High, Measured Low."

                feedback = (
                    f"Evaluation Score: {traj.total_score:.3f}. "
                    f"Diagnosis: {specific_critique}"
                )

                rec = {
                    "Inputs": {
                        "history": traj.history_str,
                        "psych_vector": str(traj.schwartz_vector),
                    },
                    "Generated Outputs": traj.traits_json.get("persona_description", ""),
                    "Feedback": feedback,
                    "score": traj.total_score,
                    "user_id": traj.user_id,
                }
                records.append(rec)

            datasets["persona_prompt"] = records
            return datasets
        
        # propose_new_texts: ProposalFn | None = None

def custom_proposal_function(
    candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    components_to_update: list[str],
) -> dict[str, str]:
    
    current_prompt = candidate["persona_prompt"]
    failures = reflective_dataset.get("persona_prompt", [])
    
    # 1. Compile the Failure Report
    examples_str = ""
    for i, fail in enumerate(failures):
        examples_str += f"\n--- FAILURE CASE {i+1} ---\n"
        examples_str += f"Context (Schwartz): {fail['Inputs']['psych_vector']}\n"
        examples_str += f"Behavior (History): {fail['Inputs']['history'][:200]}...\n"
        examples_str += f"Bad Output: {fail['Generated Outputs']}\n"
        examples_str += f"DIAGNOSIS: {fail['Feedback']}\n"

    # 2. The "System Architect" Prompt
    # This instructs the Teacher to fix the "Behavioral Overwrite" bug
    meta_prompt = f"""
    You are an AI System Architect optimizing a Persona Generator Prompt.
    
    THE PROBLEM: "Behavioral Overwrite"
    Our model is failing to bridge "Psychological Context" with "Shopping History."
    When the inputs contradict (e.g., Anxious User buys a Motorcycle), the model ignores the Anxiety (Context) and writes a Risk-Taking Persona (Behavior).
    
    THE EVIDENCE (Failures):
    {examples_str}
    
    YOUR GOAL:
    Rewrite the "CURRENT PROMPT" to enforce **Conflict Resolution.**
    
    Strategies to implement in the new prompt:
    1. **Hierarchy:** Explicitly state that Schwartz Values represent the "True Internal Motivation," even if behavior seems different.
    2. **Synthesis:** Instruct the model to explain *why* a user with [Value X] would buy [Item Y] (e.g., "Buying a fast car to feel in control," not just "likes speed").
    3. **Evidence Requirement:** Demand that every trait cites the Vector if available.
    
    === CURRENT PROMPT ===
    {current_prompt}
    
    === NEW OPTIMIZED PROMPT ===
    Return ONLY the full text of the new System Prompt.
    """

    print("Optimizing Prompt based on PVQ Failures...")
    new_prompt_text = teacher_model(meta_prompt, temperature=0.7)
    
    return {"persona_prompt": new_prompt_text}
if __name__ == "__main__":
    #test loading 1 user and their purchases

    
    
    
    
    base_candidate = {
    "persona_prompt": UCSD_PERSONA_PROMPT
}


  
    trainset = load_persona_dataset("traininggdelt_enriched.jsonl")
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(trainset)
    # print(trainset[0].history[0].get("rating"))
    # print(trainset[1].history[0].get("review_excerpt"))
    # print(trainset[0].schwartz_vector)
    # schwartz_vector = trainset[4].schwartz_vector
    # print(trainset[0].history[0].get("product").get("title"))
    # print(trainset[0].heldout)
    valset   = load_persona_dataset("validationgdelt_enriched.jsonl")
    adapter = PersonaGEPAAdapter()
 


 
    # heldout_str = adapter.build_heldout_str(trainset[0].heldout)
    # print("Heldout str:", heldout_str)
    # print(product_list_str)
    # prompt = UCSD_PERSONA_PROMPT.format(history_str=product_list_str, psych_vector_str=str(balanced_vector))
    # response_message = persona_model([{"role": "user", "content": prompt}])

    # print(response_message.content)
    # traits, persona_description = adapter.parse_persona_response(response_message.content)

    # print(persona_description)
    # example_persona = "This individual consistently seeks out predictable and reliable experiences, demonstrating a preference for well-known establishments. They seem to derive satisfaction from comfort and routine, with a notable aversion to risk or unpredictability in their downtime. While not averse to modest enjoyment, they do not appear driven by intense thrills or impulsive behaviors, suggesting a measured and pragmatic temperament. Their choices indicate a desire for stability and a comfort within established social norms, highlighting a cautious and security-oriented outlook on leisure activities."
    # pnq = adapter._score_schwartz_alignment(example_persona, balanced_vector)
    # print("PVQ Test Results:", pnq)
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


    

    adapter.propose_new_texts = custom_proposal_function

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




    # user_vector_distributions = {}
    # totals = {}
    # counts = {}

    # for i in range(len(trainset)):
    #     schwartz_vector = trainset[i].schwartz_vector
    #     if not schwartz_vector:
    #         print(f"User {i} has no schwartz vector, skipping. {trainset[i].user_id}")
    #         continue
    #     balanced_vector = adapter.calibrate_psych_vector(schwartz_vector)
    #     print(f"User {i} Balanced Schwartz Vector:", balanced_vector)
    #     #get the dominant trait
    #     dominant_trait = max(balanced_vector, key=balanced_vector.get)
    #     if dominant_trait not in user_vector_distributions:
    #         user_vector_distributions[dominant_trait] = 1
    #     else:
    #         user_vector_distributions[dominant_trait] += 1
    # print("User Vector Distributions:", user_vector_distributions)
