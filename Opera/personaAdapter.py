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

# ============================================================================
# PURCHASE-ONLY MODE CONFIGURATION
# ============================================================================
# When True, filters gold personas and scoring to only evaluate traits that 
# can be inferred from purchase data alone (not clicks/views/reviews)
PURCHASE_ONLY_MODE = True

# Keywords indicating traits that CANNOT be inferred from purchases alone
# These will be filtered out during scoring and reflection
NON_INFERABLE_KEYWORDS = [
    # Review-related behaviors (can't infer how someone reads reviews from purchases)
    "read review", "reading review", "reviews with image", "negative review", 
    "positive review", "examines review", "focuses on review", "checks review",
    "review strategy", "detailed review", "star rating", "ratings and review",
    "product's rating", "value a product's rating", "customer review",
    # Browsing/viewing behaviors
    "view multiple", "views multiple", "browsing", "compares options",
    "time researching", "researching product", "comparing product",
    "shopping process", "search behavior", "filtering", "decision-making process",
    # Sponsored/ad awareness (can't infer from purchases)
    "sponsored product", "sponsored listing", "advertisement", "ad awareness",
    "promoted product", "paid placement", "hold negative stereotypes against sponsored",
    # AI/tech tool usage
    "ai tool", "rufus", "ai-generated", "chatbot", "virtual assistant",
    "review summar", "ai tools like",
    # Decision timing/process
    "time-consuming", "quick decision", "impulse", "deliberate",
    "return policy", "considers return", "difficult or time",
    # ChatGPT/AI recommendations
    "chatgpt", "gpt recommendation", "ai recommendation",
]

# Fallback persona for when all gold persona content is non-inferable
FALLBACK_GOLD_PERSONA = """[Participant] demonstrates specific category preferences and shopping patterns 
based on their purchase history. Their brand choices and price points reveal their value orientation 
and quality expectations. The categories they purchase online versus what they likely buy elsewhere 
indicate their channel preferences."""

def filter_gold_persona_for_purchases(gold_persona: str) -> str:
    """
    Extract only the purchase-inferable aspects from a gold persona.
    Removes sentences that reference behaviors requiring click/view/review data.
    
    Returns a filtered persona focusing on:
    - Category preferences (what they buy vs don't buy)
    - Brand tier preferences (premium vs value brands)
    - Price sensitivity (inferable from actual prices paid)
    - Shopping channel preferences (online vs offline for categories)
    """
    if not PURCHASE_ONLY_MODE:
        return gold_persona
    
    sentences = re.split(r'(?<=[.!?])\s+', gold_persona)
    filtered_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains non-inferable keywords
        contains_non_inferable = any(
            keyword in sentence_lower for keyword in NON_INFERABLE_KEYWORDS
        )
        if not contains_non_inferable and sentence.strip():
            filtered_sentences.append(sentence)
    
    # If we filtered out everything, use fallback that focuses on purchase-inferable aspects
    if not filtered_sentences:
        print(f"Warning: Gold persona contained no purchase-inferable content. Using fallback.")
        return FALLBACK_GOLD_PERSONA
    
    return " ".join(filtered_sentences)

@dataclass
class PersonaDataInst:
    user_id: str
    interactions: list[dict[str, Any]]  # your JSON: asin, title, price, options, etc.
    gold_persona: str                # human-written shopping-preference text only

@dataclass
class PersonaTrajectory:
    user_id: str
    purchases_str: str
    gold_persona: str
    generated_persona: str
    score: float
    # you can add extra fields if you like, e.g. error messages



RolloutOutput = TypeVar("RolloutOutput") # the generated persona description
Trajectory = PersonaTrajectory
DataInst = PersonaDataInst
Candidate = dict[str, str]
EvaluatorFn = Callable[[list[DataInst], Candidate], tuple[list[RolloutOutput], list[float]]] # the evaluator function

BASE_PROMPT_STRING = """
You are an expert Consumer Psychologist. Your goal is to infer a user's detailed **Shopping Persona** based ONLY on their confirmed Purchase History.

Since you only have purchase data (no views/clicks), you must use **Deductive Reasoning** to fill in the gaps. You must look for what is *missing* just as much as what is *present*.

=== EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===

**Input Purchase History:**
- Hair Dryer Blow Dryer, 180000 RPM High-Speed Brushless Motor (None)
- License Plate Screws with Rustproof Finish - Stainless Steel (4-Pack, Black) (None)
- AIRROBO Robot Vacuum and Mop, 3000Pa Powerful Suction (None)
- NADALY D200 Robot Vacuum and Mop Combo, Lidar Navigation (None)

**Psychological Analysis (Internal Reasoning):**
1. **Brand Detective:** The user bought "AIRROBO" and "NADALY" vacuums. These are not famous "default" brands like Roomba or Dyson. They are high-spec, value-priced, online-native brands.
   * *Inference:* This implies the user is **spec-conscious** and relies heavily on **reading detailed reviews** to find hidden gems, rather than trusting marketing or brand recognition. They are cautious about overpaying for big names.
2. **Inference by Omission:** The list is 100% "Hard Goods" (Hardware, Electronics, Tools). There are no clothes, food, or consumables.
   * *Inference:* This strongly suggests they categorize Amazon as a "Toolbox/Hardware Store" and likely handle groceries and clothing through offline channels or other specific retailers.
3. **Micro-Optimization:** Buying specific "Rustproof Black License Plate Screws" and a "180000 RPM" dryer indicates a high attention to detail. They prioritize **functionality, durability, and specific fit** over generic solutions.
4. **Strategic Synthesis:** The user is a researcher. They compare highly positive and negative reviews to ensure the "unknown" brands (Nadaly) are safe.

**Final Persona Description (Clean Output):**
<persona_description>
[Participant] prefers shopping for certain categories like home essentials and specialized tools online but tends to buy groceries and clothes offline. They read reviews, especially for unfamiliar products, focusing on detailed reviews and images to assess product quality and fit, such as ease of assembly or actual appearance. [Participant] compares both highly positive and negative reviews to get a balanced perspective. They are cautious about sponsored products, often avoiding them due to concerns over biased promotion, and prefer to check non-sponsored listings to ensure a more genuine assessment.
</persona_description>

=== END EXAMPLE ===

**YOUR TASK:**
Analyze the following PURCHASE HISTORY for the current user.

**Purchase History:**
{product_list_str}

**Instructions:**
1. **Analyze Brand Tier:** Are these "Famous Brands," "Value Brands," or "High-Spec Unknowns"? 
   - *Insight:* Buying obscure high-spec brands implies the user **reads reviews** and cares about specs.
   
2. **Analyze "Inference by Omission":** - If they buy durable goods but NO food/clothes, you **MUST** infer: *"Likely handles groceries and clothing through offline channels."*

3. **Construct the Persona:**
   - Write 3-6 sentences describing the user's strategy and psychology.
   - **CRITICAL RULE:** Do NOT cite specific items in the final description (e.g., do not say "evidenced by the vacuum"). Just state the trait (e.g., "They prioritize home automation").
   - **CRITICAL RULE:** Do NOT mention "insufficient data." Use the deductions above to form a complete picture.

**Output Format:**
**1. Brand & Tier Analysis:** [Your deductive reasoning]
**2. Strategic Omissions:** [What are they NOT buying?]
**3. Behavioral Conclusion:** [Synthesize the traits]

<persona_description>
[Your clean final paragraph]
</persona_description>
"""
# BASE_PROMPT_STRING = BASE_PROMPT_STRING.format(product_list_str=product_list_str)



# base_candidate = {
#     "persona_prompt": "Based on the items "  # your hand-written persona prompt
# }


def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        raw = json.load(f)
    examples: list[PersonaDataInst] = []
    for row in raw:
        examples.append(
            PersonaDataInst(
                user_id=row["user_id"],
                interactions=row["interactions"],
                gold_persona=row["gold_persona"],
            )
        )
    return examples
@dataclass
class EvaluationBatch[PersonaTrajectory, str]:
    """
    Container for the result of evaluating a proposed candidate on a batch of data.

    - outputs: raw per-example outputs from upon executing the candidate. GEPA does not interpret these;
      they are forwarded to other parts of the user's code or logging as-is.
    - scores: per-example numeric scores (floats). GEPA sums these for minibatch acceptance
      and averages them over the full validation set for tracking/pareto fronts.
    - trajectories: optional per-example traces used by make_reflective_dataset to build
      a reflective dataset (See `GEPAAdapter.make_reflective_dataset`). If capture_traces=True is passed to `evaluate`, trajectories
      should be provided and align one-to-one with `outputs` and `scores`.
    """

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None


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

class PersonaGEPAAdapter(GEPAAdapter[PersonaDataInst, PersonaTrajectory, str]):
    def _build_product_list_str(self, interactions: list[dict[str, Any]], filter_type: str | list[str] | None = "purchase") -> str:
        """
        Build a string of products grouped by type with section headers.
        
        Args:
            interactions: List of interaction dicts with 'type', 'title', 'price', etc.
            filter_type: Type(s) to filter by. Can be:
                - A single string: "purchase", "cart", "click"
                - A list of strings: ["purchase", "click"] to include multiple types
                - None: include all interactions
        """
        # Determine which types to include
        if filter_type is None:
            types_to_include = ["purchase", "cart", "click"]
        elif isinstance(filter_type, list):
            types_to_include = filter_type
        else:
            types_to_include = [filter_type]
        
        # Group items by type
        grouped = {
            "purchase": [],
            "cart": [],
            "click": []
        }
        
        for item in interactions:
            if item.get("type") is None:
                print(f"Item type is None: {item}")
                continue
            item_type = item.get("type")
            if item_type in types_to_include and item_type in grouped:
                grouped[item_type].append(item)
        
        # Build formatted sections
        sections = []
        
        # Purchased Products section
        if grouped["purchase"]:
            purchase_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["purchase"]]
            sections.append("**Purchased Products:**\n" + "\n".join(purchase_lines))
        
        # Items Added to Cart section
        if grouped["cart"]:
            cart_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["cart"]]
            sections.append("**Items Added to Cart (but not bought):**\n" + "\n".join(cart_lines))
        
        # Clicked / Viewed Items section
        if grouped["click"]:
            click_lines = [f"- {item['title']} ({item.get('price', 'N/A')})" for item in grouped["click"]]
            sections.append("**Clicked / Viewed Items:**\n" + "\n".join(click_lines))
        
        # Join all sections with double newline for spacing
        return "\n\n".join(sections) if sections else ""

    def _score_with_llm(self, generated: str, gold: str) -> float:
        """
        Score using LLM judge with PURCHASE-ALIGNED criteria.
        
        Key change: We evaluate based on what CAN be inferred from purchases,
        not on matching aspects that require click/view data.
        """
        # Filter gold persona if in purchase-only mode
        filtered_gold = filter_gold_persona_for_purchases(gold)
        
        if PURCHASE_ONLY_MODE:
            # PURCHASE-ALIGNED RUBRIC: Only evaluate inferable traits
            rubric_prompt = textwrap.dedent(f"""
                You are evaluating a shopping persona generated from PURCHASE DATA ONLY.
                The model had NO access to: clicks, views, browsing behavior, or review reading patterns.
                
                Filtered Reference (purchase-inferable traits only): "{filtered_gold}"
                Generated Profile: "{generated}"
                
                Evaluate on these 5 PURCHASE-INFERABLE criteria.
                For each, provide "Reasoning" then "Verdict" (YES/NO).

                1. Does it mention specific product names/brands in the final description? (YES = Bad)
                   - Mentioning brand TIERS (e.g., "value brands") is OK
                   - Mentioning specific brands (e.g., "bought NADALY vacuum") is BAD
                   
                2. Does it correctly infer CATEGORY PREFERENCES from purchases? (YES = Good)
                   - E.g., "prefers home goods over clothing" based on purchase mix
                   
                3. Does it make UNSUPPORTED CLAIMS about behaviors requiring click/view data? (YES = Bad)
                   - BAD: "spends time reading reviews" (can't infer from purchases)
                   - BAD: "compares many options before buying" (can't infer from purchases)
                   - OK: "likely researches before buying niche brands" (logical inference)
                   
                4. Does it identify SHOPPING CHANNEL PREFERENCES based on category gaps? (YES = Good)
                   - E.g., "likely shops offline for groceries" if no food purchases
                   
                5. Does it identify meaningful INTEREST CLUSTERS from purchase patterns? (YES = Good)
                   - E.g., "Home Automation enthusiast" or "DIY/Repair focused"

                Output format:
                1. Reasoning: ... Verdict: [YES/NO]
                2. Reasoning: ... Verdict: [YES/NO]
                3. Reasoning: ... Verdict: [YES/NO]
                4. Reasoning: ... Verdict: [YES/NO]
                5. Reasoning: ... Verdict: [YES/NO]
            """)
        else:
            # Original rubric for full-data mode
            rubric_prompt = textwrap.dedent(f"""
                Gold Standard: "{gold}"
                Student Profile: "{generated}"
                
                Evaluate the Student Profile on these 5 criteria.
                For each, provide a brief "Reasoning" then a "Verdict" (YES/NO).

                1. Does it mention specific items (chicken, brands) instead of broad habits? (YES = Bad)
                2. Does it accurately reflect the user's review strategy (e.g. ignores ads)? (YES = Good)
                3. Does it contradict the purchase history? (YES = Bad)
                4. Does it identify a shopping strategy (e.g. offline vs online)? (YES = Good)
                5. **DEPTH BONUS:** Does it identify specific INTEREST CLUSTERS? (YES = Good)

                Output format:
                1. Reasoning: ... Verdict: [YES/NO]
                2. Reasoning: ... Verdict: [YES/NO]
                ...
            """)
            
        # Call your teacher model
        response_message = teacher_model(
            [{"role": "user", "content": rubric_prompt}],
            temperature=0.0,
            seed=42
        )

        # Handle response
        if hasattr(response_message, 'content'):
            response_str = response_message.content or ""
        elif isinstance(response_message, str):
            response_str = response_message
        else:
            response_str = str(response_message) if response_message else ""
            
        criteria = {}
        for i in range(1, 6):
            # Match patterns like "1: YES", "1. ... Verdict: YES", "Verdict: YES" near criterion number
            patterns = [
                rf"{i}[.:\s]+.*?(YES|NO)",  # "1. ... YES" or "1: YES"
                rf"{i}[.:\s]+.*?Verdict[:\s]*(YES|NO)",  # "1. Reasoning: ... Verdict: YES"
            ]
            matched = False
            for pattern in patterns:
                match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
                if match:
                    criteria[i] = match.group(1).upper() == "YES"
                    matched = True
                    break
            if not matched:
                # Default to worst case if parsing fails
                criteria[i] = False if i in [2, 4, 5] else True
        
        # Calculate score: Good criteria (2,4,5) minus Bad criteria (1,3)
        raw_score = (int(criteria[2]) + int(criteria[4]) + int(criteria[5])) - (int(criteria[1]) + int(criteria[3]))
        
        # Normalize from [-2, 3] to [0.0, 1.0]
        normalized_score = (raw_score + 2) / 5.0
        
        return max(0.0, min(1.0, normalized_score))


    def _score_persona(self, persona_description: str, gold_persona: str) -> float:
        """
        Score the persona description based on the gold persona.
        
        In PURCHASE_ONLY_MODE, we filter the gold persona to only include
        traits that can be inferred from purchase data before scoring.
        """
        # Filter gold persona if in purchase-only mode
        filtered_gold = filter_gold_persona_for_purchases(gold_persona)
        
        if PURCHASE_ONLY_MODE:
            # In purchase-only mode, skip NLI check entirely and use LLM judge only.
            # Rationale: NLI is too strict when filtered gold personas are short/generic.
            # The LLM judge with purchase-aligned rubric is more appropriate.
            llm_quality_score = self._score_with_llm(persona_description, gold_persona)
            return llm_quality_score
        
        # Full-data mode: Use NLI as a sanity check
        scores = nli_model.predict([(filtered_gold, persona_description)])
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        entailment_score = probs[0][2]
        nli_score = float(entailment_score)
    
        if nli_score < 0.3:
            # If it contradicts the truth, fail immediately
            return nli_score 
            
        # Quality Check (LLM Judge): Is it insightful?
        llm_quality_score = self._score_with_llm(persona_description, gold_persona)
        
        return llm_quality_score



    def load_persona_dataset(self, path: str) -> list[PersonaDataInst]:
        with open(path, "r") as f:
            raw = json.load(f)
        examples: list[PersonaDataInst] = []
        for row in raw:
            examples.append(
                PersonaDataInst(
                    user_id=row["user_id"],
                    interactions=row["interactions"],
                    gold_persona=row["gold_persona"],
                )
            )
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
            try:
                product_list_str = self._build_product_list_str(
                    data_inst.interactions, "purchase"
                )

                prompt = prompt_template.format(product_list_str=product_list_str)
                @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
                def call_persona_model():
                    response_message = persona_model([{"role": "user", "content": prompt}])
                    return response_message.content or ""
                raw_output = call_persona_model()

                # (optional) extract only <persona_description>...</persona_description>
                persona_text = raw_output  # you can refine this later
                # Use regex to find the last occurrence, handling whitespace variations
                matches = list(re.finditer(r'<persona_description>(.*?)</persona_description>', persona_text, re.DOTALL))
                if matches:
                    # Get the last match (in case there are examples earlier in the output)
                    extracted_persona = matches[-1].group(1).strip()
                else:
                    # Fallback: use the entire output if tags are missing
                    extracted_persona = persona_text.strip()
                    print(f"Warning: No <persona_description> tags found, using full output")


                score = self._score_persona(extracted_persona, data_inst.gold_persona)


            except Exception as e:
                print(f"Error generating persona: {e}")
                extracted_persona = ""
                score = 0.0

            outputs.append(extracted_persona)
            scores.append(score)

            if capture_traces:
                product_list_str = self._build_product_list_str(
                    data_inst.interactions, None
                )
                trajectories.append(
                    PersonaTrajectory(
                        user_id=data_inst.user_id,
                        purchases_str=product_list_str,
                        gold_persona=data_inst.gold_persona,
                        generated_persona=extracted_persona,
                        score=score,
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
        sorted_trajs = sorted(trajectories, key=lambda t: t.score)
        selected_trajs = sorted_trajs[:8] 

        print(f"Generating dynamic critiques for {len(selected_trajs)} trajectories...")

        for traj in selected_trajs:
            # Filter the gold persona to only include purchase-inferable traits
            filtered_gold = filter_gold_persona_for_purchases(traj.gold_persona)
            
            if PURCHASE_ONLY_MODE:
                # PURCHASE-ALIGNED CRITIQUE PROMPT
                critique_prompt = f"""
                I am optimizing an AI to generate "Shopper Psychographics" from PURCHASE DATA ONLY.
                
                CRITICAL CONTEXT: The AI only sees what was PURCHASED. It has NO access to:
                - What items were viewed/clicked but not bought
                - How many items were compared
                - Whether the user read reviews
                - Time spent browsing
                
                User Purchases (THE ONLY DATA AVAILABLE):
                {traj.purchases_str}

                Purchase-Inferable Reference Traits:
                {filtered_gold}

                Generated Output:
                {traj.generated_persona}

                === CRITIQUE TASK (PURCHASE-ONLY MODE) ===
                Evaluate ONLY on traits that CAN be inferred from purchases:
                
                1. ABSTRACTION CHECK: Does it avoid listing specific products?
                   - BAD: "Bought NADALY vacuum"
                   - GOOD: "Prioritizes home automation"
                
                2. CATEGORY INFERENCE: Does it correctly identify category preferences from what WAS and WASN'T purchased?
                   - If only electronics purchased: "Likely shops offline for groceries/clothes"
                
                3. BRAND TIER ANALYSIS: Does it infer traits from brand choices?
                   - Niche brands → "Likely researches before buying"
                   - Premium brands → "Values quality over price"
                   - Budget brands → "Price-conscious"
                
                DO NOT CRITICIZE the model for failing to mention:
                - Review reading behavior (not observable from purchases)
                - Browsing patterns (not observable from purchases)
                - Sponsored product awareness (not observable from purchases)
                - Decision-making process details (not observable from purchases)
                
                Provide ACTIONABLE feedback focused on PURCHASE-INFERABLE improvements only.
                """
            else:
                # Original critique prompt for full-data mode
                critique_prompt = f"""
                I am optimizing an AI to generate "Shopper Psychographics" from transaction logs.
                Compare the Generated Persona with the Gold Truth.

                User Purchases:
                {traj.purchases_str}

                Gold Truth (Psychology):
                {traj.gold_persona}

                Generated Output:
                {traj.generated_persona}

                === CRITIQUE TASK ===
                Analyze the gap between the Generated Output and the Gold Truth.
                
                1. ABSTRACTION CHECK: Does the output just list items or infer lifestyle?
                2. INFERENCE QUALITY: Did the model miss "Inference by Omission"?
                3. BRAND ANALYSIS: Does it correctly infer traits from Brand Choices?
                
                Provide specific, constructive feedback.
                """
                        
            try:
                specific_critique = teacher_model(critique_prompt)
            except Exception as e:
                print(f"Critique generation failed: {e}")
                specific_critique = "Improve alignment with purchase-inferable traits."

            feedback = (
                f"Score: {traj.score:.3f}. "
                f"Critique: {specific_critique} "
                "Focus on improving purchase-inferable traits only."
            )

            rec = {
                "Inputs": {
                    "purchases": traj.purchases_str,
                    # Use filtered gold persona so GEPA doesn't chase impossible targets
                    "gold_persona": filtered_gold if PURCHASE_ONLY_MODE else traj.gold_persona,
                },
                "Generated Outputs": traj.generated_persona,
                "Feedback": feedback,
                "score": traj.score,
                "user_id": traj.user_id,
            }
            records.append(rec)

        datasets["persona_prompt"] = records
        return datasets

    propose_new_texts: ProposalFn | None = None


def custom_proposal_function(
    candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    components_to_update: list[str],
) -> dict[str, str]:
    
    current_prompt = candidate["persona_prompt"]
    failures = reflective_dataset.get("persona_prompt", [])
    
    examples_str = ""
    for i, fail in enumerate(failures):
        examples_str += f"\n--- Example {i+1} ---\n"
        examples_str += f"User Input (Purchases):\n{fail['Inputs']['purchases']}\n"
        examples_str += f"Current AI Output:\n{fail['Generated Outputs']}\n"
        examples_str += f"CRITIQUE (What went wrong):\n{fail['Feedback']}\n"

    if PURCHASE_ONLY_MODE:
        # PURCHASE-ALIGNED META-PROMPT
        meta_prompt_core = """You are an expert Prompt Engineer for an e-commerce AI system.
Your goal is to optimize a "System Instruction" that converts a user's *PURCHASE HISTORY ONLY* into a "Persona Description."

CRITICAL CONSTRAINT: The AI ONLY has access to PURCHASED items. It has NO access to:
- Viewed/clicked items that weren't purchased
- Browsing time or session patterns  
- Review reading behavior
- Sponsored product awareness

The prompt MUST NOT instruct the model to:
- Claim how the user researches (e.g., "reads reviews thoroughly") - this cannot be observed
- Reference "viewing patterns" or "click behavior" - this data doesn't exist
- Make claims about decision-making process - only outcomes are visible

The prompt SHOULD instruct the model to:
1. INFER FROM PURCHASES: What categories do they buy? What brand tiers?
2. INFER FROM OMISSIONS: What categories are ABSENT? (e.g., no food = shops offline for groceries)
3. INFER FROM BRAND CHOICES: Niche brands suggest research; premium brands suggest quality focus
4. SYNTHESIZE ABSTRACTLY: "Home automation enthusiast" not "bought vacuum and smart speaker"

I will show you:
1. The CURRENT PROMPT being used.
2. FAILURE CASES with critiques focused on PURCHASE-INFERABLE improvements only.

Your Task: Rewrite the prompt to better extract purchase-inferable insights while AVOIDING any instructions that assume click/view/browsing data exists.
"""
    else:
        meta_prompt_core = """You are an expert Prompt Engineer for an e-commerce AI system.
Your goal is to optimize a "System Instruction" that converts a user's *Purchase History* into a specific "Persona Description."

I will show you:
1. The CURRENT PROMPT being used.
2. A list of FAILURE CASES (User History -> Generated Persona -> Critique).

Your Task:
Analyze the Critical Feedback. Identify patterns in what the Current Prompt is missing.
Then, write a NEW, IMPROVED PROMPT that addresses these specific weaknesses.

CRITICAL GUIDELINES FOR THE NEW PROMPT:
1. BAN "ITEM LISTING": Forbid listing specific purchases.
2. INFER FROM OMISSION: Look for what is *missing*.
3. INFER FROM CONSISTENCY: Analyze Brand/Price Tiers.
4. SYNTHESIZE CATEGORIES: Abstract from items to patterns.
"""

    meta_prompt_input = textwrap.dedent(f"""
    {meta_prompt_core}

    === CRITICAL CONSTRAINTS ===
    {"The new prompt MUST NOT reference 'views', 'clicks', 'browsing', or 'review reading' as these are NOT observable from purchase data." if PURCHASE_ONLY_MODE else ""}
    
    === HINT ===
    Add a mandatory "Deductive Analysis Step" where the model must answer:
    "What do the Brand Choices and Missing Categories tell us about the user's strategy?"
    before writing the final description.
    
    === CURRENT PROMPT ===
    {current_prompt}
    
    === FAILURE ANALYSIS (Critiques from the Judge) ===
    {examples_str}
    
    === TASK ===
    Based on the critiques above, rewrite the "CURRENT PROMPT" to fix the recurring errors.
    {"REMEMBER: Do NOT add instructions about analyzing views/clicks/reviews - this data is NOT available." if PURCHASE_ONLY_MODE else ""}
    Return ONLY the new prompt text, ready to be pasted into the system.
    """)

    new_prompt_text = teacher_model(meta_prompt_input, temperature=0.7)
    
    return {"persona_prompt": new_prompt_text}
    
if __name__ == "__main__":
    #test loading 1 user and their purchases

    
    
    
    
    base_candidate = {
    "persona_prompt": BASE_PROMPT_STRING
}


  
    trainset = load_persona_dataset("data/train.json")
    valset   = load_persona_dataset("data/val.json")
    adapter = PersonaGEPAAdapter()
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
    