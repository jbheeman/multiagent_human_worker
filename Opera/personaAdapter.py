# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
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
        model_id="qwen3",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

#This model evaluates the persona description   
teacher_model_raw= OpenAIServerModel( # Still used for persona agent
        model_id="gpt-oss",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
    )

teacher_model = GEPACompatibleModel(teacher_model_raw)

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
Based on the following shopping interaction history, please infer a persona for the user.

The interaction list may include:
- **Purchased Products**: strong evidence of true preferences and needs.
- **Items Added to Cart (but not bought)**: moderate evidence of interest or intent.
- **Clicked / Viewed Items**: weak evidence of curiosity or early-stage interest.

Treat purchases as the strongest signal, cart items as secondary, and clicks/views as the weakest signal.

{product_list_str}

**Instructions:**
You must follow these steps and show your work for each one:

1. **Extract Traits:** For each relevant product or interaction, identify key traits (e.g., brand, category, price point, features, implied hobbies or interests).
2. **Identify Buying Patterns:** Look for patterns across interactions to determine the user's buying habits and preferences. Consider at least:
   - Shopping frequency/intensity (do they appear to shop often or occasionally?)
   - Price sensitivity (budget-conscious vs willing to pay more for quality)
   - Category focus (e.g., pet care, electronics, home goods, beauty, etc.)
   - Brand behavior (brand-loyal vs exploratory)
   - How they might use reviews/ratings when choosing products
   - Openness to novelty (trying new product types vs sticking to familiar ones)
3. **Categorize Traits:** Group the user's inferred buying traits into the following four categories:
   - **Confident Likes:** Things we are confident the person likes.
   - **Somewhat Confident Likes:** Things we are somewhat confident the person likes.
   - **Confident Dislikes:** Things we are confident the person dislikes.
   - **Somewhat Confident Dislikes:** Things we are somewhat confident the person dislikes.
4. **Generate Persona Description:** Based on the categorized traits, write a concise plaintext paragraph describing the user's *shopping persona*. This description should be suitable for guiding a research assistant and should be 3–6 sentences long.
5. **Do not infer demographic details** (age, gender, location, education level, family status, etc.) unless they are explicitly stated in the product descriptions. Focus only on shopping behavior and preferences.

**Output Format:**
You must provide your full reasoning for steps 1–3. After your reasoning, provide the final persona description enclosed in `<persona_description>` tags.

**Example Output:**
**1. Extracted Traits:**
- Airkeep Car Air Freshener: Low price, home/car accessory, scent-focused.
- Lumiere & Co. Bike Seat Bag: Mid-range price, cycling accessory, practical.
...

**2. Buying Patterns:**
- The user frequently buys cycling-related gear, suggesting a hobby in cycling.
- The user purchases items at various price points, but seems to value function over luxury.
...

**3. Categorized Traits:**
- **Confident Likes:** Cycling, practical items.
- **Somewhat Confident Likes:** Home fragrance, pet safety.
...

<persona_description>
The user is a practical, budget-conscious individual who prioritizes functionality and value. They are an avid cyclist, investing in quality components for their hobby. They are not brand-loyal but seem to prefer items with good reviews and a focus on durability. They show some interest in home and pet accessories, but are not driven by luxury or high-end brands.
</persona_description>
"""

# BASE_PROMPT_STRING = BASE_PROMPT_STRING.format(product_list_str=product_list_str)



# base_candidate = {
#     "persona_prompt": "Based on the items "  # your hand-written persona prompt
# }

def _build_product_list_str(interactions: list[dict[str, Any]], filter_type: str | list[str] | None = "purchase") -> str:
    """
    Build a string of products filtered by type.
    
    Args:
        interactions: List of interaction dicts with 'type', 'title', 'price', etc.
        filter_type: Type to filter by ("purchase", "cart", "click", or None for all)
    """
    if filter_type:
        filtered = [item for item in interactions if item.get("type") == filter_type]
    else:
        filtered = interactions
    
    return "\n".join([f"- {item['title']} ({item.get('price', 'N/A')})" for item in filtered])


def _score_persona(persona_description: str, gold_persona: str) -> float:
    """
    Score the persona description based on the gold persona.
    """
    #Do we use BERT or something? 
    emb_pred = eval_model.encode(persona_description, normalize_embeddings=True)
    emb_gold = eval_model.encode(gold_persona, normalize_embeddings=True)
    sim = float(np.dot(emb_pred, emb_gold))  # cosine because normalized
    # Map from [-1, 1] to [0, 1]
    return (sim + 1.0) / 2.0



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
        # We ask the Teacher (Qwen) to be a harsh professor.
        rubric_prompt = f"""
        You are a psychology professor grading a student's analysis of a consumer.
        
        Gold Standard Profile (The Truth):
        "{gold}"
        
        Student's Generated Profile:
        "{generated}"
        
        Grade the Student's Profile on a scale of 0.0 to 1.0 based ONLY on these criteria:
        
        1. **Abstraction Level (Max 0.4):** Does the student identify *strategies* (e.g., "avoids sponsored ads")? Or do they just list items (e.g., "bought a license plate")? 
        - PENALIZE heavily for listing specific items like "chicken", "fries", or specific university names unless they explain the *psychology* behind it.
        
        2. **Completeness of Insight (Max 0.4):** Did they catch the nuance about *reviews*? (e.g., Gold says "reads reviews for UNFAMILIAR products"). Did they catch the "Online vs Offline" split?
        
        3. **Factuality (Max 0.2):** Are there any contradictions?

        Return ONLY the numeric float score (e.g., 0.75). Do not write an explanation.
        """
    
        # Call your teacher model - pass as string to get consistent string response
        response_str = teacher_model(rubric_prompt)
        # Ensure response is a string
        if not isinstance(response_str, str):
            response_str = str(response_str)
        # Extract numeric score from response
        try:
            # Try to find a float in the response
            score_match = re.search(r'0?\.\d+|1\.0|\d+\.\d+', response_str.strip())
            if score_match:
                score_str = score_match.group(0)
            else:
                score_str = response_str.strip()
            return float(score_str)
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse score from response: {response_str}, error: {e}")
            return 0.0

    def _score_persona(self, persona_description: str, gold_persona: str) -> float:
        """
        Score the persona description based on the gold persona.
        """

        # CrossEncoder input is a list of pairs: [(Premise, Hypothesis)]
        # We ask: "Does the Gold Persona entail the Generated Persona?"
        # (i.e., is the generated text factually consistent with the gold truth?)
        scores = nli_model.predict([(gold_persona, persona_description)])

        # The model outputs logits for [Contradiction, Neutral, Entailment]
        # We want the probability of "Entailment" (index 2) or "Not Contradiction"
        
        # We apply softmax to get probabilities

        # Score = Probability of Entailment
        # If Entailment is high, it means the generated persona aligns with the gold truth.
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        entailment_score = probs[0][2]
        nli_score =  float(entailment_score)

       
    
        if nli_score < 0.3:
            # If it contradicts the truth, fail immediately. Don't waste money on LLM scoring.
            return nli_score 
            
        # 2. Quality Check (LLM Judge): Is it insightful?
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
                    data_inst.interactions, None
                )

                prompt = prompt_template.format(product_list_str=product_list_str)
                response_message = persona_model([{"role": "user", "content": prompt}])
                raw_output = response_message.content or ""

                # (optional) extract only <persona_description>...</persona_description>
                persona_text = raw_output  # you can refine this later
                if "<persona_description>" in persona_text and "</persona_description>" in persona_text:
                    extracted_persona = persona_text.split("<persona_description>")[1].split("</persona_description>")[0].strip()
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
        # We process the worst performing examples to get the best gradients
        sorted_trajs = sorted(trajectories, key=lambda t: t.score)
        
        # KEY CHANGE: Reduce sample size slightly because we are adding LLM calls here. 
        # 5-8 detailed critiques are often better than 16 generic ones.
        selected_trajs = sorted_trajs[:8] 

        print(f"Generating dynamic critiques for {len(selected_trajs)} trajectories...")

        for traj in selected_trajs:
            # --- DYNAMIC JUDGE STEP ---

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
            Does the Generated Output sound like a "Receipt Summary" or a "Psychological Profile"?

            1. DEPTH CHECK: The Gold Truth often explains the *strategy* (e.g., "avoids sponsored ads," "shops offline for clothes"). Does the Generated Output miss this and just list items instead?
            2. INFERENCE FAILURE: Look at the "Reviews" behavior. Does the Generated output assume "Viewing Items = Liking Reviews"? The Gold Truth might say they *ignore* reviews. Did the model get this backwards?
            3. SPECIFICITY: If the Generated output says "User bought chicken and robot vacuums," it is failing. It should say "User prioritizes convenience and easy-prep meals."

            Provide a critique that forces the model to stop listing items and start analyzing decision-making strategies.
            """
           
                        
            # We use the teacher model (Qwen3) to generate the critique
            try:
                # Note: We use the raw model or wrapper depending on your setup. 
                # Since 'teacher_model' is your GEPACompatibleModel wrapper:
                specific_critique = teacher_model(critique_prompt)
            except Exception as e:
                print(f"Critique generation failed: {e}")
                specific_critique = "Improve alignment with the gold persona."

            # Construct the feedback string
            feedback = (
                f"Score: {traj.score:.3f}. "
                f"Critique: {specific_critique} "
                "Ensure the new prompt addresses these specific failures."
            )
            # ---------------------------

            rec = {
                "Inputs": {
                    "purchases": traj.purchases_str,
                    "gold_persona": traj.gold_persona,
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

    # Define the core instruction
    meta_prompt_core = """You are an expert Prompt Engineer for an e-commerce AI system.
Your goal is to optimize a "System Instruction" that converts a user's shopping history into a specific "Persona Description."

I will show you:
1. The CURRENT PROMPT being used.
2. A list of FAILURE CASES (User History -> Generated Persona -> Critique).

Your Task:
Analyze the Critical Feedback. Identify patterns in what the Current Prompt is missing.
Then, write a NEW, IMPROVED PROMPT that addresses these specific weaknesses.

CRITICAL GUIDELINES FOR THE NEW PROMPT:
1. BAN "ITEM LISTING": The new prompt must explicitly forbid listing specific purchases (e.g., "Do not mention specific brand names like 'Sharpie' or 'Robitussin' unless establishing a pattern").
2. FORCE "WHY" OVER "WHAT": The prompt must instruct the model to look at the *gaps* between clicks. (e.g., "If they viewed expensive items but bought the cheap one, infer Price Sensitivity, not Quality Seeking").
3. DETECT "NEGATIVE SIGNALS": Instruct the model that *viewing* an item but *not buying* it is a signal of rejection. If they view highly-rated items but buy a low-rated one, they likely disregard reviews.
4. SYNTHESIZE CATEGORIES: Instead of "Bought chicken and fries," require "Prefers convenience foods."

The new prompt should force the model to act like a Psychologist, not an Accountant.
"""

    # Construct the final input using dedent to strip code indentation
    meta_prompt_input = textwrap.dedent(f"""
    {meta_prompt_core}

    === HINT ===
    Consider adding a mandatory "Analysis Step" in the new prompt where the model must answer: 
    "What does the user's rejection of the other viewed items tell us about their priorities?"
    before writing the final description.
    
    
    === CURRENT PROMPT ===
    {current_prompt}
    
    === FAILURE ANALYSIS (Critiques from the Judge) ===
    {examples_str}
    
    === TASK ===
    Based on the critiques above, rewrite the "CURRENT PROMPT" to fix the recurring errors.
    Ensure the new prompt forces the model to verify its claims against the purchase history.
    Return ONLY the new prompt text, ready to be pasted into the system.
    """)

    # Call optimizer
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
    max_metric_calls=200, # <-- Set a budget
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
    