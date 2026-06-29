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
from tau_bench.run_gepa_eval import run_evaluation, clean_transcript_for_judge
import random
from alignment import schwartz_alignment
from pvq import PVQ_DATA, score_pvq_value_means

# ---------------------------------------------------------------------------
# Rework configuration (AAAI). These constants ARE the ablation ladder:
# toggling them and re-running GEPA produces the unoptimized / value-only /
# behavior-only / full arms. Weights are read at module load.
# ---------------------------------------------------------------------------
USE_TAU = bool(int(os.getenv("USE_TAU", "1")))            # behavioral interaction signal
USE_UTILITY = bool(int(os.getenv("USE_UTILITY", "1")))    # held-out behavioral prediction
USE_GROUNDING_GATE = bool(int(os.getenv("USE_GROUNDING_GATE", "1")))  # anti-hallucination gate
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")  # "cosine" | "spearman"

# Combined-score weights for the value / behavioral / utility signals.
W_ALIGN = float(os.getenv("W_ALIGN", "0.4"))
W_TAU = float(os.getenv("W_TAU", "0.3"))
W_UTILITY = float(os.getenv("W_UTILITY", "0.3"))

# Offline structural validation: stub every LLM/encoder call with deterministic
# values so the GEPA scoring pipeline runs without NAUT_API_KEY / heavy models.
MOCK_LLM = bool(os.getenv("MOCK_LLM"))

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



if MOCK_LLM:
    eval_model = None
    nli_model = None
else:
    eval_model = SentenceTransformer("all-mpnet-base-v2")
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')


# Custom HTTP client that skips SSL verification (for Nautilus SSL issues)
http_client = httpx.Client(verify=False)

#This model generates the persona description
persona_model = OpenAIServerModel(
        model_id="kimi",
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
    # Per-user record (v2 schema). One persona per user, grounded in their
    # cross-subreddit history; the Schwartz vector is the source-conditioned
    # construct and `heldout` is the behavioral-prediction target.
    user_id: str
    subreddits: list[str]            # all subreddits the user is active in
    history: list[dict]              # [{"subreddit": str, "posts": [str]}, ...] (held-out removed)
    demographics: dict               # {age, gender, occupation, location} (inferred upstream)
    heldout: dict                    # {"subreddit": str, "post": str} behavioral target
    target_vector: dict              # 10-dim Schwartz [0,1], re-inferred on held-out-removed corpus

    @property
    def shift_vector(self) -> dict:
        """Back-compat alias; the source-conditioned value vector."""
        return self.target_vector

    @property
    def posts(self) -> list[str]:
        """Flattened posts across subreddits (held-out post already removed)."""
        return [p for item in self.history for p in item.get("posts", [])]

    @property
    def anchor_demographics(self) -> dict:
        """Back-compat alias for the inferred demographics dict."""
        return self.demographics
@dataclass
class PersonaTrajectory:
    """Trajectory for one evaluation; must match constructor call in evaluate()."""
    user_id: str
    posts: list[str]
    subreddit: str
    anchor_demographics: str
    schwartz_alignment_score: float
    generated_persona: str
    raw_pvq_score: float
    shift_vector: dict | None = None
    agent_vector: dict | None = None  # PVQ results from _administer_pvq_test
    tau_result: dict | None = None  # Stores {'score': 4, 'critique': '...'}
    grounding_score: float = 1.0  # anti-hallucination gate (persona vs posts)
    utility_score: float = 0.0    # held-out behavioral prediction
    combined_score: float = 0.0  # weighted: value alignment + tau + utility, gated by grounding

    @property
    def total_score(self) -> float:
        """Used by make_reflective_dataset for sorting and feedback. Returns the combined score."""
        return self.combined_score

    @property
    def target_vector(self) -> dict | None:
        """Same as shift_vector (target psychological values)."""
        return self.shift_vector

    @property
    def schwartz_vector(self) -> dict | None:
        """Same as shift_vector."""
        return self.shift_vector



RolloutOutput = TypeVar("RolloutOutput") # the generated persona description
Trajectory = PersonaTrajectory
DataInst = PersonaDataInst
Candidate = dict[str, str]
EvaluatorFn = Callable[[list[DataInst], Candidate], tuple[list[RolloutOutput], list[float]]] # the evaluator function
UCSD_PERSONA_PROMPT = """
You are an expert Psychological Profiler.
Generate a persona definition that is self-explanatory. The persona description must be so coherent and psychologically vivid that an AI acting as this person will naturally deduce how to behave in any situation (Retail, Airline, Medical) purely by reading the description.

Do not write specific rules (e.g., 'Do not give zip code'). Instead, write the psychological reasoning (e.g., 'He is deeply skeptical of digital surveillance and treats personal data as a currency to be hoarded').

=== INPUT DATA ===
1. DEMOGRAPHIC ANCHOR:
{anchor_demographics}

2. INFERRED PSYCHOLOGICAL PROFILE (latent grounding from r/{subreddit}):
{psych_vector_str}
(Use this profile ONLY to shape behavior. Do NOT name these values or print any
numbers in your output — a coherent person never recites their own value scores.)

3. BEHAVIORAL SAMPLES:
{history_str}

=== OUTPUT FORMAT ===
You must output the persona in the following strict format:

### 1. CORE IDENTITY
(A first-person introduction: "I am a [Age] year old [Job]...")

### 2. PSYCHOLOGICAL DRIVERS
(A narrative explanation of *why* they act the way they do. Connect their background and
lived experience to their priorities — expressed in plain language, never as named
psychological values or numeric scores.)

### 3. INTERNAL MONOLOGUE STYLE
Describe how this person thinks and reasons under pressure (tempo, what they fixate on,
how they weigh risk vs. novelty vs. duty vs. others' needs). Then give exactly two
example internal thoughts, prefixed "Example 1: " and "Example 2: ", set in a retail
support interaction. The examples must reveal the person's priorities IMPLICITLY through
concrete reactions to a situation — they must NOT name any psychological value or cite
any number.

=== YOUR RESPONSE === """

def load_persona_dataset(path: str) -> list[PersonaDataInst]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst] = []
        for row in f:
            # Skip completely empty / whitespace-only lines to be robust to trailing newlines
            stripped = row.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            examples.append(
                PersonaDataInst(
                    user_id=data["user_id"],
                    subreddits=data.get("subreddits", [it.get("subreddit") for it in data.get("history", [])]),
                    history=data["history"],
                    demographics=data.get("demographics", {}),
                    heldout=data.get("heldout", {}),
                    target_vector=data["target_vector"],
                )
            )
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
        
        if MOCK_LLM:
            # Deterministic stub so the GEPA scoring pipeline runs offline.
            return {k: 3.5 for k in PVQ_DATA['mapping'].keys()}

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
            final_trait_scores = score_pvq_value_means(item_scores)
            print(f"Final Trait Scores: {final_trait_scores}")
            return final_trait_scores

        except Exception as e:
            print(f"PVQ Critical Failure: {e}")
            return {}
           
    
    def _score_schwartz_alignment(self, persona_text: str, target_vector: dict[str, float]) -> tuple[float, float, dict]:
        """Full-vector Schwartz alignment.

        Administers the PVQ to the persona, then compares the measured trait
        profile (scale 1-6) against the source-derived target vector ([0,1])
        across ALL 10 traits via mean-centered cosine (or Spearman). This
        is a consistency/leakage diagnostic used as a GEPA reward signal, not
        external validation. Returns (alignment_grade, mean_pvq, pvq_results).
        """
        if not target_vector:
            return 0.5, 0.0, {}

        pvq_results = self._administer_pvq_test(persona_text)  # {SECURITY: 5.5, POWER: 2.1, ...}
        if not pvq_results:
            return 0.0, 0.0, {}

        grade = schwartz_alignment(target_vector, pvq_results, metric=SIMILARITY_METRIC)
        mean_pvq = sum(pvq_results.values()) / len(pvq_results) if pvq_results else 0.0
        print(f"DEBUG: full-vector alignment ({SIMILARITY_METRIC}) = {grade:.3f}")
        return grade, mean_pvq, pvq_results
    
    def _score_persona_with_tau(self, persona_description: str) -> dict:
        """
        Runs Tau Bench and returns a dict with 'score' (1-5) and 'critique'.
        Scores OBSERVABLE behavioral consistency, not value-recitation.
        """
        if MOCK_LLM:
            return {"score": 3, "critique": "mock tau result"}

       # 1. Run Simulation
        result = run_evaluation(persona_description)
        clean_transcript = clean_transcript_for_judge(result)

        # 2. Behavioral judge (P3): reward observable interaction quality, NOT the
        #    persona naming its own psychological values in the monologue. This
        #    decouples the satisfaction signal from the value channel the persona
        #    writes into, which is the construct fix behind the human-validation gap.
        TAU_ALIGNMENT_JUDGE_PROMPT = """
        You are an expert Evaluator of simulated customer-support interactions.

        Your Goal: Judge whether the User Simulator's OBSERVABLE BEHAVIOR (its
        requests, refusals, escalations, persistence, and tone in the dialogue)
        is consistent with the priorities implied by the Persona Description.

        Judge ONLY the dialogue acts and outcomes. Do NOT reward the user for
        naming psychological values or citing numbers in its internal monologue;
        a persona that merely announces its values but behaves inconsistently
        should score LOW.

        ### SCORING CRITERIA (1-5)
        - **5 (Perfect):** The user's actions and tone consistently and specifically
          reflect the persona's priorities throughout the interaction.
        - **3 (Passable):** Behavior is broadly plausible but generic or only
          partially consistent with the persona.
        - **1 (Fail):** The user acts randomly, breaks character, or behaves in a
          way that contradicts the persona's priorities.

        ### INPUT DATA
        **PERSONA:**
        {persona_description}

        **TRANSCRIPT:**
        {clean_transcript}

        ### OUTPUT FORMAT
        You must return a valid JSON object with two fields:
        1. "score": An integer from 1 to 5.
        2. "critique": A specific analysis of the user's BEHAVIOR (what they did,
           refused, or escalated) and how well it matched the persona. Use this to
           guide future improvements.

        Example:
        {{
            "score": 4,
            "critique": "The user pushed back on the data request and escalated when stonewalled, consistent with a privacy-guarding, low-trust persona; but they conceded too quickly at the end."
        }}
        """

        formatted_prompt = TAU_ALIGNMENT_JUDGE_PROMPT.format(
            persona_description=persona_description, 
            clean_transcript=clean_transcript
        )
        
        # 3. Get Response and Parse JSON
        raw_response = teacher_model(formatted_prompt)
        
        try:
            # Extract JSON if the model wraps it in markdown blocks
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return data # Returns {'score': 4, 'critique': '...'}
            else:
                # Fallback if model fails to output JSON
                return {"score": 1, "critique": f"Failed to parse Judge output: {raw_response}"}
                
        except Exception as e:
            return {"score": 0, "critique": f"Judge Error: {str(e)}"}
        
        
       
        





    def _grounding_score(self, persona_text: str, posts: list[str]) -> float:
        """Anti-hallucination gate in [0,1]: are the persona's claims supported by
        the user's actual posts? Uses the NLI cross-encoder (entailment minus
        contradiction of persona sentences against the post corpus)."""
        if MOCK_LLM or nli_model is None:
            return 1.0
        if not persona_text or not posts:
            return 0.5

        premise = " ".join(posts)[:2000]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", persona_text) if len(s.strip()) > 20]
        if not sentences:
            return 0.5
        sentences = sentences[:20]  # cap cost

        try:
            logits = nli_model.predict([(premise, s) for s in sentences])
            logits = np.asarray(logits, dtype=float)
            # nli-deberta label order: [contradiction, entailment, neutral]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            per_sentence = probs[:, 1] - probs[:, 0]  # entail - contra in [-1,1]
            grade = float((per_sentence.mean() + 1.0) / 2.0)
            return max(0.0, min(1.0, grade))
        except Exception as e:
            print(f"Grounding score failure: {e}")
            return 0.5

    def _utility_score(self, persona_text: str, heldout: dict) -> float:
        """Held-out behavioral prediction in [0,1]: prompt the persona (in
        character) to react to the held-out post's CONTEXT, then measure semantic
        similarity between its predicted reaction and the true held-out post."""
        if MOCK_LLM or eval_model is None:
            return 0.5
        if not heldout or not heldout.get("post"):
            return 0.0

        subreddit = heldout.get("subreddit", "")
        true_post = heldout["post"]
        prompt = (
            f"You are role-playing the following person:\n{persona_text}\n\n"
            f"Write the post this person would most plausibly write in r/{subreddit}. "
            f"Match their voice and concerns. Output only the post text."
        )
        try:
            @retry_with_backoff(max_retries=3, initial_delay=2.0)
            def _call():
                return (persona_model([{"role": "user", "content": prompt}]).content or "")
            predicted = _call()
            if not predicted.strip():
                return 0.0
            emb = eval_model.encode([predicted, true_post], normalize_embeddings=True)
            sim = float(np.dot(emb[0], emb[1]))  # cosine in [-1,1]
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception as e:
            print(f"Utility score failure: {e}")
            return 0.0

    def load_persona_dataset(self, path: str) -> list[PersonaDataInst]:
        return load_persona_dataset(path)

   
        
    
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
        total = len(batch)

        prompt_template = candidate["persona_prompt"]
        for i,data_inst in enumerate(batch):
            print(f"Evaluating {i+1}/{total}...")
            traits = []
            persona_text = ""
            score = 0.0
            grounding_score = 0.0
            schwartz_alignment_score = 0.0

            shift_vector = data_inst.shift_vector
            
            try:
               
                demographics_str = json.dumps(data_inst.anchor_demographics)
                reddit_context = ", ".join(data_inst.subreddits)
                psych_vector_str = ", ".join([f"{k}: {v:.2f}" for k, v in shift_vector.items()])
                history_str = "\n---\n".join(data_inst.posts)
                # Substitute only our placeholders; GEPA-evolved prompts may contain literal { } (e.g. JSON) which would break .format()
                prompt = prompt_template.replace("{anchor_demographics}", demographics_str).replace("{subreddit}", reddit_context).replace("{psych_vector_str}", psych_vector_str).replace("{history_str}", history_str)
                @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
                def call_persona_model():
                    response_message = persona_model([{"role": "user", "content": prompt}])
                    return response_message.content or ""
                raw_output = call_persona_model()
                generated_persona = raw_output
              
                # --- Value alignment (full-vector), behavioral tau, and held-out utility ---
                alignment_grade, raw_pvq_val, agent_vector = self._score_schwartz_alignment(generated_persona, shift_vector)

                if USE_TAU:
                    tau_result = self._score_persona_with_tau(generated_persona)
                else:
                    tau_result = {"score": 0, "critique": "tau disabled (ablation)"}
                tau_scalar = float(tau_result.get("score", 0))
                normalized_tau = tau_scalar / 5.0

                utility = self._utility_score(generated_persona, data_inst.heldout) if USE_UTILITY else 0.0
                grounding = self._grounding_score(generated_persona, data_inst.posts)

                # Weighted objective; grounding acts as an anti-hallucination GATE,
                # not an additive term (a fabricated persona is penalized, not rewarded).
                base = (W_ALIGN * alignment_grade
                        + W_TAU * normalized_tau
                        + W_UTILITY * utility)
                score = base * grounding if USE_GROUNDING_GATE else base

                print(f"✅ Evaluation Complete:")
                print(f"   - Value alignment: {alignment_grade:.2f}")
                print(f"   - Tau: {tau_scalar}/5 ({normalized_tau:.2f}) | Utility: {utility:.2f} | Grounding: {grounding:.2f}")
                print(f"   - Combined Score: {score:.2f}")

            except Exception as e:
                print(f"Error generating persona: {e}")
                generated_persona = ""
                score = 0.0
                alignment_grade = 0.0
                raw_pvq_val = 0.0
                agent_vector = None
                utility = 0.0
                grounding = 0.0
                tau_result = {"score": 0, "critique": f"Error: {str(e)}"}  # Default failure dict

            outputs.append(generated_persona)
            scores.append(score)
            if capture_traces:
                trajectories.append(
                    PersonaTrajectory(
                        user_id=data_inst.user_id,
                        posts=data_inst.posts,
                        subreddit=", ".join(data_inst.subreddits),
                        anchor_demographics=str(data_inst.anchor_demographics),
                        schwartz_alignment_score=alignment_grade,
                        generated_persona=generated_persona,
                        raw_pvq_score=raw_pvq_val,
                        shift_vector=shift_vector,
                        agent_vector=agent_vector,
                        tau_result=tau_result,
                        grounding_score=grounding,
                        utility_score=utility,
                        combined_score=score,
                    )
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

            
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        
        datasets: dict[str, list[dict[str, Any]]] = {}
        if "persona_prompt" not in components_to_update:
            return datasets

        trajectories = eval_batch.trajectories or []
        records: list[dict[str, Any]] = []

        # Sort by Combined Score (Lowest = Needs Improvement)
        # Uses total_score property which returns combined_score (50% PVQ + 50% Tau)
        sorted_trajs = sorted(trajectories, key=lambda t: t.total_score)
        
        # Focus on the bottom 5 failures
        selected_trajs = sorted_trajs[:5] 

        print(f"Generating diagnostic critiques for {len(selected_trajs)} trajectories...")

        for traj in selected_trajs:
            # Extract scores and critique
            tau_score = traj.tau_result.get("score", 0) if traj.tau_result else 0
            judge_critique = traj.tau_result.get("critique", "No critique available.") if traj.tau_result else "No critique"
            pvq_alignment = traj.schwartz_alignment_score
            combined = traj.combined_score

            # Surface ALL signals so the optimizer doesn't optimize blind to
            # grounding/utility (R5).
            feedback = (
                f"Combined Score: {combined:.2f} "
                f"(Value alignment: {pvq_alignment:.2f}, Tau: {tau_score}/5, "
                f"Utility: {traj.utility_score:.2f}, Grounding: {traj.grounding_score:.2f})\n"
                f"Behavioral Judge Critique: {judge_critique}\n"
                f"Note: low grounding => persona claims unsupported by the user's posts; "
                f"low utility => persona fails to predict the user's held-out post."
            )

            rec = {
                "Inputs": {
                    "schwartz_vector": str(traj.target_vector),
                    # Pass whatever inputs generated this persona
                },
                "Generated Outputs": traj.generated_persona,
                "Tau Result": traj.tau_result,
                "PVQ Alignment": pvq_alignment,
                "Tau Score": tau_score,
                "Utility Score": traj.utility_score,
                "Grounding Score": traj.grounding_score,
                "Combined Score": combined,
                "Feedback": feedback,
                "score": combined,  # Use combined score for GEPA optimization
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
    
    # 1. Compile the Failure Report (Same as before)
    examples_str = ""
    for i, fail in enumerate(failures):
        examples_str += f"\n--- FAILURE CASE {i+1} ---\n"
        examples_str += f"Target Values: {fail['Inputs']['schwartz_vector']}\n"
        examples_str += f"Generated Persona: {fail['Generated Outputs']}\n"
        examples_str += f"Tau Result: {fail['Tau Result']}\n"
        examples_str += f"JUDGE CRITIQUE: {fail['Feedback']}\n"  # <--- This is the source of truth

    # 2. The "Adaptive" Meta-Prompt
    meta_prompt = f"""
    You are an AI System Architect optimizing a "Persona Profiler" System Prompt.
    
    THE OBJECTIVE:
    We are training a "Profiler" AI to write System Instructions for a "User Simulator" (Agent).
    The Agent must BEHAVE consistently with its source user's priorities in a Retail Environment.
    A good persona is grounded in the user's actual posts (no fabrication), reproduces their
    value profile when surveyed, and predicts their held-out behavior — WITHOUT ever naming
    psychological values or reciting numbers in its text. Reward behavior, not self-description.
    
    === EVIDENCE OF FAILURE ===
    Below are recent cases where the current prompt failed to produce good results. 
    Read the "JUDGE CRITIQUE" for each case to understand the current weakness.
    {examples_str}
    
    === YOUR TASK ===
    1. **DIAGNOSE:** Based on the evidence above, what is the *current* biggest flaw in the System Prompt? (e.g., Is it too vague? Too verbose? Ignoring values? Hallucinating?)
    2. **OPTIMIZE:** Rewrite the "CURRENT PROMPT" to fix this specific diagnosis.
    
    Your goal is to satisfy the Judge (who wrote the critiques) by addressing their specific complaints.
    
    === CURRENT PROMPT ===
    {current_prompt}
    
    === NEW OPTIMIZED PROMPT ===
    Return ONLY the full text of the new System Prompt. Do not include the diagnosis text or markdown blocks.
    """

    print("Optimizing Prompt based on Adaptive Diagnostics...")
    new_prompt_text = teacher_model(meta_prompt, temperature=0.7)
    
    return {"persona_prompt": new_prompt_text}


if __name__ == "__main__":
    #test loading 1 user and their purchases

    
    
    
    
    base_candidate = {
    "persona_prompt": UCSD_PERSONA_PROMPT
}


    TRAIN_N = int(os.getenv("TRAIN_N", "8"))
    VAL_N = int(os.getenv("VAL_N", "6"))
    MAX_METRIC_CALLS = int(os.getenv("MAX_METRIC_CALLS", "150"))

    trainset_full = load_persona_dataset(os.getenv("TRAIN_FILE", "train_reddit_v2.jsonl"))
    random.seed(42)
    trainset = random.sample(trainset_full, min(TRAIN_N, len(trainset_full)))

    valset_full = load_persona_dataset(os.getenv("VAL_FILE", "val_reddit_v2.jsonl"))
    # Reproducible validation set across iterations.
    valset = random.sample(valset_full, min(VAL_N, len(valset_full)))
    random.seed()  # Reset so future sampling is random
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
    max_metric_calls=MAX_METRIC_CALLS, # <-- budget (env: MAX_METRIC_CALLS)
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
