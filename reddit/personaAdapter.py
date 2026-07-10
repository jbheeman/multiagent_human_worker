# https://github.com/gepa-ai/gepa

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar
import json
import math
import re
import threading
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
from run_gepa_eval import run_evaluation, clean_transcript_for_judge
import random
from alignment import schwartz_alignment
from pvq import PVQ_DATA, score_pvq_value_means
from build_dd_seed import render_corpus
from persona_pipeline_datadesigner import PersonaProfile, persona_to_yaml

# ---------------------------------------------------------------------------
# Rework configuration (AAAI). GEPA_ARM is the ablation ladder: it selects the
# signal weights AND the reflection objective, so a single env var switches
# between the value_only / behavior_only / full arms. Inactive signals are skipped
# (no tau sim / utility / PVQ when their weight is 0); active ones enter the score
# and reflective feedback.
# ---------------------------------------------------------------------------
ARM = os.getenv("GEPA_ARM", "full")
ARM_CONFIG = {
    "value_only":    dict(w_align=1.0, w_tau=0.0, w_utility=0.0),
    "behavior_only": dict(w_align=0.0, w_tau=0.5, w_utility=0.5),
    "full":          dict(w_align=0.4, w_tau=0.3, w_utility=0.3),
}[ARM]
ARM_OBJECTIVE = {
    "value_only":    "A good persona reproduces the source user's value profile when surveyed.",
    "behavior_only": "A good persona behaves consistently in interactive tasks and predicts the user's held-out behavior.",
    "full":          "A good persona is grounded in the user's actual posts, reproduces their value profile when surveyed, and predicts their held-out behavior.",
}[ARM]

# ARM_CONFIG is the single source of truth for the combined-score weights.
W_ALIGN = ARM_CONFIG["w_align"]
W_TAU = ARM_CONFIG["w_tau"]
W_UTILITY = ARM_CONFIG["w_utility"]

USE_GROUNDING_GATE = bool(int(os.getenv("USE_GROUNDING_GATE", "1")))  # anti-hallucination gate
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")  # "cosine" | "spearman"

# Wall-clock / scoring knobs.
EVAL_WORKERS = int(os.getenv("EVAL_WORKERS", "8"))       # ThreadPoolExecutor width in evaluate()
GROUNDING_SKIP = float(os.getenv("GROUNDING_SKIP", "0.25"))  # skip tau sim below this grounding
UTILITY_K = int(os.getenv("UTILITY_K", "9"))            # distractors in the utility ranking
UTILITY_SMOOTH = bool(int(os.getenv("UTILITY_SMOOTH", "0")))  # smooth margin vs. rank fraction
TAU_JUDGE_SAMPLES = int(os.getenv("TAU_JUDGE_SAMPLES", "1"))  # median-of-N judge samples

# Per-arm persistence dir (created in __main__).
RUN_DIR = os.path.join("gepa_runs", ARM)

# Offline structural validation: stub every LLM/encoder call with deterministic
# values so the GEPA scoring pipeline runs without NAUT_API_KEY / heavy models.
MOCK_LLM = bool(os.getenv("MOCK_LLM"))

# Only transient network faults are retried; everything else propagates so real
# bugs (bad JSON, KeyErrors, assertion failures) surface loudly instead of being
# swallowed and slept on.
RETRYABLE_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    TimeoutError,
    ConnectionError,
)


def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator to retry a function with exponential backoff on transient network errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
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
        model_id="gpt-oss",
        model_id="gpt-oss",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
        client_kwargs={"http_client": http_client}
    )

#This model evaluates the persona description
teacher_model_raw= OpenAIServerModel( # Still used for persona agent
        model_id="gpt-oss",
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
        client_kwargs={"http_client": http_client}
    )

teacher_model = GEPACompatibleModel(teacher_model_raw)

@dataclass
class PersonaDataInst:
    # Per-user record (enriched schema, produced by enrich_users.py). One persona
    # per user, grounded in their attribution-labeled cross-subreddit corpus; the
    # Schwartz `target_vector` is the source-conditioned construct and
    # `heldout_post` is the behavioral-prediction target.
    user_id: str
    subreddits: list[str]            # all subreddits the user is active in
    clean_corpus: list[dict]         # [{"label": USER/INTERLOCUTOR/QUOTED, "text", "sub", "post_idx"}, ...]
    heldout_post: str                # held-out USER-turn text (behavioral target)
    quote_signals: list              # quoted/external content signals (often empty)
    target_vector: dict              # 10-dim Schwartz [0,1], re-inferred on held-out-removed corpus
    demographics: dict = field(default_factory=dict)  # omitted for the Reddit arm; kept for compat

    @property
    def shift_vector(self) -> dict:
        """Back-compat alias; the source-conditioned value vector."""
        return self.target_vector

    @property
    def posts(self) -> list[str]:
        """USER-turn texts only (the user's own posts, held-out already removed).
        Used for the grounding premise and the placeholder-injection assertion."""
        return [s["text"] for s in self.clean_corpus if s.get("label") == "USER"]

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
    valid: bool = True           # False if an ACTIVE signal hard-failed => excluded from reflection

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

GEPA_PARAGRAPH_PROMPT = """
You are an expert psychological profiler. From the user's behavioral corpus, write a
self-explanatory first-person portrait — psychologically vivid enough that someone
reading it could predict how this person communicates, argues, and reacts under pressure.

Do not write rigid rules ("never give my zip code"). Write the psychological reasoning
("I treat personal data like something to hoard because..."). Do not write dialogues,
transcripts, or invented transactional details.

USER CORPUS (speaker-attributed; [USER]/[INTERLOCUTOR]/[QUOTED] turns under [r/subreddit] headers):
{history_str}

CORPUS INSTRUCTIONS:
- Build the portrait ONLY from USER turns — character, register, disposition, triggers.
- INTERLOCUTOR and QUOTED turns are context for how this user argues and reacts; never
  attribute quoted views, beliefs, or biography to the portrait.
- Transfer how they write and argue, not their Reddit topics or subculture jargon as identity.
- REGISTER IS NON-NEGOTIABLE: match the source's actual vocabulary level, sentence length,
  and formatting. Do not upgrade their diction, smooth their syntax, or wrap the portrait in
  essay structure (headers, roman numerals, bolded thesis lines) unless the user actually
  writes that way -- that structure is YOUR default style leaking in, not theirs.
- If the source is blunt, profane, sarcastic, contemptuous, or impatient, the portrait MUST
  read that way: keep the profanity, the contempt, the impatience on the page. Sanding a
  hostile or crude user into an articulate, agreeable, or academic-sounding one is a FAILURE,
  not a stylistic choice.

LATENT VALUE PROFILE (ground the portrait's priorities and reasoning in this vector; treat it
as research you will never quote, not as vocabulary to use):
{psych_vector_str}

VALUE-NAME LEAK IS A FAILURE, NOT A STYLE CHOICE. Never write the literal Schwartz dimension
names or close derivatives -- in any form (capitalized, lowercase, or as an adjective/noun) --
including: power, achievement, hedonism/hedonistic, stimulation, self-direction, universalism,
benevolence, tradition, conformity, security. Do not print numbers, percentages, or
vector/profile language ("my X score", "my vector", "rates high on Y", "feeds my Z streak").
Express the SAME priorities only through what the person notices, wants, argues for, and
reacts to -- never through the label of the value itself. Before finishing, re-read every
sentence for one of the banned words above and rewrite it if found.
"""


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
                    subreddits=data.get("subreddits", []),
                    clean_corpus=data["clean_corpus"],
                    heldout_post=data.get("heldout_post", ""),
                    quote_signals=data.get("quote_signals", []),
                    target_vector=data["target_vector"],
                    demographics=data.get("demographics", {}),
                )
            )
    return examples


# Canonical placeholder set. Both the seed prompt and any GEPA-evolved prompt are
# rendered through render_prompt(); .replace() (not .format()) is used so evolved
# prompts may contain literal { } (e.g. JSON) without breaking substitution.
PLACEHOLDERS = {
    "{history_str}":         lambda d: render_corpus(d.clean_corpus),
    "{psych_vector_str}":    lambda d: json.dumps(d.shift_vector),
    "{subreddit}":           lambda d: ", ".join(d.subreddits),
    "{anchor_demographics}": lambda d: json.dumps(d.anchor_demographics),
}


def render_prompt(template: str, d: PersonaDataInst) -> str:
    """Fill the canonical placeholders and hard-assert the corpus actually landed
    in the prompt. Guards against the run-invalidating failure mode where a
    placeholder mismatch leaves the persona model generating from nothing."""
    out = template
    for ph, fn in PLACEHOLDERS.items():
        out = out.replace(ph, fn(d))
    # The user's corpus MUST be in the prompt.
    assert d.posts and d.posts[0][:40] in out, "history_str not injected — placeholder missing from prompt"
    # No unfilled placeholders may remain (Jinja-style or canonical).
    assert not re.search(
        r"\{\{\s*\w+\s*\}\}|\{(history_str|psych_vector_str|subreddit|anchor_demographics)\}",
        out,
    ), "unfilled placeholder remains after render"
    return out


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


# Mirrors the production compile step (persona_pipeline_datadesigner build_config
# step (b)): turn the free-text portrait into a schema-valid PersonaProfile so the
# tau sim runs on the same YAML the real pipeline feeds the user simulator.
PERSONA_COMPILE_PROMPT = """\
Compile this psychological portrait into a JSON object matching the schema EXACTLY.

ROLE: This persona is the person who contacts support to get something done — never
the support agent.
- example_utterances: 2-3 short lines this person would say when seeking help, in
  their voice from the portrait. Not agent-side troubleshooting.
- state_transition_rules: this person's POV — "IF the other party does X, THEN I
  react Y." (3 to 5 rules).
- escalation_trigger: what makes this person demand a human or walk away.

Do NOT set demographics (use null). Derive all fields from the portrait.

JSON SCHEMA:
{schema}

PORTRAIT:
{paragraph}

Return ONLY the JSON object — no prose, no markdown fences."""


class PersonaGEPAAdapter(GEPAAdapter[PersonaDataInst, PersonaTrajectory, str]):

    def __init__(self):
        super().__init__()
        # PVQ is re-administered for identical personas across minibatch re-evals;
        # cache successful results by persona text hash to avoid redundant calls.
        self._pvq_cache: dict[int, dict] = {}
        # Distractor pool for the utility ranking (built in __main__).
        self._candidates: list[dict] = []
        self._user_split: dict[str, str] = {}
        self._util_rng = random.Random(1234)
        self._distractor_tier_counts: dict[int, int] = {}
        # Per-arm run dir for persistence; set in __main__.
        self._run_dir: str | None = None
        self._log_lock = threading.Lock()

    # ---- Distractor pool for utility ranking -----------------------------
    def build_distractor_index(self, splits: dict[str, list["PersonaDataInst"]]):
        """Index candidate posts (USER history segments + heldouts) tagged with
        owner + split so the utility ranking can draw same-split, non-self
        distractors. Called once before optimization."""
        self._candidates = []
        self._user_split = {}
        for split, insts in splits.items():
            for d in insts:
                self._user_split[d.user_id] = split
                for seg in d.clean_corpus:
                    if seg.get("label") == "USER":
                        text = seg["text"]
                        self._candidates.append({
                            "text": text, "tok": len(text.split()),
                            "sub": seg.get("sub"), "owner": d.user_id,
                            "split": split, "hist": True,
                        })
                if d.heldout_post:
                    self._candidates.append({
                        "text": d.heldout_post, "tok": len(d.heldout_post.split()),
                        "sub": None, "owner": d.user_id, "split": split, "hist": False,
                    })

    def _sample_distractors(self, d: "PersonaDataInst", K: int | None = None) -> list[str]:
        """K same-split, non-self distractors with length parity to the true post.
        Tier 1: same-subreddit history posts; Tier 2: any posts; Tier 3: global
        (length filter dropped). Logs which fallback fired."""
        K = K or UTILITY_K
        if not self._candidates:
            return []
        split = self._user_split.get(d.user_id)
        true_tok = len(d.heldout_post.split()) if d.heldout_post else 0
        lo, hi = 0.5 * true_tok, 2.0 * true_tok
        subs = set(d.subreddits)

        def base(c):
            return c["owner"] != d.user_id and c["split"] == split

        def lenok(c):
            return true_tok == 0 or (lo <= c["tok"] <= hi)

        pool = [c for c in self._candidates if base(c) and c["hist"] and c["sub"] in subs and lenok(c)]
        tier = 1
        if len(pool) < K:
            pool = [c for c in self._candidates if base(c) and lenok(c)]
            tier = 2
        if len(pool) < K:
            pool = [c for c in self._candidates if base(c)]
            tier = 3
        if not pool:
            return []
        chosen = self._util_rng.sample(pool, min(K, len(pool)))
        self._distractor_tier_counts[tier] = self._distractor_tier_counts.get(tier, 0) + 1
        if tier > 1:
            print(f"Distractor fallback tier {tier} for {d.user_id} (pool={len(pool)})")
        return [c["text"] for c in chosen]

    # ---- Per-iteration trajectory persistence ----------------------------
    def _log_trajectory(self, rec: dict):
        if not self._run_dir:
            return
        with self._log_lock:
            with open(os.path.join(self._run_dir, "trajectories.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

    # PVQ-40 Items and Scoring Key
    # Scale: 1 (Not like me at all) to 6 (Very much like me)

    def _administer_pvq_test(self, persona_text: str) -> dict[str, float] | None:
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

        cache_key = hash(persona_text)
        if cache_key in self._pvq_cache:
            return self._pvq_cache[cache_key]

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
            # --- C. The Call (retried on transient network faults) ---
            @retry_with_backoff(max_retries=3, initial_delay=2.0)
            def _call():
                return persona_model([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])
            response = _call()

            # --- D. Parsing ---
            # Robust JSON extraction
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if not match:
                print("PVQ Failed: No JSON found in response.")
                return None

            item_scores = json.loads(match.group(0))

            # --- E. Scoring (Aggregating Items into Values) ---
            final_trait_scores = score_pvq_value_means(item_scores)
            self._pvq_cache[cache_key] = final_trait_scores
            return final_trait_scores

        except Exception as e:
            # Do NOT cache failures: a transient fault shouldn't poison re-evals.
            print(f"PVQ Critical Failure: {e}")
            return None
           
    
    def _score_schwartz_alignment(self, persona_text: str, target_vector: dict[str, float]) -> tuple[float | None, float, dict]:
        """Full-vector Schwartz alignment.

        Administers the PVQ to the persona, then compares the measured trait
        profile (scale 1-6) against the source-derived target vector ([0,1])
        across ALL 10 traits via mean-centered cosine (or Spearman). This
        is a consistency/leakage diagnostic used as a GEPA reward signal, not
        external validation. Returns (alignment_grade, mean_pvq, pvq_results).

        A None grade signals a HARD PVQ failure (parse/network exhausted) so the
        caller can exclude the trajectory instead of scoring it a noisy 0.0.
        """
        if not target_vector:
            return 0.5, 0.0, {}

        pvq_results = self._administer_pvq_test(persona_text)  # {SECURITY: 5.5, POWER: 2.1, ...}
        if not pvq_results:
            return None, 0.0, {}

        grade = schwartz_alignment(target_vector, pvq_results, metric=SIMILARITY_METRIC)
        mean_pvq = sum(pvq_results.values()) / len(pvq_results) if pvq_results else 0.0
        print(f"DEBUG: full-vector alignment ({SIMILARITY_METRIC}) = {grade:.3f}")
        return grade, mean_pvq, pvq_results
    
    def _call_tau_judge(self, formatted_prompt: str) -> dict | None:
        """Single deterministic judge call. Retries once on any failure; returns a
        parsed {'score','critique'} dict or None (never a fabricated 0/1 score)."""
        for attempt in range(2):
            try:
                raw_response = teacher_model(formatted_prompt, temperature=0)
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    if "score" in data:
                        return {"score": int(data["score"]), "critique": data.get("critique", "")}
                print(f"Tau judge attempt {attempt + 1}: no parseable score.")
            except Exception as e:
                print(f"Tau judge attempt {attempt + 1} failed: {e}")
        return None

    def _compile_persona_to_yaml(self, paragraph: str, user_id: str) -> str | None:
        """Mirror the production pipeline: paragraph -> structured PersonaProfile ->
        YAML. The tau2 user simulator is driven by that YAML (not the free-text
        paragraph), so the tau signal reflects the SAME artifact the real sims run
        on. Returns the YAML string, or None on hard failure so the trajectory is
        excluded rather than scored on a malformed spec."""
        schema = json.dumps(PersonaProfile.model_json_schema())
        prompt = PERSONA_COMPILE_PROMPT.format(schema=schema, paragraph=paragraph)
        for attempt in range(2):
            try:
                raw = _strip_fences(persona_model([{"role": "user", "content": prompt}]).content or "")
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    print(f"Persona compile attempt {attempt + 1}: no JSON object.")
                    continue
                data = json.loads(match.group(0))
                data["id"] = user_id  # id isn't in the portrait; stamp it deterministically
                profile = PersonaProfile.model_validate(data)
                return persona_to_yaml(profile.model_dump())
            except Exception as e:
                print(f"Persona compile attempt {attempt + 1} failed: {e}")
        return None

    def _score_persona_with_tau(self, persona_description: str, user_id: str = "") -> dict | None:
        """
        Runs Tau Bench and returns a dict with 'score' (1-5) and 'critique', or
        None on hard failure (so the trajectory is EXCLUDED from the reflective
        dataset rather than scored 0 and learned from as noise). Scores OBSERVABLE
        behavioral consistency, not value-recitation.

        The sim is driven by the compiled PersonaProfile YAML (production path); the
        judge scores the resulting transcript against the ORIGINAL paragraph — the
        GEPA artifact under optimization — so the signal is end-to-end: does the
        paragraph, once compiled and run, produce faithful behavior?
        """
        if MOCK_LLM:
            return {"score": 3, "critique": "mock tau result"}

        # 1. Compile to the YAML spec the real sims consume, then run the sim.
        persona_yaml = self._compile_persona_to_yaml(persona_description, user_id)
        if persona_yaml is None:
            return None  # can't build the spec the real sim would use -> exclude
        result = run_evaluation(persona_yaml)
        clean_transcript = clean_transcript_for_judge(result)

        # 2. Behavioral judge (P3): reward observable interaction quality, NOT the
        #    persona naming its own psychological values in the monologue. The judge
        #    first extracts concrete behavioral predictions from the persona, then
        #    verifies each against the transcript with a cited line, THEN scores —
        #    anchoring the 1-5 scale and making the critique a useful GEPA signal.
        TAU_ALIGNMENT_JUDGE_PROMPT = """
        You are an expert Evaluator of simulated customer-support interactions.

        Your Goal: Judge whether the User Simulator's OBSERVABLE BEHAVIOR (its
        requests, refusals, escalations, persistence, and tone in the dialogue)
        is consistent with the priorities implied by the Persona Description.

        Judge ONLY the dialogue acts and outcomes. Do NOT reward the user for
        naming psychological values or citing numbers in its internal monologue;
        a persona that merely announces its values but behaves inconsistently
        should score LOW.

        ### METHOD (do this before scoring)
        1. From the PERSONA, extract 3-4 concrete, checkable behavioral predictions
           (e.g. "would refuse to share personal data", "escalates after ~2 stonewalls",
           "uses short, profane, impatient sentences").
        2. For each prediction, find whether the TRANSCRIPT confirms or violates it,
           quoting the specific line that shows it (or noting "no evidence").
        3. Only then assign the overall score, justified by those per-prediction checks.

        ### SCORING CRITERIA (1-5)
        - **5 (Perfect):** Actions and tone consistently and specifically reflect the
          persona's priorities throughout; every prediction confirmed.
        - **4 (Strong):** Most predictions confirmed; at most one weakly supported or
          slightly off, none contradicted.
        - **3 (Passable):** Behavior broadly plausible but generic or only partially
          consistent; a mix of confirmed and unsupported predictions.
        - **2 (Weak):** Mostly generic or off-persona; at most one prediction confirmed,
          or the register is noticeably wrong.
        - **1 (Fail):** The user acts randomly, breaks character, or behaves in a way
          that contradicts the persona's priorities.

        ### INPUT DATA
        **PERSONA:**
        {persona_description}

        **TRANSCRIPT:**
        {clean_transcript}

        ### OUTPUT FORMAT
        You must return a valid JSON object with two fields:
        1. "score": An integer from 1 to 5.
        2. "critique": The per-prediction checks (prediction -> confirmed/violated +
           cited line) followed by a one-line justification of the score. Use this to
           guide future improvements.

        Example:
        {{
            "score": 4,
            "critique": "P1 'refuses to share data' -> confirmed ('I'm not giving you my zip'); P2 'escalates quickly' -> confirmed (demanded a manager turn 3); P3 'blunt/profane register' -> partial (curt but not profane). Behavior matches a privacy-guarding, low-trust persona; conceded slightly early."
        }}
        """

        formatted_prompt = TAU_ALIGNMENT_JUDGE_PROMPT.format(
            persona_description=persona_description,
            clean_transcript=clean_transcript,
        )

        # 3. Deterministic judging: single sample at temp 0, or median-of-N to beat
        #    residual sampling noise in the "bottom failures" the reflection reads.
        samples = []
        critique = None
        for _ in range(max(1, TAU_JUDGE_SAMPLES)):
            parsed = self._call_tau_judge(formatted_prompt)
            if parsed is not None:
                samples.append(parsed["score"])
                critique = parsed["critique"] or critique
        if not samples:
            return None
        median_score = sorted(samples)[len(samples) // 2]
        return {"score": median_score, "critique": critique or ""}


    def _grounding_score(self, persona_text: str, posts: list[str]) -> float:
        """Anti-hallucination gate in [0,1] via retrieval-then-NLI, CONTRADICTION-based.

        For each persona sentence we retrieve its top-3 most similar posts (bi-encoder)
        and run NLI against each. Fabrication surfaces as CONTRADICTION; psychological
        abstraction surfaces as NEUTRAL and must NOT be punished (an abstraction like
        "I hoard personal data" is never *entailed* by raw posts). So the gate is
        `1 - mean(hinge)` where a sentence only contributes when its contradiction
        probability exceeds 0.5. Retrieval also removes the 512-token premise-truncation
        confound of concatenating the whole corpus."""
        if MOCK_LLM or nli_model is None or eval_model is None:
            return 1.0
        if not persona_text or not posts:
            return 0.5

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", persona_text) if len(s.strip()) > 20]
        if not sentences:
            return 0.5
        sentences = sentences[:20]  # cap cost

        try:
            post_emb = eval_model.encode(posts, normalize_embeddings=True)
            sent_emb = eval_model.encode(sentences, normalize_embeddings=True)

            pairs: list[tuple[str, str]] = []
            pair_sent_idx: list[int] = []
            for si in range(len(sentences)):
                sims = sent_emb[si] @ post_emb.T  # cosine (normalized)
                top = np.argsort(-sims)[:3]
                for pi in top:
                    pairs.append((posts[int(pi)], sentences[si]))
                    pair_sent_idx.append(si)

            logits = np.asarray(nli_model.predict(pairs), dtype=float)
            # nli-deberta label order: [contradiction, entailment, neutral]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            contra = probs[:, 0]

            # Per-sentence contradiction = worst (max) over its retrieved evidence.
            per_sent = np.zeros(len(sentences))
            for pair_i, si in enumerate(pair_sent_idx):
                per_sent[si] = max(per_sent[si], float(contra[pair_i]))

            hinge = np.where(per_sent > 0.5, per_sent, 0.0)
            gate = 1.0 - float(hinge.mean())
            return max(0.0, min(1.0, gate))
        except Exception as e:
            print(f"Grounding score failure: {e}")
            return 0.5

    def _utility_score(self, persona_text: str, data_inst: "PersonaDataInst") -> float:
        """Held-out behavioral prediction as a RANKING task in [0,1]: prompt the
        persona (in character) to write its next post, then check whether that
        prediction sits closer to the user's TRUE held-out post than to K real
        distractor posts by other users. This restores dynamic range (raw cosine
        between two Reddit posts is compressed ~0.2-0.5) and answers the circularity
        charge — utility becomes retrieval against real data, not a similarity vibe."""
        if MOCK_LLM or eval_model is None:
            return 0.5
        true_post = data_inst.heldout_post
        if not true_post:
            return 0.0
        distractors = self._sample_distractors(data_inst)
        if not distractors:
            return 0.0

        subreddit = ", ".join(data_inst.subreddits)
        prompt = (
            f"You are role-playing the following person:\n{persona_text}\n\n"
            f"Write the post this person would most plausibly write next in one of "
            f"these communities: {subreddit}. Match their voice and concerns. "
            f"Output only the post text."
        )
        try:
            @retry_with_backoff(max_retries=3, initial_delay=2.0)
            def _call():
                return (persona_model([{"role": "user", "content": prompt}], temperature=0).content or "")
            predicted = _call()
            if not predicted.strip():
                return 0.0

            texts = [predicted, true_post] + distractors
            emb = eval_model.encode(texts, normalize_embeddings=True)
            pred_emb = emb[0]
            cos_true = float(pred_emb @ emb[1])
            cos_dist = [float(pred_emb @ emb[j]) for j in range(2, len(texts))]
            K = len(cos_dist)

            if UTILITY_SMOOTH:
                mean_dist = sum(cos_dist) / K
                return 1.0 / (1.0 + math.exp(-(cos_true - mean_dist)))
            # rank of the true post among [true] + distractors (1 = best)
            rank = 1 + sum(1 for c in cos_dist if c >= cos_true)
            return (K + 1 - rank) / K
        except Exception as e:
            print(f"Utility score failure: {e}")
            return 0.0

    def load_persona_dataset(self, path: str) -> list[PersonaDataInst]:
        return load_persona_dataset(path)

   
        
    
    def _evaluate_one(self, data_inst: PersonaDataInst, prompt_template: str) -> PersonaTrajectory:
        """Score a single instance. Only ACTIVE signals (weight > 0) are computed;
        inactive ones are skipped to avoid paying for tau sims / utility / PVQ on
        ablation arms. A hard failure of an active signal marks the trajectory
        invalid so it is excluded from the reflective dataset."""
        shift_vector = data_inst.shift_vector
        valid = True
        raw_pvq_val = 0.0
        agent_vector = None
        try:
            prompt = render_prompt(prompt_template, data_inst)

            @retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
            def call_persona_model():
                return persona_model([{"role": "user", "content": prompt}]).content or ""
            generated_persona = call_persona_model()

            # Grounding first: it gates the score AND lets us skip the expensive tau
            # sim on hopeless candidates (the gate is going to crush the score anyway).
            grounding = self._grounding_score(generated_persona, data_inst.posts)

            if W_ALIGN > 0:
                alignment_grade, raw_pvq_val, agent_vector = self._score_schwartz_alignment(
                    generated_persona, shift_vector
                )
                if alignment_grade is None:
                    valid = False
                    alignment_grade = 0.0
            else:
                alignment_grade, raw_pvq_val, agent_vector = 0.0, 0.0, None

            utility = (
                self._utility_score(generated_persona, data_inst)
                if W_UTILITY > 0 else 0.0
            )

            if W_TAU > 0 and grounding >= GROUNDING_SKIP:
                tau_result = self._score_persona_with_tau(generated_persona, data_inst.user_id)
            else:
                tau_result = None  # inactive arm or early-exit on low grounding
            if tau_result is None and W_TAU > 0 and grounding >= GROUNDING_SKIP:
                valid = False  # active tau signal genuinely failed (not an early-exit skip)
            normalized_tau = (float(tau_result["score"]) / 5.0) if tau_result else 0.0

            base = (W_ALIGN * alignment_grade
                    + W_TAU * normalized_tau
                    + W_UTILITY * utility)
            score = base * grounding if USE_GROUNDING_GATE else base

            print(f"[{data_inst.user_id}] align={alignment_grade:.2f} "
                  f"tau={normalized_tau:.2f} util={utility:.2f} ground={grounding:.2f} "
                  f"score={score:.2f} valid={valid}")

        except Exception as e:
            print(f"Error generating persona for {data_inst.user_id}: {e}")
            generated_persona = ""
            score = 0.0
            alignment_grade = 0.0
            utility = 0.0
            grounding = 0.0
            tau_result = {"score": 0, "critique": f"Error: {str(e)}"}
            valid = False

        self._log_trajectory({
            "user_id": data_inst.user_id, "alignment": alignment_grade,
            "tau": (float(tau_result["score"]) / 5.0) if tau_result else None,
            "utility": utility, "grounding": grounding,
            "score": score, "valid": valid, "arm": ARM,
        })

        return PersonaTrajectory(
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
            valid=valid,
        )

    def evaluate(
        self,
        batch: list[PersonaDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PersonaTrajectory, str]:
        prompt_template = candidate["persona_prompt"]

        # All per-instance work is I/O-bound API traffic -> parallelize. Order is
        # preserved by mapping over the batch and keeping results in index order.
        with ThreadPoolExecutor(max_workers=max(1, EVAL_WORKERS)) as ex:
            trajs = list(ex.map(lambda d: self._evaluate_one(d, prompt_template), batch))

        outputs = [t.generated_persona for t in trajs]
        scores = [t.combined_score for t in trajs]
        trajectories = trajs if capture_traces else None
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

        # Exclude invalid trajectories (an active signal hard-failed) so the
        # proposer never diagnoses API noise as a prompt weakness.
        trajectories = [t for t in (eval_batch.trajectories or []) if getattr(t, "valid", True)]
        records: list[dict[str, Any]] = []

        # Contrast improves reflection: show the worst failures AND a couple of
        # strong examples so the proposer sees what a good persona looks like under
        # this prompt instead of over-rotating on one failure mode.
        sorted_trajs = sorted(trajectories, key=lambda t: t.total_score)
        bottom = sorted_trajs[:4]
        top = sorted_trajs[-2:] if len(sorted_trajs) >= 6 else []
        selected = [("BOTTOM", t) for t in bottom] + [("TOP", t) for t in top]

        print(f"Generating diagnostic critiques for {len(selected)} trajectories "
              f"({len(bottom)} bottom + {len(top)} top)...")

        for label, traj in selected:
            # Only the ACTIVE signals for this arm enter the feedback + record, so
            # value_only / behavior_only optimize without behavioral fields in view.
            parts = []
            if W_ALIGN > 0:
                parts.append(f"Value alignment: {traj.schwartz_alignment_score:.2f}")
            if W_TAU > 0:
                tau_score = traj.tau_result.get("score", 0) if traj.tau_result else 0
                parts.append(f"Tau: {tau_score}/5")
            if W_UTILITY > 0:
                parts.append(f"Utility: {traj.utility_score:.2f}")
            parts.append(f"Grounding: {traj.grounding_score:.2f}")

            feedback = f"[{label}] Combined Score: {traj.combined_score:.2f} (" + ", ".join(parts) + ")\n"
            if W_TAU > 0 and traj.tau_result:
                feedback += f"Behavioral Judge Critique: {traj.tau_result.get('critique', '')}\n"
            feedback += "Note: low grounding => persona claims unsupported by the user's posts"
            if W_UTILITY > 0:
                feedback += "; low utility => persona fails to predict the user's held-out post"
            feedback += "."

            # The reflection model needs to SEE the corpus to diagnose a grounding
            # failure, so include a truncated snippet of the user's own posts.
            corpus_snippet = ("\n".join(traj.posts))[:500] if traj.posts else ""

            rec: dict[str, Any] = {
                "Inputs": {
                    "schwartz_vector": str(traj.target_vector),
                    "corpus_snippet": corpus_snippet,
                },
                "Generated Outputs": traj.generated_persona,
                "Feedback": feedback,
                "score": traj.combined_score,
            }
            if W_TAU > 0:
                rec["Tau Result"] = traj.tau_result
            records.append(rec)

        datasets["persona_prompt"] = records
        return datasets
        
REQUIRED_PLACEHOLDERS = ["{history_str}", "{psych_vector_str}"]


def _strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening ``` (possibly ```text)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _valid_proposal(text: str) -> bool:
    """A proposal is only usable if it kept the corpus/value placeholders; without
    them render_prompt() would inject nothing and every downstream iteration would
    optimize a corpus-free prompt (the classic GEPA placeholder-deletion failure)."""
    return all(ph in text for ph in REQUIRED_PLACEHOLDERS)


def _log_proposal(before: str, after: str):
    if not os.path.isdir(RUN_DIR):
        return
    with open(os.path.join(RUN_DIR, "proposals.jsonl"), "a") as f:
        f.write(json.dumps({"arm": ARM, "before": before, "after": after}) + "\n")


def custom_proposal_function(
    candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    components_to_update: list[str],
) -> dict[str, str]:

    current_prompt = candidate["persona_prompt"]
    cases = reflective_dataset.get("persona_prompt", [])

    # 1. Compile the evidence report (BOTTOM = failures to fix, TOP = keep working).
    examples_str = ""
    for i, case in enumerate(cases):
        inputs = case.get("Inputs", {})
        examples_str += f"\n--- CASE {i+1} ---\n"
        examples_str += f"Target Values: {inputs.get('schwartz_vector', '')}\n"
        examples_str += f"Corpus Snippet: {inputs.get('corpus_snippet', '')}\n"
        examples_str += f"Generated Persona: {case.get('Generated Outputs', '')}\n"
        if "Tau Result" in case:
            examples_str += f"Tau Result: {case['Tau Result']}\n"
        examples_str += f"CRITIQUE: {case.get('Feedback', '')}\n"

    # 2. Arm-aware meta-prompt with a HARD placeholder constraint.
    meta_prompt = f"""
    You are an AI System Architect optimizing a "Persona Profiler" System Prompt.

    THE OBJECTIVE:
    We are training a "Profiler" AI to write System Instructions for a "User Simulator" (Agent).
    The Agent must BEHAVE consistently with its source user's priorities in interactive
    customer-support tasks.
    {ARM_OBJECTIVE}
    The persona must express these priorities through behavior — WITHOUT ever naming
    psychological values or reciting numbers in its text. Reward behavior, not self-description.

    === EVIDENCE (labeled [BOTTOM] = failures to fix, [TOP] = strong examples to preserve) ===
    Read each CRITIQUE to understand the current weakness, and use the Corpus Snippet to
    judge whether the persona was grounded in the user's real posts.
    {examples_str}

    === YOUR TASK ===
    1. DIAGNOSE: Based on the evidence, what is the biggest current flaw in the System Prompt?
    2. OPTIMIZE: Rewrite the CURRENT PROMPT to fix it while preserving what makes the [TOP] cases work.

    HARD CONSTRAINT: The new prompt MUST contain these exact placeholder tokens, verbatim,
    exactly once each: {{history_str}}, {{psych_vector_str}}
    If a placeholder is missing your output will be rejected.

    === CURRENT PROMPT ===
    {current_prompt}

    === NEW OPTIMIZED PROMPT ===
    Return ONLY the full text of the new System Prompt. No diagnosis text, no markdown code fences.
    """

    print(f"Optimizing Prompt (arm={ARM}) based on Adaptive Diagnostics...")

    def _generate() -> str:
        return _strip_fences(teacher_model(meta_prompt, temperature=0.7))

    new_prompt_text = _generate()
    if not _valid_proposal(new_prompt_text):
        print("Proposal dropped a required placeholder; retrying once...")
        new_prompt_text = _generate()
    if not _valid_proposal(new_prompt_text):
        print("Proposal still invalid; falling back to the current candidate.")
        new_prompt_text = current_prompt

    _log_proposal(current_prompt, new_prompt_text)
    return {"persona_prompt": new_prompt_text}


def _stratified_sample(insts: list[PersonaDataInst], n: int, rng: random.Random) -> list[PersonaDataInst]:
    """Sample n instances spread across dominant Schwartz values so the GEPA
    minibatch sees diverse value profiles (diversity matters more than size).
    Deterministic given rng."""
    if n >= len(insts):
        out = list(insts)
        rng.shuffle(out)
        return out
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for d in insts:
        tv = d.target_vector or {}
        dom = max(tv, key=tv.get) if tv else "NONE"
        buckets[dom].append(d)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[PersonaDataInst] = []
    keys = sorted(buckets)
    i = 0
    while len(out) < n:
        advanced = False
        for k in keys:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
                advanced = True
                if len(out) >= n:
                    break
        if not advanced:
            break
        i += 1
    return out


if __name__ == "__main__":
    base_candidate = {"persona_prompt": GEPA_PARAGRAPH_PROMPT}

    TRAIN_N = int(os.getenv("TRAIN_N", "40"))
    VAL_N = int(os.getenv("VAL_N", "20"))
    MAX_METRIC_CALLS = int(os.getenv("MAX_METRIC_CALLS", "350"))

    TRAIN_FILE = os.getenv("TRAIN_FILE", "selected_users_pvq_gepa_train_k50.jsonl")
    # NOTE: the valset is what GEPA SELECTS on (Pareto front) -> it is contaminated
    # and is NOT a test set. The clean N=100 eval split (selected_users_pvq_eval_k100.jsonl)
    # is untouched here and reserved for E1/E2.
    VAL_FILE = os.getenv("VAL_FILE", "selected_users_pvq_gepa_test_k25.jsonl")

    trainset_full = load_persona_dataset(TRAIN_FILE)
    valset_full = load_persona_dataset(VAL_FILE)

    # Decoupled RNGs so changing TRAIN_N never perturbs the validation sample.
    train_rng = random.Random(42)
    val_rng = random.Random(43)
    trainset = _stratified_sample(trainset_full, min(TRAIN_N, len(trainset_full)), train_rng)
    valset = val_rng.sample(valset_full, min(VAL_N, len(valset_full)))

    adapter = PersonaGEPAAdapter()
    # Utility distractors come from the FULL splits (more real candidates), drawn
    # same-split and non-self at scoring time.
    adapter.build_distractor_index({"train": trainset_full, "val": valset_full})

    # Persistence: write config + seed prompt now, best prompt after the run.
    os.makedirs(RUN_DIR, exist_ok=True)
    adapter._run_dir = RUN_DIR
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump({
            "arm": ARM, "weights": ARM_CONFIG, "seed": 42,
            "train_n": len(trainset), "val_n": len(valset),
            "max_metric_calls": MAX_METRIC_CALLS,
            "train_file": TRAIN_FILE, "val_file": VAL_FILE,
        }, f, indent=2)
    with open(os.path.join(RUN_DIR, "seed_prompt.txt"), "w") as f:
        f.write(GEPA_PARAGRAPH_PROMPT)

    adapter.propose_new_texts = custom_proposal_function

    gepa_result = gepa.optimize(
        seed_candidate=base_candidate,
        trainset=trainset,
        valset=valset,
        max_metric_calls=MAX_METRIC_CALLS,   # budget (env: MAX_METRIC_CALLS)
        reflection_lm=teacher_model,         # strong model reflects + proposes
        adapter=adapter,
    )

    best = gepa_result.best_candidate
    with open(os.path.join(RUN_DIR, "best_prompt.txt"), "w") as f:
        f.write(best["persona_prompt"])
    print("\n=== Best persona prompt ===")
    print(best)
    print(f"Distractor tier usage: {adapter._distractor_tier_counts}")
    print(f"Artifacts written to {RUN_DIR}/")


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
