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
from run_gepa_eval import run_evaluation, clean_transcript_for_judge, select_task
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
# Piecewise gate: g >= FULL → multiplier 1.0 (kill noise in the clean band);
# SKIP <= g < FULL → g/FULL; g < SKIP → tau early-exit + still g/FULL crush.
GROUNDING_FULL = float(os.getenv("GROUNDING_FULL", "0.7"))
UTILITY_K = int(os.getenv("UTILITY_K", "9"))            # distractors in the utility ranking
UTILITY_SMOOTH = bool(int(os.getenv("UTILITY_SMOOTH", "1")))  # sigmoid margin (continuous); 0 = rank lumps
# Tau continuous score = (1-BLEND)*(verified/n) + BLEND*(anchor/5). Judge is temp-0
# so multi-sample medians are identical — single call only.
TAU_ANCHOR_BLEND = float(os.getenv("TAU_ANCHOR_BLEND", "0.5"))
TAU_JUDGE_SAMPLES = int(os.getenv("TAU_JUDGE_SAMPLES", "1"))  # unused (temp-0); kept for env compat
# Paired evaluation: fixed stratified panel reused for every candidate, k persona
# samples averaged per (candidate, user). GEPA's subsample gate then compares the
# same users; optional sign-test rejects challengers that don't win a majority of
# per-user deltas (stricter than sum-of-deltas).
EVAL_PANEL_N = int(os.getenv("EVAL_PANEL_N", "5"))
PERSONA_SAMPLES_K = int(os.getenv("PERSONA_SAMPLES_K", "2"))
PAIRED_SIGN_TEST = bool(int(os.getenv("PAIRED_SIGN_TEST", "1")))

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


def grounding_multiplier(g: float) -> float:
    """Anti-hallucination gate as a piecewise multiplier, not a continuous objective.

    g >= GROUNDING_FULL (0.7) → 1.0  (clean band: no differential rescaling)
    g <  GROUNDING_FULL       → g/FULL (ramp; g < SKIP also early-exits tau)
    """
    if not USE_GROUNDING_GATE:
        return 1.0
    if g >= GROUNDING_FULL:
        return 1.0
    return g / GROUNDING_FULL


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
        api_base="https://ellm.nrp-nautilus.io/v1",
        api_key=os.getenv("NAUT_API_KEY"),
        client_kwargs={"http_client": http_client}
    )

#This model evaluates the persona description
teacher_model_raw= OpenAIServerModel( # Still used for persona agent
        model_id="minimax-m2",
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
    combined_score: float = 0.0  # weighted align+tau+utility, piecewise-gated by grounding
    valid: bool = True           # False if an ACTIVE signal API-failed => excluded from reflection
                                 # (compile defects stay valid so GEPA can learn from them)

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
        # Sampling RNG is derived per user_id (not a shared Random) so the
        # distractor set is a pure function of the user — frozen across candidate
        # evals and thread-safe under ThreadPoolExecutor.
        self._candidates: list[dict] = []
        self._user_split: dict[str, str] = {}
        self._distractor_tier_counts: dict[int, int] = {}
        # Per-arm run dir for persistence; set in __main__.
        self._run_dir: str | None = None
        self._log_lock = threading.Lock()
        # Paired acceptance: scores from the last capture_traces=True panel eval
        # (incumbent). The next same-user capture_traces=False eval is the
        # challenger; optional sign-test gate forces rejection if it doesn't win
        # a majority of per-user deltas. Cleared after the challenger eval so
        # full-valset scoring is unaffected.
        self._incumbent_panel_scores: dict[str, float] | None = None
        self._incumbent_panel_ids: list[str] | None = None

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
        (length filter dropped). Logs which fallback fired.

        Distractor set is frozen per user: RNG seed is `1234:{user_id}`, so the
        same user always draws the same set across candidate evaluations and
        threads (no shared-RNG interleaving noise in the utility ranking).
        """
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
        rng = random.Random(f"1234:{d.user_id}")
        chosen = rng.sample(pool, min(K, len(pool)))
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
        """Single deterministic (temp-0) judge call. Retries once on failure.

        Expects JSON with per-prediction verdicts + optional 1-5 anchor. Returns a
        dict with continuous `score` in [0, 5] = 5 * blend(verified/n, anchor/5),
        plus critique / verified_frac for reflection — or None on hard failure.
        """
        for attempt in range(2):
            try:
                raw_response = teacher_model(formatted_prompt, temperature=0)
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if not match:
                    print(f"Tau judge attempt {attempt + 1}: no parseable JSON.")
                    continue
                data = json.loads(match.group(0))
                parsed = self._tau_score_from_judge_json(data)
                if parsed is not None:
                    return parsed
                print(f"Tau judge attempt {attempt + 1}: missing predictions/anchor.")
            except Exception as e:
                print(f"Tau judge attempt {attempt + 1} failed: {e}")
        return None

    @staticmethod
    def _tau_score_from_judge_json(data: dict) -> dict | None:
        """Map structured judge JSON → continuous tau score in [0, 5].

        Primary granularity: verified/total over per-prediction verdicts
        (confirmed=1, partial=0.5, else 0). Optionally blend with the 1-5 anchor
        so the ordinal scale still regularizes wild verification counts.
        """
        preds = data.get("predictions")
        anchor_raw = data.get("anchor_score", data.get("score"))
        critique = data.get("critique", "")

        frac = None
        if isinstance(preds, list) and preds:
            verified = 0.0
            for p in preds:
                if not isinstance(p, dict):
                    continue
                verdict = str(p.get("verdict", p.get("status", ""))).lower()
                if verdict in ("confirmed", "confirm", "yes", "true"):
                    verified += 1.0
                elif verdict in ("partial", "partially", "weak"):
                    verified += 0.5
            frac = verified / len(preds)

        anchor_01 = None
        if anchor_raw is not None:
            try:
                anchor_01 = max(0.0, min(5.0, float(anchor_raw))) / 5.0
            except (TypeError, ValueError):
                anchor_01 = None

        if frac is None and anchor_01 is None:
            return None
        if frac is None:
            score_01 = anchor_01
        elif anchor_01 is None:
            score_01 = frac
        else:
            b = max(0.0, min(1.0, TAU_ANCHOR_BLEND))
            score_01 = (1.0 - b) * frac + b * anchor_01

        if not critique and isinstance(preds, list):
            bits = []
            for i, p in enumerate(preds, 1):
                if not isinstance(p, dict):
                    continue
                bits.append(
                    f"P{i} {p.get('prediction', '?')!r} -> {p.get('verdict', p.get('status', '?'))}"
                    f" ({p.get('cite', 'no cite')})"
                )
            critique = "; ".join(bits)

        return {
            "score": float(score_01) * 5.0,  # keep /5.0 normalization downstream
            "critique": critique or "",
            "verified_frac": frac,
            "anchor_score": (anchor_01 * 5.0) if anchor_01 is not None else None,
        }

    def _compile_persona_to_yaml(self, paragraph: str, user_id: str) -> str | None:
        """Mirror the production pipeline: paragraph -> structured PersonaProfile ->
        YAML. The tau2 user simulator is driven by that YAML (not the free-text
        paragraph), so the tau signal reflects the SAME artifact the real sims run
        on. Returns the YAML string, or None if the paragraph cannot compile to a
        schema-valid PersonaProfile (a prompt-attributable defect)."""
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
        Runs Tau Bench and returns a dict with continuous `score` in [0, 5]
        (primarily verified/total of structured predictions, blended with a 1-5
        anchor) and `critique`.

        Failure modes (caller must treat differently):
          * Compile failure → {'score': 0, 'critique': 'failed to compile...'}.
            Prompt-attributable: keep in the score as tau=0 and in reflection.
          * API / sim / judge failure → None. Transient noise: exclude from
            reflection and drop the tau term from the renormalized GEPA score.

        Scores OBSERVABLE behavioral consistency, not value-recitation. The sim is
        driven by the compiled PersonaProfile YAML (production path); the judge
        scores the transcript against the ORIGINAL paragraph — the GEPA artifact
        under optimization.
        """
        if not user_id:
            raise ValueError("user_id is required for deterministic tau task assignment")
        if MOCK_LLM:
            return {"score": 3.0, "critique": "mock tau result", "verified_frac": 0.6, "anchor_score": 3.0}

        # 1. Compile to the YAML spec the real sims consume, then run the sim.
        #    Task is hash(user_id) over the mock pool — same user always gets the
        #    same task so candidate prompts are compared on identical pairs.
        persona_yaml = self._compile_persona_to_yaml(persona_description, user_id)
        if persona_yaml is None:
            return {
                "score": 0,
                "critique": "failed to compile to PersonaProfile schema",
            }
        try:
            task = select_task(user_id)
            result = run_evaluation(persona_yaml, user_id)
            clean_transcript = clean_transcript_for_judge(result)
        except Exception as e:
            print(f"Tau sim API failure for {user_id}: {e}")
            return None

        # Impossible-task guard: request cannot be fulfilled with available tools;
        # don't let the judge treat correct frustration/escalation as a miss.
        task_note = ""
        if str(getattr(task, "id", "")).startswith("impossible_task_"):
            task_note = (
                "\n        **TASK CONTEXT:** The user's request cannot be fulfilled "
                "with the available tools; appropriate user behavior ranges from "
                "acceptance to escalation depending on the profile.\n"
            )

        # 2. Behavioral judge (P3): reward observable interaction quality, NOT the
        #    persona naming its own psychological values in the monologue. The judge
        #    extracts concrete behavioral predictions and verifies each against the
        #    transcript; the GEPA score is primarily verified/total (continuous),
        #    blended with a 1-5 anchor so the ordinal scale still regularizes.
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
        3. Only then assign an overall anchor_score (1-5), justified by those checks.

        ### ANCHOR SCALE (1-5) — for anchor_score only
        - **5 (Perfect):** Every prediction confirmed.
        - **4 (Strong):** Most predictions confirmed; at most one weakly supported;
          none contradicted.
        - **3 (Passable):** Mix of confirmed and unsupported; broadly plausible.
        - **2 (Weak):** At most one prediction confirmed, or wrong register.
        - **1 (Fail):** Random, breaks character, or contradicts the persona.

        ### INPUT DATA
        **PERSONA:**
        {persona_description}
        {task_note}
        **TRANSCRIPT:**
        {clean_transcript}

        ### OUTPUT FORMAT
        Return a valid JSON object with:
        1. "predictions": array of 3-4 objects, each with:
           - "prediction": the checkable claim
           - "verdict": one of "confirmed" | "partial" | "violated" | "no_evidence"
           - "cite": quoted transcript line, or "no evidence"
        2. "anchor_score": integer 1-5 from the scale above
        3. "critique": brief justification tying verdicts to the anchor

        Example:
        {{
            "predictions": [
                {{"prediction": "refuses to share personal data", "verdict": "confirmed", "cite": "I'm not giving you my zip"}},
                {{"prediction": "escalates quickly", "verdict": "confirmed", "cite": "demanded a manager turn 3"}},
                {{"prediction": "blunt/profane register", "verdict": "partial", "cite": "curt but not profane"}}
            ],
            "anchor_score": 4,
            "critique": "2 confirmed + 1 partial; privacy-guarding low-trust persona, conceded slightly early."
        }}
        """

        formatted_prompt = TAU_ALIGNMENT_JUDGE_PROMPT.format(
            persona_description=persona_description,
            task_note=task_note,
            clean_transcript=clean_transcript,
        )

        # Single temp-0 call (N>1 is identical under greedy decoding).
        return self._call_tau_judge(formatted_prompt)


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

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", persona_text) if len(s.strip()) > 35]
        if not sentences:
            return 0.5
        sentences = sentences[:35]  # cap cost

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
        """Held-out behavioral prediction in [0,1]: persona writes a next post;
        compare embedding similarity to the TRUE held-out post vs K real distractors.

        Default (UTILITY_SMOOTH=1): continuous sigmoid of (cos_true - mean_cos_dist).
        UTILITY_SMOOTH=0: discrete rank fraction with 1/K lumps."""
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
        ablation arms.

        API/judge failures of an active signal mark valid=False (excluded from
        reflection) and DROP that term from the score returned to GEPA
        (renormalize over succeeded signals). Compile-to-PersonaProfile failure
        is prompt-attributable: tau=0 with a critique, kept in score + reflection.
        """
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

            # Grounding first: piecewise gate on the final score, and early-exit
            # the expensive tau sim when g < GROUNDING_SKIP (hopeless / hallucinated).
            grounding = self._grounding_score(generated_persona, data_inst.posts)

            # terms: (weight, score) for signals that count toward the GEPA score.
            # API-failed signals are omitted here (and set valid=False); compile
            # failures enter as score 0 so the prompt is penalized and reflected on.
            terms: list[tuple[float, float]] = []

            if W_ALIGN > 0:
                alignment_grade, raw_pvq_val, agent_vector = self._score_schwartz_alignment(
                    generated_persona, shift_vector
                )
                if alignment_grade is None:
                    valid = False  # PVQ API/parse failure — drop term, exclude reflection
                    alignment_grade = 0.0
                else:
                    terms.append((W_ALIGN, alignment_grade))
            else:
                alignment_grade, raw_pvq_val, agent_vector = 0.0, 0.0, None

            if W_UTILITY > 0:
                utility = self._utility_score(generated_persona, data_inst)
                terms.append((W_UTILITY, utility))
            else:
                utility = 0.0

            tau_result = None
            normalized_tau = 0.0
            if W_TAU > 0 and grounding >= GROUNDING_SKIP:
                tau_result = self._score_persona_with_tau(generated_persona, data_inst.user_id)
                if tau_result is None:
                    valid = False  # sim/judge API failure — drop term, exclude reflection
                else:
                    normalized_tau = float(tau_result["score"]) / 5.0
                    terms.append((W_TAU, normalized_tau))  # includes compile-fail at 0
            elif W_TAU > 0:
                # Deliberate low-grounding skip: count tau as 0 (don't renormalize away).
                terms.append((W_TAU, 0.0))

            if terms:
                w_sum = sum(w for w, _ in terms)
                base = sum(w * s for w, s in terms) / w_sum
            else:
                base = 0.0
            g_mult = grounding_multiplier(grounding)
            score = base * g_mult

            print(f"[{data_inst.user_id}] align={alignment_grade:.2f} "
                  f"tau={normalized_tau:.2f} util={utility:.2f} ground={grounding:.2f} "
                  f"g_mult={g_mult:.2f} score={score:.2f} valid={valid} "
                  f"terms={[(w, round(s, 2)) for w, s in terms]}")

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

    def _evaluate_one_k(self, data_inst: PersonaDataInst, prompt_template: str) -> PersonaTrajectory:
        """k persona generations per (candidate, user); return traj with averaged score.

        Persona gen is the remaining stochastic source (sim/judge/agent are temp-0).
        Averaging k samples cuts that σ by √k. Reflection uses the sample whose
        score is closest to the mean so the critique still matches a real persona.
        """
        k = max(1, PERSONA_SAMPLES_K)
        if k == 1:
            return self._evaluate_one(data_inst, prompt_template)

        samples = [self._evaluate_one(data_inst, prompt_template) for _ in range(k)]
        valid_samples = [t for t in samples if t.valid]
        pool = valid_samples or samples
        mean_score = sum(t.combined_score for t in pool) / len(pool)
        chosen = min(pool, key=lambda t: abs(t.combined_score - mean_score))
        # Stamp averaged signal onto the chosen traj (GEPA reads combined_score).
        chosen.combined_score = mean_score
        chosen.schwartz_alignment_score = sum(t.schwartz_alignment_score for t in pool) / len(pool)
        chosen.utility_score = sum(t.utility_score for t in pool) / len(pool)
        chosen.grounding_score = sum(t.grounding_score for t in pool) / len(pool)
        chosen.valid = bool(valid_samples)  # invalid only if every sample API-failed
        # Average tau if present
        tau_scores = [
            float(t.tau_result["score"]) for t in pool
            if t.tau_result and "score" in t.tau_result
        ]
        if tau_scores and chosen.tau_result is not None:
            chosen.tau_result = dict(chosen.tau_result)
            chosen.tau_result["score"] = sum(tau_scores) / len(tau_scores)
            chosen.tau_result["k_samples"] = k
        print(f"[{data_inst.user_id}] k={k} scores={[round(t.combined_score, 3) for t in samples]} "
              f"mean={mean_score:.3f} valid={chosen.valid}")
        return chosen

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
            trajs = list(ex.map(lambda d: self._evaluate_one_k(d, prompt_template), batch))

        outputs = [t.generated_persona for t in trajs]
        scores = [t.combined_score for t in trajs]
        user_ids = [t.user_id for t in trajs]

        # Paired sign-test gate vs incumbent (same panel, challenger eval).
        if capture_traces:
            self._incumbent_panel_scores = dict(zip(user_ids, scores))
            self._incumbent_panel_ids = list(user_ids)
        elif (
            PAIRED_SIGN_TEST
            and self._incumbent_panel_scores is not None
            and self._incumbent_panel_ids == user_ids
        ):
            deltas = [
                scores[i] - self._incumbent_panel_scores[user_ids[i]]
                for i in range(len(user_ids))
            ]
            n_pos = sum(1 for d in deltas if d > 1e-9)
            n_neg = sum(1 for d in deltas if d < -1e-9)
            n_tie = len(deltas) - n_pos - n_neg
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
            # Accept only if a strict majority of non-tie users improve.
            decided = n_pos + n_neg
            passes = decided > 0 and n_pos > n_neg and n_pos > decided / 2.0
            print(f"Paired sign-test: n={len(deltas)} +={n_pos} -={n_neg} tie={n_tie} "
                  f"mean_Δ={mean_delta:+.4f} pass={passes}")
            if not passes:
                # Force GEPA's sum(new) < sum(old) rejection without inventing wins.
                scores = [
                    self._incumbent_panel_scores[uid] - 1e-6 for uid in user_ids
                ]
                for t, s in zip(trajs, scores):
                    t.combined_score = s
            self._incumbent_panel_scores = None
            self._incumbent_panel_ids = None

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

        # Exclude invalid trajectories (active-signal API failure) so the
        # proposer never diagnoses transient noise as a prompt weakness.
        # Compile-to-schema failures stay valid and carry a tau critique.
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
                parts.append(f"Tau: {tau_score:.2f}/5")
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


class FixedPanelBatchSampler:
    """Always return the full loader — the fixed evaluation panel.

    Used so every reflective propose step evaluates the SAME users (paired
    incumbent vs challenger), not a reshuffled size-3 minibatch.
    """

    def next_minibatch_ids(self, loader, state):
        ids = list(loader.all_ids())
        if not ids:
            raise ValueError("FixedPanelBatchSampler: empty evaluation panel")
        return ids


if __name__ == "__main__":
    base_candidate = {"persona_prompt": GEPA_PARAGRAPH_PROMPT}

    MAX_METRIC_CALLS = int(os.getenv("MAX_METRIC_CALLS", "350"))

    TRAIN_FILE = os.getenv("TRAIN_FILE", "selected_users_pvq_gepa_train_k50.jsonl")
    # Distractor pool still draws from train+val files; the GEPA selection panel
    # is a fixed stratified subset (not a clean test set). Clean E1/E2 eval stays
    # on selected_users_pvq_eval_k100.jsonl.
    VAL_FILE = os.getenv("VAL_FILE", "selected_users_pvq_gepa_test_k25.jsonl")

    trainset_full = load_persona_dataset(TRAIN_FILE)
    valset_full = load_persona_dataset(VAL_FILE)

    # Fixed paired panel: same users for every candidate (train reflection subsample
    # AND val Pareto). Stratified by dominant Schwartz value.
    panel_rng = random.Random(42)
    panel = _stratified_sample(
        trainset_full, min(EVAL_PANEL_N, len(trainset_full)), panel_rng
    )
    trainset = panel
    valset = panel

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
            "eval_panel_n": len(panel),
            "persona_samples_k": PERSONA_SAMPLES_K,
            "paired_sign_test": PAIRED_SIGN_TEST,
            "utility_smooth": UTILITY_SMOOTH,
            "grounding_full": GROUNDING_FULL,
            "grounding_skip": GROUNDING_SKIP,
            "max_metric_calls": MAX_METRIC_CALLS,
            "train_file": TRAIN_FILE, "val_file": VAL_FILE,
            "panel_user_ids": [d.user_id for d in panel],
        }, f, indent=2)
    with open(os.path.join(RUN_DIR, "seed_prompt.txt"), "w") as f:
        f.write(GEPA_PARAGRAPH_PROMPT)

    adapter.propose_new_texts = custom_proposal_function

    print(f"Paired panel: n={len(panel)} k={PERSONA_SAMPLES_K} "
          f"sign_test={PAIRED_SIGN_TEST} users={[d.user_id for d in panel]}")

    gepa_result = gepa.optimize(
        seed_candidate=base_candidate,
        trainset=trainset,
        valset=valset,
        max_metric_calls=MAX_METRIC_CALLS,   # budget (env: MAX_METRIC_CALLS)
        reflection_lm=teacher_model,         # strong model reflects + proposes
        adapter=adapter,
        batch_sampler=FixedPanelBatchSampler(),
        val_evaluation_policy="full_eval",
    )

    best = gepa_result.best_candidate
    with open(os.path.join(RUN_DIR, "best_prompt.txt"), "w") as f:
        f.write(best["persona_prompt"])
    print("\n=== Best persona prompt ===")
    print(best)
    print(f"Distractor tier usage: {adapter._distractor_tier_counts}")
    print(f"Artifacts written to {RUN_DIR}/")
