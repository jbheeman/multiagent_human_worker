"""Persona generation at scale via NeMo Data Designer (v0.6.x).

Replaces the `pipeline.py` __main__ scale loop. Mental model: GEPA
(`personaAdapter.py`) stays OFFLINE and optimizes a prompt STRING; this module is
the scale harness that applies that fixed prompt over N seed users. GEPA is never
in the per-persona loop.

Key rework choices (see plan):
  * Construct = Schwartz ONLY (no OCEAN in the source arm).
  * `cognitive_profile.schwartz` is COPIED VERBATIM from the seed's `schwartz_json`
    (we do NOT re-infer it in DD). A deterministic post-pass pins it even if the
    structured-compile LLM drifts.
  * Deterministic YAML via yaml.safe_dump (no LLM-to-YAML call, no invented keys).
  * Value-alignment + grounding surfaced as judge columns.
  * Nemotron baseline uses its native demographic/OCEAN population; judged only on
    downstream realism + diversity, never source-fidelity.
  * Matched-N via IndexRange selection over a shared seed (paired by user across
    the ablation arms).

API verified against `data-designer-config==0.6.1`. The `data_designer` import is
lazy (inside functions) so the pure schema / YAML / pin helpers can be unit-tested
with only `pydantic` + `pyyaml` installed.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. FIXED SCHEMA (P1) -- Schwartz-only, no invented keys
# ---------------------------------------------------------------------------

class SchwartzValues(BaseModel):
    self_direction: float = Field(ge=0, le=1)
    stimulation: float = Field(ge=0, le=1)
    hedonism: float = Field(ge=0, le=1)
    achievement: float = Field(ge=0, le=1)
    power: float = Field(ge=0, le=1)
    security: float = Field(ge=0, le=1)
    conformity: float = Field(ge=0, le=1)
    tradition: float = Field(ge=0, le=1)
    benevolence: float = Field(ge=0, le=1)
    universalism: float = Field(ge=0, le=1)


class CognitiveProfile(BaseModel):
    """IS the Schwartz vector, copied verbatim. No invented per-persona keys, no OCEAN."""
    schwartz: SchwartzValues


class CommunicationStyle(BaseModel):
    """Free-text vividness lives HERE only -- not in the value axes."""
    formality: Literal["Low", "Medium", "High"]
    sentence_structure: str
    vocabulary_and_lexicon: str
    punctuation_and_formatting: str
    example_utterances: list[str] = Field(min_length=2, max_length=3)


class InteractionPolicy(BaseModel):
    """Fixed axes derived from the value vector."""
    gratification_delay_tolerance: Literal["Low", "Medium", "High"]
    authority_challenge: Literal["Passive", "Neutral", "Active"]
    policy_friction_tolerance: Literal["Low", "Medium", "High"]
    verification_patience: Literal["Low", "Medium", "High"]
    escalation_trigger: str


class PersonaProfile(BaseModel):
    id: str
    demographics: str
    cognitive_profile: CognitiveProfile
    communication_style: CommunicationStyle
    interaction_policy: InteractionPolicy
    state_transition_rules: list[str] = Field(min_length=3, max_length=5)
    termination_success: str
    termination_abandonment: str


# Map the UPPERCASE seed Schwartz keys -> schema (lowercase) field names.
_SCHWARTZ_KEY_MAP = {
    "SELF_DIRECTION": "self_direction", "STIMULATION": "stimulation",
    "HEDONISM": "hedonism", "ACHIEVEMENT": "achievement", "POWER": "power",
    "SECURITY": "security", "CONFORMITY": "conformity", "TRADITION": "tradition",
    "BENEVOLENCE": "benevolence", "UNIVERSALISM": "universalism",
}


def schwartz_from_seed(schwartz_json: str) -> dict:
    """Convert the seed's UPPERCASE Schwartz dict (JSON string) to schema fields."""
    raw = json.loads(schwartz_json) if isinstance(schwartz_json, str) else dict(schwartz_json)
    return {field: float(raw[key]) for key, field in _SCHWARTZ_KEY_MAP.items() if key in raw}


def pin_schwartz_verbatim(persona: dict, schwartz_json: str) -> dict:
    """Overwrite cognitive_profile.schwartz with the seed vector (verbatim guarantee).

    The structured-compile LLM is asked to copy the vector, but we never trust it:
    this deterministic post-pass makes the copy exact regardless of LLM drift.
    """
    persona = dict(persona)
    cp = dict(persona.get("cognitive_profile") or {})
    cp["schwartz"] = schwartz_from_seed(schwartz_json)
    persona["cognitive_profile"] = cp
    return persona


def persona_to_yaml(persona: dict) -> str:
    """Deterministic, non-LLM serialization. Replaces the YAML_PROMPT LLM call."""
    return yaml.safe_dump(
        {"persona_profile": persona},
        sort_keys=True, default_flow_style=False, allow_unicode=True,
    )


# ---------------------------------------------------------------------------
# 2. GEPA OUTPUT (a STRING produced offline by personaAdapter.py; pasted in)
# ---------------------------------------------------------------------------
# IMPORTANT (P2): the GEPA-optimized prompt must NOT instruct the persona to print
# raw Schwartz JSON or recite values-by-number. Replace this with
# gepa_result.best_candidate["persona_prompt"].

GEPA_PARAGRAPH_PROMPT = """\
You are an expert psychological profiler. Using the user's behavioral corpus and
their demographic anchor, write a coherent, behaviorally consistent first-person
persona for a customer-support simulation.

DEMOGRAPHICS: {{ demographics }}
USER CORPUS:
{{ user_corpus }}

INFERRED VALUE PROFILE (latent grounding only; do NOT name values or print numbers):
{{ schwartz_json }}

Convey priorities through behavior and attitude, never by naming psychological
values or citing numbers.
"""


# ---------------------------------------------------------------------------
# 3. MODEL WIRING (the only place the NRP endpoint + NAUT_API_KEY live)
# ---------------------------------------------------------------------------

NRP_ENDPOINT = os.getenv("NRP_ENDPOINT", "https://ellm.nrp-nautilus.io/v1")
PROVIDER_NAME = "nrp"

GENERATOR_ALIAS = "generator"
CRITIC_ALIAS = "critic"
NEMOTRON_ALIAS = "nemotron"


def make_providers():
    from data_designer.config import ModelProvider  # lazy
    return [ModelProvider(
        name=PROVIDER_NAME,
        endpoint=NRP_ENDPOINT,
        provider_type="openai",
        api_key=os.getenv("NAUT_API_KEY"),
    )]


def make_model_configs():
    from data_designer.config import ModelConfig  # lazy
    return [
        ModelConfig(alias=GENERATOR_ALIAS, model=os.getenv("GEN_MODEL", "kimi"), provider=PROVIDER_NAME),
        ModelConfig(alias=CRITIC_ALIAS, model=os.getenv("CRITIC_MODEL", "qwen3"), provider=PROVIDER_NAME),
        ModelConfig(alias=NEMOTRON_ALIAS, model=os.getenv("NEMOTRON_MODEL", "nemotron"), provider=PROVIDER_NAME),
    ]


# ---------------------------------------------------------------------------
# 4. PIPELINE FACTORY -- arms + ablations
# ---------------------------------------------------------------------------

class Arm(str, Enum):
    GEPA_FULL = "gepa_full"
    GEPA_UNOPT = "gepa_unopt"
    GEPA_VALUE_ONLY = "gepa_value"
    GEPA_BEHAVIOR_ONLY = "gepa_behav"
    NEMOTRON_BASELINE = "nemotron"


# Each GEPA arm gets its own optimized prompt string (swap in the GEPA outputs).
ARM_PROMPTS = {
    Arm.GEPA_FULL: GEPA_PARAGRAPH_PROMPT,
    Arm.GEPA_UNOPT: GEPA_PARAGRAPH_PROMPT,
    Arm.GEPA_VALUE_ONLY: GEPA_PARAGRAPH_PROMPT,
    Arm.GEPA_BEHAVIOR_ONLY: GEPA_PARAGRAPH_PROMPT,
}


def build_config(arm: Arm, seed_path: str | None, n_personas: int, seed_start: int = 0):
    from data_designer import config as dd  # lazy

    cb = dd.DataDesignerConfigBuilder(model_configs=make_model_configs())

    # ----- NEMOTRON / DEMOGRAPHIC BASELINE (the arm to beat) -----
    if arm is Arm.NEMOTRON_BASELINE:
        cb.add_column(dd.SamplerColumnConfig(
            name="person",
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(),  # native demographic/OCEAN population
        ))
        cb.add_column(dd.LLMStructuredColumnConfig(
            name="persona",
            model_alias=NEMOTRON_ALIAS,
            prompt=("Create a customer-support user persona for this person, using only "
                    "their demographic/personality attributes (no behavioral history):\n"
                    "{{ person }}"),
            output_format=PersonaProfile,
        ))
        return cb

    # ----- GEPA / BEHAVIORAL ARMS -----
    # Matched-N: shared seed, IndexRange selection (inclusive end). Use the SAME
    # range across ablation arms so rows are paired by user.
    cb.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
        selection_strategy=dd.IndexRange(start=seed_start, end=seed_start + n_personas - 1),
    )
    # Seed columns expected: user_corpus, schwartz_json, demographics, heldout_post, source, user_id.

    # (a) GEPA-optimized PARAGRAPH (kept as its own column; the regime GEPA tuned).
    cb.add_column(dd.LLMTextColumnConfig(
        name="persona_paragraph",
        model_alias=GENERATOR_ALIAS,
        prompt=ARM_PROMPTS[arm],
    ))

    # (b) Structured compile -> PersonaProfile. Copy Schwartz verbatim from the seed;
    #     generate only the communication/interaction fields. (A deterministic
    #     post-pass re-pins Schwartz after generation; see attach_yaml.)
    cb.add_column(dd.LLMStructuredColumnConfig(
        name="persona",
        model_alias=GENERATOR_ALIAS,
        prompt=("Compile this persona paragraph into the schema. Copy the value vector "
                "from {{ schwartz_json }} VERBATIM into cognitive_profile.schwartz "
                "(lowercase field names). Set demographics from {{ demographics }}. "
                "Derive communication_style, interaction_policy, and state_transition_rules "
                "from the paragraph.\n\nPARAGRAPH:\n{{ persona_paragraph }}"),
        output_format=PersonaProfile,
    ))

    # (c) Value-alignment judge -- REPORT THIS (reviewer: alignment numbers missing).
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="value_alignment",
        model_alias=CRITIC_ALIAS,
        prompt=("Does the persona's behavior/policies faithfully reflect the source value "
                "vector across ALL dimensions (not just the dominant one)? Do NOT reward "
                "the persona for naming values.\nPERSONA: {{ persona }}\nSOURCE VECTOR: {{ schwartz_json }}"),
        scores=[dd.Score(
            name="value_fidelity",
            description="Fidelity of the persona's behavior/policies to the source value vector across ALL dimensions.",
            options={
                1: "Contradicts the source vector.",
                2: "Reflects only the single dominant value.",
                3: "Reflects the top values but ignores the rest.",
                4: "Faithful across most dimensions.",
                5: "Faithful across all 10 dimensions.",
            },
        )],
    ))

    # (d) Grounding judge -- anti-hallucination vs the user's own posts.
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="grounding",
        model_alias=CRITIC_ALIAS,
        prompt=("Is every claim in the persona supported by the user's posts? Penalize "
                "fabricated biography or traits not evidenced in the corpus.\n"
                "PERSONA: {{ persona }}\nUSER CORPUS: {{ user_corpus }}"),
        scores=[dd.Score(
            name="grounding",
            description="Whether the persona's claims are supported by the user's actual posts (anti-hallucination).",
            options={
                1: "Largely fabricated; claims not in the corpus.",
                2: "Several unsupported claims.",
                3: "Mostly supported with some embellishment.",
                4: "Well supported by the corpus.",
                5: "Fully supported by the corpus.",
            },
        )],
    ))

    return cb


# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------

def attach_yaml(df, is_gepa_arm: bool):
    """Post-pass: pin Schwartz verbatim (GEPA arms) and serialize deterministic YAML."""
    def _row_yaml(row):
        persona = row["persona"]
        if isinstance(persona, str):
            persona = json.loads(persona)
        if is_gepa_arm and "schwartz_json" in row:
            persona = pin_schwartz_verbatim(persona, row["schwartz_json"])
        return persona_to_yaml(persona)

    df["persona_yaml"] = df.apply(_row_yaml, axis=1)
    return df


def generate(arm: Arm, seed_path: str | None, n_personas: int, seed_start: int = 0):
    from data_designer.interface import DataDesigner  # lazy

    designer = DataDesigner(model_providers=make_providers())
    cb = build_config(arm, seed_path, n_personas, seed_start)

    preview = designer.preview(config_builder=cb)
    try:
        preview.display_sample_record()
    except Exception:
        pass

    result = designer.create(config_builder=cb, num_records=n_personas, dataset_name=f"persona_{arm.value}")
    df = result.load_dataset()
    df = attach_yaml(df, is_gepa_arm=(arm is not Arm.NEMOTRON_BASELINE))
    return df


if __name__ == "__main__":
    N = int(os.getenv("N_PERSONAS", "200"))
    seed = os.getenv("SEED_FILE", "seed_gepa.jsonl")

    # Treatment vs baseline-to-beat (matched N).
    gepa_df = generate(Arm.GEPA_FULL, seed, n_personas=N, seed_start=0)
    nemotron_df = generate(Arm.NEMOTRON_BASELINE, None, n_personas=N)

    # Ablation ladder on the same users (paired): swap ARM_PROMPTS with the
    # corresponding GEPA outputs before running.
    for arm in (Arm.GEPA_UNOPT, Arm.GEPA_VALUE_ONLY, Arm.GEPA_BEHAVIOR_ONLY):
        generate(arm, seed, n_personas=N, seed_start=0)
