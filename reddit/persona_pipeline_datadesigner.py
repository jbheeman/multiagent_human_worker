"""Persona generation at scale via NeMo Data Designer (v0.6.x).

Replaces the `pipeline.py` __main__ scale loop. Mental model: GEPA
(`personaAdapter.py`) stays OFFLINE and optimizes a prompt STRING; this module is
the scale harness that applies that fixed prompt over N seed users. GEPA is never
in the per-persona loop.

Key rework choices (see plan):
  * Construct = Schwartz ONLY (no OCEAN in the source arm).
  * Schwartz vector lives ONLY in the schwartz_json sidecar column — not in the
    PersonaProfile object. Keeps the simulator from reciting numbers and makes the
    value-alignment judge non-tautological (must infer from behavior, not field-match).
  * Deterministic YAML via yaml.safe_dump (no LLM-to-YAML call, no invented keys).
  * Value-alignment + grounding + register + role-check surfaced as judge columns.
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
import tempfile
from enum import Enum
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------------------------
# 1. FIXED SCHEMA (P1) -- Schwartz-only, no invented keys
# ---------------------------------------------------------------------------

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
    demographics: Optional[str] = None  # omitted for Reddit/GEPA arm; kept for Nemotron
    communication_style: CommunicationStyle
    interaction_policy: InteractionPolicy
    state_transition_rules: list[str] = Field(min_length=3, max_length=5)
    termination_success: str
    termination_abandonment: str
    # cognitive_profile (Schwartz numbers) intentionally absent from the persona object.
    # The vector lives in the schwartz_json sidecar column for analysis and judge ground
    # truth only — keeping it out of the persona prevents simulator recitation and makes
    # the value_alignment judge non-tautological (no stored numbers to match against).


def _json_safe(obj):
    """Coerce numpy/pandas leaves to plain Python for YAML/JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _json_safe(obj.tolist())
    return obj


def persona_to_yaml(persona: dict) -> str:
    """Deterministic, non-LLM serialization. Replaces the YAML_PROMPT LLM call."""
    return yaml.safe_dump(
        {"persona_profile": _json_safe(persona)},
        sort_keys=True, default_flow_style=False, allow_unicode=True,
    )


# ---------------------------------------------------------------------------
# 2. GEPA OUTPUT (a STRING produced offline by personaAdapter.py; pasted in)
# ---------------------------------------------------------------------------
# IMPORTANT (P2): the GEPA-optimized prompt must NOT instruct the persona to print
# raw Schwartz JSON or recite values-by-number. Replace this with
# gepa_result.best_candidate["persona_prompt"].

OPTIMIZED_GEPA_FULL_PROMPT = """\
You are an expert psychological profiler. From the user's behavioral corpus, write a functional first-person behavioral profile — psychologically vivid enough that someone reading it could predict how this person communicates, argues, and reacts under pressure in a task-oriented dialogue.

Do not write rigid rules ("never give my zip code"). Write the psychological reasoning ("I treat personal data like something to hoard because..."). Do not write dialogues, transcripts, or invented transactional details.

USER CORPUS (speaker-attributed; [USER]/[INTERLOCUTOR]/[QUOTED] turns under [r/subreddit] headers):
{{ user_corpus }}

CORPUS INSTRUCTIONS:
- Build the profile ONLY from USER turns — character, register, disposition, triggers.
- INTERLOCUTOR and QUOTED turns are context for how this user argues and reacts; never attribute quoted views, beliefs, or biography to the profile.
- Transfer how they write and argue, not their Reddit topics or subculture jargon as identity.
- ABSTRACT THE METHOD, NOT THE METAPHOR: If a user argues about sports fouls to demand precision, do not write "I treat vague language like a foul call." Write "I treat vague language as a risk to accuracy that requires immediate clarification." Keep the cognitive drive, drop the topic-specific analogy. The profile must remain valid even if the interaction format changes from forum post to direct task.
- VALUE-CALIBRATED INTENSITY: Use the Latent Value Profile to weight the intensity of behaviors. If a value score is low (<0.35), do not write behaviors driven by that value even if the corpus shows occasional spikes. Prioritize the top 2-3 values from the vector as the core drivers. A user with low "Power" scores must not sound domineering, even if they are confident.
- GROUNDING IS ABSOLUTE: You cannot attribute a writing habit (bullets, edits, headers, specific punctuation) to the user unless it appears in the provided USER CORPUS. If the user writes in paragraphs, do not claim "I use bullet points." If the user does not use "Edit:" tags, do not claim "I add Edit notes."
- TONAL FIDELITY: Match the corpus temperature exactly. Do not invent hostility, profanity, or emotional intensity unless it is present in the text. Do not invent physical modalities (raising voice, physical presence) for text-based users. A user who writes politely about marriage must be profiled as polite, even if their values suggest "passion." Do not sand down a hostile user, but do not amplify a polite one.
- REGISTER IS NON-NEGOTIABLE: Match the source's actual vocabulary level and sentence length. Do not upgrade their diction, smooth their syntax, or wrap the profile in essay structure (headers, roman numerals, bolded thesis lines) unless the user actually writes that way.

LATENT VALUE PROFILE (use this vector to weight the priority and intensity of traits observed in the corpus; treat it as research you will never quote, not as vocabulary to use):
{{ schwartz_json }}

VALUE-NAME LEAK IS A FAILURE, NOT A STYLE CHOICE. Never write the literal Schwartz dimension names or close derivatives -- in any form (capitalized, lowercase, or as an adjective/noun) -- including: power, achievement, hedonism/hedonistic, stimulation, self-direction, universalism, benevolence, tradition, conformity, security. Do not print numbers, percentages, or vector/profile language ("my X score", "my vector", "rates high on Y", "feeds my Z streak"). Express the SAME priorities only through what the person notices, wants, argues for, and reacts to -- never through the label of the value itself. Use the values to understand *why* the user cares, but NEVER let them override the *style* observed in the corpus. Before finishing, re-read every sentence for one of the banned words above and rewrite it if found.

BEHAVIORAL PREDICTION CHECK: Ensure every psychological claim implies an observable action in a dialogue. Instead of abstract traits ("I value precision"), describe the reaction ("I interrupt vague answers to request exact timestamps"). Focus on **triggers** (what frustrates them), **verification** (how they confirm truth), and **escalation** (what makes them demand authority), rather than **narrative flow** (how they order sentences). The goal is to simulate a user in a customer-support task, not to replicate a forum post. Ensure behaviors are **Cross-Context Valid**: Would this behavior still make sense if the user were booking a flight instead of discussing sports? If not, generalize the underlying need.

EXTERNAL CONTENT THIS USER QUOTED OR ENGAGED WITH (selection + stance; secondary signal):
{{ quote_signals }}
"""



GEPA_VALUE_ONLY_PROMPT = """\
You are an expert psychological profiler. From the user's behavioral corpus, write a
self-explanatory first-person portrait — psychologically vivid in its behavioral precision
and tonal authenticity, enabling a simulator to reproduce how this person communicates,
argues, and reacts under pressure.

Do not write rigid rules. Write the psychological reasoning and behavioral drivers.
Do not write dialogues, transcripts, or invented transactional details.

USER CORPUS (speaker-attributed; [USER]/[INTERLOCUTOR]/[QUOTED] turns under [r/subreddit] headers):
{{ user_corpus }}

CORPUS INSTRUCTIONS:
- STRICT GROUNDING PROTOCOL: The portrait must be a behavioral synthesis of {{ user_corpus }}
  ONLY. Every topic, skill, reference, relationship, and biographical detail in the portrait
  must be traceable to the user's actual turns. If the user discusses a game, the portrait
  may reference that game, but must not invent specific mechanics, items, ranks, or lore
  absent from the corpus. If the user discusses a law, the portrait may reference that law,
  but must not invent specific clauses, cases, or statutes not cited in the corpus.
- NO HALLUCINATION: Never fill gaps in the corpus with plausible fiction. If the corpus does
  not reveal a hobby, profession, or preference, the portrait must not supply one. A partial
  portrait grounded in evidence is superior to a complete portrait built on invention.
- INTERLOCUTOR and QUOTED turns are context for how this user argues and reacts; never
  attribute quoted views, beliefs, or biography to the portrait.
- Transfer how they write and argue, not their Reddit topics or subculture jargon as identity.
- REGISTER IS NON-NEGOTIABLE: match the source's actual vocabulary level, sentence length,
  and formatting. Do not upgrade their diction, smooth their syntax, or wrap the portrait in
  essay structure unless the user actually writes that way. Sanding a hostile or crude user
  into an articulate, agreeable, or academic-sounding one is a FAILURE.

LATENT VALUE PROFILE (ground the portrait's behavioral drivers in this vector; treat it as
a lens for priorities and reactions, never as a source of content):
{{ schwartz_json }}

- BEHAVIORAL MAPPING ONLY: Use {{ schwartz_json }} exclusively to modulate the persona's
  priorities, emotional triggers, argumentation style, risk tolerance, and what they notice
  or value. The vector explains the "how" and "why" of behavior, never the "what".
- VALUES DO NOT GENERATE CONTENT: High scores in dimensions like ACHIEVEMENT, STIMULATION,
  or SECURITY must not authorize the addition of new topics, skills, or facts. For example,
  high ACHIEVEMENT must manifest as a drive for mastery or progress expressed through the
  topics already present in the corpus, not by inventing gaming stats or career goals.
  High SECURITY must manifest as a preference for stability or risk-aversion applied to the
  corpus subjects, not by inventing legal frameworks or safety protocols.
- VALUE-NAME LEAK IS A FAILURE: Never write the literal Schwartz dimension names or close
  derivatives -- in any form -- including: power, achievement, hedonism/hedonistic,
  stimulation, self-direction, universalism, benevolence, tradition, conformity, security.
  Do not print numbers, percentages, or vector language. Express priorities only through
  behavior, reactions, and what the person argues for. Before finishing, scan every sentence
  for banned words and rewrite if found.
"""

GEPA_PARAGRAPH_PROMPT = """\
You are an expert psychological profiler. From the user's behavioral corpus, write a
self-explanatory first-person portrait — psychologically vivid enough that someone
reading it could predict how this person communicates, argues, and reacts under pressure.

Do not write rigid rules ("never give my zip code"). Write the psychological reasoning
("I treat personal data like something to hoard because..."). Do not write dialogues,
transcripts, or invented transactional details.

USER CORPUS (speaker-attributed):
{{ user_corpus }}

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
{{ schwartz_json }}

VALUE-NAME LEAK IS A FAILURE, NOT A STYLE CHOICE. Never write the literal Schwartz dimension
names or close derivatives -- in any form (capitalized, lowercase, or as an adjective/noun) --
including: power, achievement, hedonism/hedonistic, stimulation, self-direction, universalism,
benevolence, tradition, conformity, security. Do not print numbers, percentages, or
vector/profile language ("my X score", "my vector", "rates high on Y", "feeds my Z streak").
Express the SAME priorities only through what the person notices, wants, argues for, and
reacts to -- never through the label of the value itself. Before finishing, re-read every
sentence for one of the banned words above and rewrite it if found.

EXTERNAL CONTENT THIS USER QUOTED OR ENGAGED WITH (selection + stance; secondary signal):
{{ quote_signals }}
"""

REDDIT_NO_PSYCH_PARAGRAPH_PROMPT = """\
You are an expert behavioral profiler. From the user's behavioral corpus, write a
self-explanatory first-person portrait -- vivid enough that someone reading it could
predict how this person communicates, argues, and reacts under pressure.

Do not write rigid rules ("never give my zip code"). Write the psychological reasoning
("I treat personal data like something to hoard because..."). Do not write dialogues,
transcripts, or invented transactional details.

USER CORPUS (speaker-attributed):
{{ user_corpus }}

CORPUS INSTRUCTIONS:
- Build the portrait ONLY from USER turns -- character, register, disposition, triggers.
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

EXTERNAL CONTENT THIS USER QUOTED OR ENGAGED WITH (selection + stance; secondary signal):
{{ quote_signals }}
"""


# ---------------------------------------------------------------------------
# 3. MODEL WIRING (the only place the NRP endpoint + NAUT_API_KEY live)
# ---------------------------------------------------------------------------

NRP_ENDPOINT = os.getenv("NRP_ENDPOINT", "https://ellm.nrp-nautilus.io/v1")
PROVIDER_NAME = "nrp"

GENERATOR_ALIAS = "generator"
CRITIC_ALIAS = "critic"           # heavy judges: register, grounding (full corpus)
CRITIC_FAST_ALIAS = "critic_fast"  # light judges: value_alignment, role_check
NEMOTRON_ALIAS = "nemotron"

# Judge prompts include full corpus+heldout by default; excerpt columns keep the
# generator on full text while reducing critic timeouts / salvage drops on heavy users.
JUDGE_CORPUS_CHARS = int(os.getenv("JUDGE_CORPUS_CHARS", "12000"))
JUDGE_HELDOUT_CHARS = int(os.getenv("JUDGE_HELDOUT_CHARS", "5000"))


def make_providers():
    from data_designer.config import ModelProvider  # lazy
    return [ModelProvider(
        name=PROVIDER_NAME,
        endpoint=NRP_ENDPOINT,
        provider_type="openai",
        api_key=os.getenv("NAUT_API_KEY"),
    )]


def make_model_configs():
    from data_designer.config import ChatCompletionInferenceParams, ModelConfig  # lazy
    gen_parallel = int(os.getenv("GEN_PARALLEL", "1"))
    critic_parallel = int(os.getenv("CRITIC_PARALLEL", "1"))
    critic_fast_parallel = int(os.getenv("CRITIC_FAST_PARALLEL", os.getenv("CRITIC_PARALLEL", "1")))
    return [
        ModelConfig(
            alias=GENERATOR_ALIAS,
            model=os.getenv("GEN_MODEL", "gpt-oss"),
            provider=PROVIDER_NAME,
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=gen_parallel),
        ),
        ModelConfig(
            alias=CRITIC_ALIAS,
            model=os.getenv("CRITIC_MODEL", "minimax-m2"),
            provider=PROVIDER_NAME,
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=critic_parallel),
        ),
        ModelConfig(
            alias=CRITIC_FAST_ALIAS,
            model=os.getenv("CRITIC_FAST_MODEL", "minimax-m2"),
            provider=PROVIDER_NAME,
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=critic_fast_parallel),
        ),
        ModelConfig(
            alias=NEMOTRON_ALIAS,
            model=os.getenv("NEMOTRON_MODEL", "gpt-oss"),
            provider=PROVIDER_NAME,
            inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=gen_parallel),
        ),
    ]


# ---------------------------------------------------------------------------
# 4. PIPELINE FACTORY -- arms + ablations
# ---------------------------------------------------------------------------

class Arm(str, Enum):
    GEPA_FULL = "gepa_full"
    GEPA_UNOPT = "gepa_unopt"
    GEPA_VALUE_ONLY = "gepa_value"
    GEPA_BEHAVIOR_ONLY = "gepa_behav"
    REDDIT_NO_PSYCH = "reddit_no_psych"
    NEMOTRON_BASELINE = "nemotron"


# Each GEPA arm gets its own optimized prompt string (swap in the GEPA outputs).
ARM_PROMPTS = {
    Arm.GEPA_FULL: OPTIMIZED_GEPA_FULL_PROMPT,
    Arm.GEPA_UNOPT: GEPA_PARAGRAPH_PROMPT,
    Arm.GEPA_VALUE_ONLY: GEPA_VALUE_ONLY_PROMPT,
    Arm.GEPA_BEHAVIOR_ONLY: GEPA_PARAGRAPH_PROMPT,
    Arm.REDDIT_NO_PSYCH: REDDIT_NO_PSYCH_PARAGRAPH_PROMPT,
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
    # Seed columns expected: user_corpus, schwartz_json, quote_signals, heldout_post, source, user_id.

    # (a) GEPA-optimized PARAGRAPH (kept as its own column; the regime GEPA tuned).
    cb.add_column(dd.LLMTextColumnConfig(
        name="persona_paragraph",
        model_alias=GENERATOR_ALIAS,
        prompt=ARM_PROMPTS[arm],
    ))

    # (b) Structured compile -> PersonaProfile. Purely behavioral fields only —
    #     the Schwartz vector stays in the schwartz_json sidecar column.
    cb.add_column(dd.LLMStructuredColumnConfig(
        name="persona",
        model_alias=GENERATOR_ALIAS,
        prompt=(
            "Compile this psychological portrait into the schema.\n\n"
            "ROLE: This persona is the person who contacts support to get something done — "
            "never the support agent.\n"
            "- example_utterances: 2–3 short lines this person would say when seeking help, "
            "in their voice from the portrait. Not agent-side troubleshooting.\n"
            "- state_transition_rules: this person's POV — 'IF the other party does X, "
            "THEN I react Y.'\n"
            "- escalation_trigger: what makes this person demand a human or walk away.\n\n"
            "Do NOT set demographics (leave None). Do NOT add cognitive_profile or schwartz "
            "fields. Derive all fields from the portrait.\n\n"
            "PORTRAIT:\n{{ persona_paragraph }}"
        ),
        output_format=PersonaProfile,
    ))

    # (c) Value-alignment judge — light critic (persona + schwartz sidecar only).
    if arm is not Arm.REDDIT_NO_PSYCH:
        cb.add_column(dd.LLMJudgeColumnConfig(
            name="value_alignment",
            model_alias=CRITIC_FAST_ALIAS,
            prompt=(
                "Score whether the behavioral profile FAITHFULLY IMPLIES the specific Schwartz "
                "levels below — not whether you can rationalize a story after seeing the numbers.\n\n"
                "RULES:\n"
                "- Judge only communication_style, interaction_policy, state_transition_rules, "
                "and escalation_trigger. No stored numbers in the persona.\n"
                "- HIGH vector dimensions (>0.6) need clear behavioral evidence. "
                "MID-RANGE (0.3–0.6) need moderate evidence. LOW (<0.3) require the behavior "
                "does NOT emphasize that value.\n"
                "- Do NOT invent post-hoc correlations (e.g. 'wants fast resolution → Security 0.7' "
                "without explicit safety/stability-seeking). Weak stretches LOWER the score.\n"
                "- If only 1–2 values are behaviorally expressed and others lack correlates, cap at 2–3.\n"
                "- Reserve 5 for proportional support across ALL 10 dimensions. "
                "When in doubt between two scores, choose the LOWER one.\n\n"
                "BEHAVIORAL PROFILE:\n{{ persona }}\n"
                "SOURCE VALUE VECTOR (ground truth):\n{{ schwartz_json }}"
            ),
            scores=[dd.Score(
                name="value_fidelity",
                description=(
                    "Do behavioral fields imply the source Schwartz vector with proportional "
                    "evidence per dimension (strict; no post-hoc rationalization)?"
                ),
                options={
                    1: "Behavior contradicts the vector OR only coincidental overlap.",
                    2: "Expresses 1–2 dominant values; most dimensions unsupported or contradicted.",
                    3: "Top values present; mid/low dimensions mostly unexpressed or stretched.",
                    4: "Most dimensions supported; 1–2 weak or slightly overstretched correlates.",
                    5: "All 10 dimensions have direct, proportionally-weighted behavioral evidence.",
                },
            )],
        ))

    # (d) Grounding judge — anti-hallucination vs USER corpus (not anti-inference).
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="grounding",
        model_alias=CRITIC_ALIAS,
        prompt=(
            "Audit whether this persona HALLUCINATES facts not evidenced in the user's history.\n\n"
            "THREE CATEGORIES:\n"
            "1. EXPECTED — do NOT penalize:\n"
            "   - Inferred communication_style, interaction_policy, state_transition_rules, "
            "escalation_trigger (corpus never states these literally).\n"
            "   - Generic support-context example lines or IF-agent-then-I rules — the eval "
            "framework supplies scenarios at test time; placeholder complaints are fine.\n"
            "2. EXPECTED IF CONSISTENT — judge match, not presence:\n"
            "   - Whether inferred style matches USER turns (tone, bluntness, profanity, "
            "patience, hostility). Style mismatch = penalize.\n"
            "3. FABRICATION — penalize:\n"
            "   - Corpus-specific biography imported as identity: Reddit topics, subreddit "
            "drama, named people, jobs, medical history, demographics not in USER turns.\n"
            "   - INTERLOCUTOR/QUOTED content credited as the user's own views.\n\n"
            "Do NOT penalize absence of literal corpus quotes in structured fields. "
            "Penalize invented biography and register mismatch.\n\n"
            "Scoring guide:\n"
            "1 = Major corpus biography imported OR style strongly contradicts USER turns.\n"
            "2 = Several unsupported biography claims OR style mostly mismatched.\n"
            "3 = Inference mostly OK; minor stretch or slight register softening.\n"
            "4 = Style/policy well-matched; no imported corpus biography.\n"
            "5 = Strong register match; zero invented biography from corpus.\n\n"
            "PERSONA: {{ persona }}\nUSER CORPUS: {{ user_corpus_judge }}"
        ),
        scores=[dd.Score(
            name="grounding",
            description=(
                "Anti-hallucination for corpus biography; inferred style and generic "
                "support examples are expected."
            ),
            options={
                1: "Corpus biography imported OR style contradicts USER turns.",
                2: "Several biography inventions OR style mostly mismatched.",
                3: "Mostly consistent inference; minor stretch.",
                4: "Well-matched style; no imported corpus biography.",
                5: "Strong style match; zero corpus biography invented.",
            },
        )],
    ))

    # (e) Register judge -- checks tonal fidelity against the held-out post.
    #     A persona noticeably softer/more polite than the source scores LOW.
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="register",
        model_alias=CRITIC_ALIAS,
        prompt=("Compare the persona's communication_style and example_utterances against "
                "the user's USER-turn corpus and this held-out post. Score whether the "
                "persona's REGISTER and AFFECT — tone, bluntness, hostility, contempt, "
                "patience — match the source's specific flavor. A persona noticeably "
                "softer or more agreeable than the source scores LOW.\n"
                "HELD-OUT POST (excerpt): {{ heldout_excerpt }}\n"
                "USER CORPUS (excerpt): {{ user_corpus_judge }}\n"
                "PERSONA: {{ persona }}"),
        scores=[dd.Score(
            name="register_fidelity",
            description="Does persona tone/affect match the source user's specific register?",
            options={
                1: "Persona is dramatically more polite/tame than the source.",
                2: "Noticeably softened; specific edge is laundered into generic difficulty.",
                3: "Partially matches; some characteristic register preserved.",
                4: "Mostly matches the source's specific register.",
                5: "Fully faithful to the source's register and flavor of difficulty.",
            },
        )],
    ))

    # (f) Role-check judge — light critic (example_utterances only).
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="role_check",
        model_alias=CRITIC_FAST_ALIAS,
        prompt=("Check whether the persona's example_utterances are CUSTOMER-SIDE "
                "(help-seeking, complaints, reactions) or AGENT-SIDE (troubleshooting "
                "questions like 'which step triggered?', 'let me explain', asking the "
                "other party for diagnostic details).\n"
                "PERSONA: {{ persona }}"),
        scores=[dd.Score(
            name="customer_role",
            description="Are all example_utterances from the customer's perspective?",
            options={
                1: "Utterances are agent-side (diagnostic questions, explaining to user).",
                2: "Mostly agent-side with one customer line.",
                3: "Mixed; both agent and customer lines.",
                4: "Mostly customer-side; one agent-ish line.",
                5: "All utterances are customer-side (help-seeking, requests, reactions).",
            },
        )],
    ))

    return cb


# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------

def attach_yaml(df):
    """Serialize persona to deterministic YAML. The Schwartz vector is NOT included —
    it stays in the schwartz_json sidecar column for analysis and judge ground truth."""
    def _row_yaml(row):
        persona = row["persona"]
        if isinstance(persona, str):
            persona = json.loads(persona)
        return persona_to_yaml(persona)

    df["persona_yaml"] = df.apply(_row_yaml, axis=1)
    return df


def prepare_seed_row(row: dict) -> dict:
    """Add truncated judge-sidecar fields; generator columns keep full corpus."""
    row = dict(row)
    corpus = row.get("user_corpus", "")
    heldout = row.get("heldout_post", "")
    if len(corpus) > JUDGE_CORPUS_CHARS:
        row["user_corpus_judge"] = corpus[:JUDGE_CORPUS_CHARS] + "\n...[truncated for judge context]"
    else:
        row["user_corpus_judge"] = corpus
    if len(heldout) > JUDGE_HELDOUT_CHARS:
        row["heldout_excerpt"] = heldout[:JUDGE_HELDOUT_CHARS] + "\n...[truncated for judge context]"
    else:
        row["heldout_excerpt"] = heldout
    return row


def materialize_seed_with_judge_excerpts(
    seed_path: str,
    seed_start: int,
    n: int,
) -> tuple[str, int, bool]:
    """Write a temp seed JSONL with judge excerpts; return (path, seed_start, is_temp)."""
    rows = load_seed_rows(seed_path, seed_start, n)
    if not rows:
        return seed_path, seed_start, False
    if "user_corpus_judge" in rows[0]:
        return seed_path, seed_start, False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as tmp:
        for row in rows:
            tmp.write(json.dumps(prepare_seed_row(row)) + "\n")
        return tmp.name, 0, True


def generate(
    arm: Arm,
    seed_path: str | None,
    n_personas: int,
    seed_start: int = 0,
    *,
    skip_preview: bool = False,
):
    from data_designer.interface import DataDesigner  # lazy

    seed_path_eff, seed_start_eff, seed_tmp = (
        materialize_seed_with_judge_excerpts(seed_path, seed_start, n_personas)
        if seed_path else (None, seed_start, False)
    )
    try:
        designer = DataDesigner(model_providers=make_providers())
        cb = build_config(arm, seed_path_eff, n_personas, seed_start_eff)

        if not skip_preview:
            preview = designer.preview(config_builder=cb, num_records=n_personas)
            try:
                preview.display_sample_record()
            except Exception:
                pass

        result = designer.create(
            config_builder=cb, num_records=n_personas, dataset_name=f"persona_{arm.value}",
        )
        return result.load_dataset()
    finally:
        if seed_tmp:
            os.unlink(seed_path_eff)


def try_generate(
    arm: Arm,
    seed_path: str | None,
    n_personas: int,
    seed_start: int = 0,
    *,
    skip_preview: bool = False,
):
    """Like generate(), but returns an empty DataFrame when DD drops every row."""
    import pandas as pd
    from data_designer.interface.errors import DataDesignerGenerationError

    try:
        return generate(
            arm, seed_path, n_personas, seed_start, skip_preview=skip_preview,
        )
    except DataDesignerGenerationError as exc:
        print(
            "warning: generation produced no rows "
            f"({exc}). A column likely failed after retries (check logs for "
            "'Non-retryable failure on <column>' or exhausted salvage). "
            "Re-run with --resume to retry."
        )
        return pd.DataFrame()


def _write_single_seed_row(seed_row: dict) -> str:
    """Write one seed row to a temp JSONL; caller must unlink the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(json.dumps(prepare_seed_row(seed_row)) + "\n")
        return tmp.name


def load_seed_rows(seed_path: str, seed_start: int, n: int) -> list[dict]:
    """Return seed rows for [seed_start, seed_start + n)."""
    with open(seed_path, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    return all_rows[seed_start : seed_start + n]


def load_existing_rows(path: str) -> dict[str, dict]:
    """Load completed persona rows keyed by user_id."""
    rows: dict[str, dict] = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = row.get("user_id")
            if user_id:
                rows[user_id] = row
    return rows


def count_existing_rows(path: str) -> int:
    """Count JSONL rows in an output file."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def merge_rows_in_seed_order(
    rows_by_id: dict[str, dict],
    seed_path: str,
    seed_start: int,
    n: int,
) -> list[dict]:
    """Order merged rows to match the seed file selection."""
    seed_rows = load_seed_rows(seed_path, seed_start, n)
    return [rows_by_id[r["user_id"]] for r in seed_rows if r["user_id"] in rows_by_id]


def generate_missing_gepa(
    arm: Arm,
    seed_path: str,
    seed_start: int,
    n: int,
    out_path: str,
    resume_attempts: int = 3,
):
    """Generate only seed users absent from out_path; return merged rows in seed order."""
    import pandas as pd

    target_seed = load_seed_rows(seed_path, seed_start, n)
    target_ids = [r["user_id"] for r in target_seed]
    rows_by_id = load_existing_rows(out_path)
    missing_seed = [r for r in target_seed if r["user_id"] not in rows_by_id]

    if not missing_seed:
        print(f"resume: already complete ({len(rows_by_id)}/{n} in {out_path})")
        return pd.DataFrame(merge_rows_in_seed_order(rows_by_id, seed_path, seed_start, n))

    missing_ids = [r["user_id"] for r in missing_seed]
    print(
        f"resume: {len(rows_by_id)}/{n} complete in {out_path}, "
        f"generating {len(missing_seed)} missing ({', '.join(missing_ids)})"
    )

    for seed_row in missing_seed:
        user_id = seed_row["user_id"]
        for attempt in range(1, resume_attempts + 1):
            tmp_path = _write_single_seed_row(seed_row)
            try:
                new_df = try_generate(
                    arm, tmp_path, n_personas=1, seed_start=0, skip_preview=True,
                )
            finally:
                os.unlink(tmp_path)

            if not new_df.empty:
                rows_by_id[user_id] = new_df.iloc[0].to_dict()
                print(f"  + {user_id}" + (f" (attempt {attempt})" if attempt > 1 else ""))
                break
            if attempt < resume_attempts:
                print(f"  - {user_id}: attempt {attempt}/{resume_attempts} dropped, retrying...")
            else:
                print(f"  - {user_id}: dropped after {resume_attempts} attempt(s)")

    merged = merge_rows_in_seed_order(rows_by_id, seed_path, seed_start, n)
    still_missing = [uid for uid in target_ids if uid not in rows_by_id]
    if still_missing:
        print(
            f"warning: still missing {len(still_missing)}/{n} after resume "
            f"({', '.join(still_missing)})"
        )
    return pd.DataFrame(merged)


def generate_missing_nemotron(n: int, out_path: str):
    """Append synthetic Nemotron personas until out_path has n rows."""
    import pandas as pd

    done = count_existing_rows(out_path)
    if done >= n:
        print(f"resume: already complete ({done}/{n} in {out_path})")
        rows = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows[:n])

    need = n - done
    print(f"resume: {done}/{n} complete in {out_path}, generating {need} more")
    new_df = try_generate(Arm.NEMOTRON_BASELINE, None, n_personas=need, skip_preview=True)
    if new_df.empty:
        print("warning: no new nemotron rows produced (re-run with --resume to retry)")
    if done:
        rows = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        base_df = pd.DataFrame(rows)
        if new_df.empty:
            return base_df
        return pd.concat([base_df, new_df], ignore_index=True)
    return new_df


def _parse_arm(value: str) -> Arm:
    try:
        return Arm(value)
    except ValueError:
        choices = ", ".join(a.value for a in Arm)
        raise SystemExit(f"unknown arm {value!r}; choose one of: {choices}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate personas via NeMo Data Designer (one arm per run).",
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=[a.value for a in Arm],
        help="pipeline arm to run (e.g. nemotron, gepa_full, gepa_unopt)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(os.getenv("N_PERSONAS", "200")),
        help="number of personas to generate (default: N_PERSONAS env or 200)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSONL path (default: personas_<arm>.jsonl)",
    )
    parser.add_argument(
        "--seed",
        default=os.getenv("SEED_FILE", "seed_eval.jsonl"),
        help="seed JSONL for GEPA arms (ignored for nemotron; default: SEED_FILE env or seed_eval.jsonl)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="IndexRange start for matched-N seed selection (default: 0)",
    )
    parser.add_argument(
        "--from-artifacts",
        default=None,
        metavar="DIR",
        help="recover from existing DD artifacts dir (skip LLM generation; e.g. artifacts/persona_nemotron)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip users already in --out and generate only missing rows (GEPA: by user_id; Nemotron: by row count)",
    )
    parser.add_argument(
        "--resume-attempts",
        type=int,
        default=int(os.getenv("RESUME_ATTEMPTS", "3")),
        help="per-user DD retries on --resume before giving up (default: 3)",
    )
    args = parser.parse_args()

    arm = _parse_arm(args.arm)
    out_path = args.out or f"personas_{arm.value}.jsonl"
    seed_path = None if arm is Arm.NEMOTRON_BASELINE else args.seed

    if args.resume and args.from_artifacts:
        raise SystemExit("--resume and --from-artifacts cannot be used together")

    if args.from_artifacts:
        import pandas as pd
        from pathlib import Path

        parquet_dir = Path(args.from_artifacts) / "parquet-files"
        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet files in {parquet_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    elif args.resume:
        if arm is Arm.NEMOTRON_BASELINE:
            df = generate_missing_nemotron(args.n, out_path)
        else:
            if not seed_path:
                raise SystemExit("--resume for GEPA arms requires --seed")
            df = generate_missing_gepa(
                arm, seed_path, args.seed_start, args.n, out_path,
                resume_attempts=args.resume_attempts,
            )
    else:
        df = generate(arm, seed_path, n_personas=args.n, seed_start=args.seed_start)

    df = attach_yaml(df)
    if df.empty and args.resume:
        print(f"warning: nothing new to write; leaving {out_path} unchanged")
        return
    df.to_json(out_path, orient="records", lines=True)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()





