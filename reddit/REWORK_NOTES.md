# Persona Generation Rework — notes for the methods section

This documents the AAAI rework of the persona-generation step. Each change is tied
to a reviewer critique. The construct is now **Schwartz values + held-out behavior**;
OCEAN is no longer used in the source arm.

## What changed and why

| Change | File(s) | Reviewer critique addressed |
|---|---|---|
| **De-leak the prompt** — removed the requirement that the persona print raw Schwartz JSON (§3) and recite "My [Value] of [number]" in its monologue (§4). | `personaAdapter.py` (`UCSD_PERSONA_PROMPT`), `pipeline.py` (`REDDIT_PROMPT`) | Weak human validation / κ=0.31 (construct leak); "vivid but arbitrary artifacts" |
| **Full-vector alignment** — replaced dominant-trait thresholding with mean-centered cosine (Pearson) across all 10 Schwartz traits; Spearman optional. | `alignment.py`, `personaAdapter._score_schwartz_alignment` | 9/10 dims floating free; matched-source fidelity |
| **Behavioral tau judge** — the satisfaction/tau judge now rewards *observable* interaction behavior, not the persona naming its own values in the monologue. | `personaAdapter._score_persona_with_tau` | Second half of the κ construct fix |
| **Measured behavioral grounding** — held out one post per user; `_utility_score` checks whether the persona predicts it; `_grounding_score` (NLI) gates hallucination. Added as a 3rd GEPA signal. | `enrich_users.py`, `personaAdapter._utility_score/_grounding_score/evaluate` | "Behaviorally grounded" was claimed but never measured |
| **Real train size + ablation toggles** — env-driven `TRAIN_N/VAL_N/MAX_METRIC_CALLS`; `USE_TAU/USE_UTILITY/USE_GROUNDING_GATE/SIMILARITY_METRIC` constants are the ablation ladder. Per-user train/val/test split from the 1000-user file. | `personaAdapter.__main__`, `enrich_users.split_into_train_val_test` | GEPA tuned on N=2 / 50 calls (indefensible) |
| **Fixed schema + Data Designer scale harness** — `PersonaProfile` (no invented keys), Schwartz copied **verbatim** from the seed (deterministic post-pass pins it), deterministic YAML, value-alignment + grounding judge columns, Nemotron demographic baseline, matched-N via `IndexRange`. | `persona_pipeline_datadesigner.py`, `build_dd_seed.py` | "Seems AI-generated" / no shared schema / blocks matched-N / value-laundering compiler |

## Circularity, stated honestly

The source Schwartz vector is **LLM-derived** (`enrich_users.get_schwartz_vector_chameleon`,
`qwen3`) from the same posts the persona is generated from. The generator is given that
vector and PVQ then measures the persona against it. We therefore (a) frame the value
channel as LLM-derived in the paper, and (b) re-infer the vector on the **held-out-removed**
corpus so the held-out behavioral target is never part of the vector it is scored against.

## Expected (lower, more honest) numbers

Removing the recitation leak should **lower** the reported value-alignment relative to the
withdrawn version — that drop is the point. The previous high scores were achievable by
copying the printed numbers off the page; the new full-vector, leak-free alignment measures
profile-shape fidelity instead. Report old vs new side by side.

## Run flow (live, requires `NAUT_API_KEY` + `pip install -r requirements.txt`)

```
export NAUT_API_KEY=...                  # NRP Nautilus
# 1. Enrich the per-user history file: hold out a post, re-infer Schwartz, infer demographics, split.
python enrich_users.py                   # -> thousand_reddit_enriched_v2.jsonl + train/val/test_reddit_v2.jsonl
# 2. Build the Data Designer seed (pure transform).
SEED_INPUT=train_reddit_v2.jsonl SEED_OUTPUT=seed_gepa.jsonl python build_dd_seed.py
# 3. GEPA (offline prompt optimization) -> capture best_candidate["persona_prompt"].
python personaAdapter.py                 # paste the printed best prompt into ARM_PROMPTS in the DD pipeline
# 4. Scale + structure + judge (GEPA arm + Nemotron baseline at matched N; then the ablation ladder).
SEED_FILE=seed_gepa.jsonl N_PERSONAS=200 python persona_pipeline_datadesigner.py
```

The ablation ladder = re-running GEPA with `USE_TAU` / `USE_UTILITY` toggled to produce the
unoptimized / value-only / behavior-only / full prompts, each scaled through the same DD
pipeline over the same users (paired by `user_id` via a shared `IndexRange`).

## Offline validation status (this environment)

`python test_offline.py` passes with only `pydantic` + `pyyaml` + `data-designer-config`
installed (no API key, no heavy deps). It covers: alignment math (identical→1.0,
anti-correlated→0.0, degenerate→0.5, centering), enrichment leakage + determinism + schema,
seed builder no-leak, `PersonaProfile` validation, verbatim Schwartz pin, deterministic YAML,
and both Data Designer arm configs building/validating against the real v0.6.1 classes.

Not runnable here (need key + heavy/research deps): the live GEPA loop (`gepa`, `smolagents`,
`tau_bench`, `sentence-transformers`) and the Data Designer `create()` calls.
