# Handoff — Persona Generation Rework (AAAI), for local Claude Code

> Self-contained. The rework is **already merged** as commit `e8d3fca` on branch
> `reddit-tau2` (`reddit/` dir). This doc explains state, the GEPA↔Data Designer
> architecture, the **3-phase run sequence you asked for** (Nemotron → unoptimized
> GEPA prompt → optimized GEPA prompt), exact commands, integration seams, and the
> next tasks.

---

## 0. TL;DR state
- All code is in `reddit/`. Offline tests pass: `python reddit/test_offline.py`.
- **Not yet run live** against the NRP endpoint (needs `NAUT_API_KEY` + `gepa`/`smolagents`/`tau_bench`/`sentence-transformers`). Everything is wired and compiles; the live numbers are the open item.
- Construct = **Schwartz values + held-out behavior** (OCEAN dropped from the source arm; Nemotron keeps its native demographic/OCEAN population).

## 1. Why (one paragraph)
COLM rejection's rebuttal-killer was κ=0.31. Root cause (in code, not noise): the
old prompt made the persona **print its Schwartz JSON and recite values-by-number**,
and the scorers read those numbers back → it measured `persona↔value-vector`
(transcription) and never `persona↔behavior`. This PR removes the leak, scores value
alignment over the **full 10-dim vector**, adds a **behavioral** signal (predict a
held-out post), decouples the tau/satisfaction judge from the persona's self-narration,
and moves scale-out to a **fixed schema** (matched-N comparable). Expect alignment
numbers to **drop** — that's the honest result.

## 2. Architecture (the key mental model)
- **GEPA (`personaAdapter.py`) is OFFLINE and optimizes ONE prompt *string*.** It runs
  once on a small train/val set of users and emits `gepa_result.best_candidate["persona_prompt"]`.
  It is NOT in the per-persona scale loop.
- **Data Designer (`persona_pipeline_datadesigner.py`) is the scale harness.** It takes
  a *fixed* paragraph prompt and applies it over N seed users → fixed `PersonaProfile`
  schema → deterministic YAML → judge columns. Replaces the old `pipeline.py` loop.
- **GEPA = learn the prompt; Data Designer = apply it at scale + structure + measure.**

## 3. Files
| File | Role |
|---|---|
| `alignment.py` | Pure-Python full-vector Schwartz alignment (mean-centered cosine / Spearman). Unit-tested. |
| `personaAdapter.py` | GEPA loop: de-leaked prompt, full-vector alignment, behavioral tau judge, `_grounding_score`/`_utility_score`, weighted+gated score, env-driven sizes, ablation toggles. |
| `enrich_users.py` | Per-user prep: hold out 1 post (leakage-safe), re-infer Schwartz on held-out-removed corpus, infer demographics, train/val/test split. `MOCK_LLM` offline. |
| `build_dd_seed.py` | Pure transform: enriched v2 → DD seed rows. |
| `persona_pipeline_datadesigner.py` | DD scale harness: `PersonaProfile` schema, verbatim Schwartz pin, deterministic YAML, value/grounding judges, Nemotron baseline, matched-N via `IndexRange`. |
| `pipeline.py` | De-leaked `REDDIT_PROMPT`; `__main__` superseded by the DD pipeline. |
| `test_offline.py`, `requirements.txt`, `REWORK_NOTES.md` | Tests, deps, methods note. |

## 4. Score model (in `personaAdapter.evaluate`)
`score = (W_ALIGN·align + W_TAU·tau/5 + W_UTILITY·utility) × grounding_gate`
defaults `0.4 / 0.3 / 0.3`, grounding multiplies (anti-hallucination gate, not additive).
Toggles (env): `USE_TAU`, `USE_UTILITY`, `USE_GROUNDING_GATE`, `SIMILARITY_METRIC` (`cosine`|`spearman`),
`W_ALIGN/W_TAU/W_UTILITY`, `TRAIN_N/VAL_N/MAX_METRIC_CALLS`. **These toggles ARE the ablation ladder.**

---

## 5. Environment setup (one time)
```bash
cd reddit
pip install -r requirements.txt          # NOTE: install tau_bench from its source (not on PyPI)
export NAUT_API_KEY=...                   # NRP Nautilus; required for any live run, this is in the .env file right now 

# Sanity check with no key/heavy deps:
pip install pydantic pyyaml data-designer-config==0.6.1
python test_offline.py                    # expect: ALL OFFLINE TESTS PASSED, this passes 
```

### Data prep (one time, needs key)
```bash
# Enrich the per-user history file: hold out a post, re-infer Schwartz, infer demographics, split.
python enrich_users.py                    # -> thousand_reddit_enriched_v2.jsonl + train/val/test_reddit_v2.jsonl

# Build the Data Designer seed for the eval users (these are the users phases 2 & 3 generate for).
# Use the SAME seed file for phases 2 and 3 so they're paired by user_id.
SEED_INPUT=test_reddit_v2.jsonl SEED_OUTPUT=seed_eval.jsonl python build_dd_seed.py
```

---

## 6. THE 3-PHASE RUN SEQUENCE (your requested order)

Set N once and reuse it so all three arms are matched-N. Phases 2 & 3 share `seed_eval.jsonl`
and `seed_start=0` → **paired by user_id**. Phase 1 (Nemotron) is synthetic, so unpaired but matched-N.

```bash
export N=200
```

### Phase 1 — Baseline: Nemotron demographic personas
No seed, no GEPA, native demographic/OCEAN population. Judged only on downstream realism/diversity.
```bash
python -c "
import persona_pipeline_datadesigner as P
df = P.generate(P.Arm.NEMOTRON_BASELINE, None, n_personas=$N)
df.to_json('personas_nemotron.jsonl', orient='records', lines=True)
print('wrote personas_nemotron.jsonl', len(df))
"
```

### Phase 2 — Unoptimized GEPA prompt personas
This is the **seed prompt with NO optimization** — the "does the prompt scaffold alone help"
control. Before running, set `ARM_PROMPTS[Arm.GEPA_UNOPT]` to the de-leaked seed prompt in
**Jinja** form (ready-to-paste version in §7). Then:
```bash
python -c "
import persona_pipeline_datadesigner as P
df = P.generate(P.Arm.GEPA_UNOPT, 'seed_eval.jsonl', n_personas=$N, seed_start=0)
df.to_json('personas_gepa_unopt.jsonl', orient='records', lines=True)
print('wrote personas_gepa_unopt.jsonl', len(df))
"
```

### Phase 3 — Optimized GEPA prompt personas
First run GEPA (offline) to produce the optimized prompt, then scale it.
```bash
# (a) Optimize. Trains on train_reddit_v2.jsonl; prints best_candidate["persona_prompt"].
TRAIN_N=8 VAL_N=6 MAX_METRIC_CALLS=150 python personaAdapter.py | tee gepa_run.log

# (b) Translate the printed prompt's placeholders to Jinja (§7 mapping) and paste it into
#     ARM_PROMPTS[Arm.GEPA_FULL] in persona_pipeline_datadesigner.py.

# (c) Scale it over the SAME eval users as Phase 2.
python -c "
import persona_pipeline_datadesigner as P
df = P.generate(P.Arm.GEPA_FULL, 'seed_eval.jsonl', n_personas=$N, seed_start=0)
df.to_json('personas_gepa_full.jsonl', orient='records', lines=True)
print('wrote personas_gepa_full.jsonl', len(df))
"
```

Each `personas_*.jsonl` row carries the structured `persona`, the deterministic `persona_yaml`,
and (for the GEPA arms) the `value_alignment` and `grounding` judge columns. Compare the three
files: Nemotron vs unoptimized vs optimized.

---

## 7. ⚠️ Integration seam: prompt placeholder syntax (read before Phase 2/3)
GEPA prompts use **single-brace** placeholders consumed by `str.replace` in
`personaAdapter.evaluate`: `{history_str}`, `{anchor_demographics}`, `{psych_vector_str}`,
`{subreddit}`. The Data Designer paragraph column uses **Jinja2** over **seed column names**.
When moving any prompt (the seed for Phase 2, or the GEPA output for Phase 3) into
`ARM_PROMPTS`, translate:

| GEPA prompt (single-brace) | Data Designer (Jinja, seed column) |
|---|---|
| `{history_str}` | `{{ user_corpus }}` |
| `{anchor_demographics}` | `{{ demographics }}` |
| `{psych_vector_str}` | `{{ schwartz_json }}` (latent grounding only) |
| `{subreddit}` | `{{ subreddits }}` |

**Ready-to-paste de-leaked seed prompt for Phase 2** (`ARM_PROMPTS[Arm.GEPA_UNOPT]`):
```text
You are an expert Psychological Profiler.
Generate a persona definition that is self-explanatory. The persona must be so coherent and
psychologically vivid that an AI acting as this person will naturally deduce how to behave in
any situation purely by reading the description.

Do not write specific rules. Write the psychological reasoning behind behavior.

=== INPUT DATA ===
1. DEMOGRAPHIC ANCHOR:
{{ demographics }}

2. INFERRED PSYCHOLOGICAL PROFILE (latent grounding from {{ subreddits }}):
{{ schwartz_json }}
(Use this profile ONLY to shape behavior. Do NOT name these values or print any numbers.)

3. BEHAVIORAL SAMPLES:
{{ user_corpus }}

=== OUTPUT FORMAT ===
### 1. CORE IDENTITY
(First-person: "I am a [Age] year old [Job]...")
### 2. PSYCHOLOGICAL DRIVERS
(Why they act as they do, in plain language — never named values or numbers.)
### 3. INTERNAL MONOLOGUE STYLE
(How they think under pressure; two example thoughts prefixed "Example 1:" / "Example 2:" in a
support interaction that reveal priorities IMPLICITLY, naming no value and citing no number.)

=== YOUR RESPONSE ===
```

## 8. Known limitations / risks (for the rebuttal + before trusting numbers)
- **No live run yet.** GEPA loop and DD `create()` are unexercised end-to-end. Confirm
  `result.load_dataset()` shape and that `SamplerType.PERSON` yields the expected Nemotron population.
- **`_utility_score`** is a semantic-similarity proxy for held-out prediction — expect noise;
  spot-check its spread on ~10 users before leaning on it.
- **Circularity is reduced, not eliminated.** The target Schwartz vector is LLM-derived (`qwen3`)
  from the same posts; we re-infer it on the held-out-removed corpus and disclose this. Frame as
  "behaviorally-grounded *specifications* → more realistic *simulated* interactions," not real reactions.
- **Matched-N pairing:** phases 2 & 3 must use identical `seed_eval.jsonl`, `N`, and `seed_start=0`.

## 9. Suggested first tasks in local Claude Code
1. **Add a CLI** to `persona_pipeline_datadesigner.py` `__main__`: `--arm`, `--n`, `--seed`,
   `--seed-start`, `--out` (replaces the `python -c` one-liners above; makes the 3 phases first-class).
2. **Live smoke test at N=5** for each of the 3 phases; verify YAML + judge columns populate.
3. Wire `ARM_PROMPTS[GEPA_VALUE_ONLY]` / `GEPA_BEHAVIOR_ONLY` to the corresponding GEPA runs
   (`USE_TAU=0 USE_UTILITY=0` etc.) to complete reviewer 2's ablation ladder.
4. Build a comparison notebook: load the 3 `personas_*.jsonl`, report value_alignment/grounding
   distributions + persona diversity, Nemotron vs unopt vs optimized.

## 10. Git note
Commit `e8d3fca` is applied on `reddit-tau2`. It shows "Unverified" only because the sandbox
couldn't GPG-sign; author/committer are correct (`Claude <noreply@anthropic.com>`). No action needed.
