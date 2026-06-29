# Persona-Conditioned Evaluation — Resubmission Plan + Code-Review Rubric (AAAI)

This doc does two jobs:
- **Sections A–C**: define the persona arms, how each is generated, and the experiments we will run.
- **Sections D–G**: the code-review rubric (verify the code matches what we told reviewers).
- **Section I**: records implementation changes already made in this repo.

Source of truth note: this file supersedes older notes in `reddit/handoff.md` and
`reddit/ammendment.md`. Where those docs conflict with current code, trust the codebase.

The human study (E5) is owned by the first author; its design is out of scope here and only referenced.

---

## A. Two axes (read this first — they are NOT the same experiment)

- **Axis A — source/content** (answers R3/R4: does behavioral grounding change the errors; matched-N source comparison). Varies *what grounds the persona*: Reddit+Schwartz vs Nemotron vs Reddit-no-psych.
- **Axis B — GEPA optimization signal** (answers R2: unoptimized / value-only / behavior-only / full). Varies *which signal GEPA optimizes the generation prompt against*; **all four rungs run on the same Reddit+Schwartz data.**

They share exactly one cell — **full-GEPA Reddit+Schwartz** — which is both the top GEPA rung and the headline system. Common mislabels to avoid: Nemotron is NOT "value-only GEPA" (it never touches GEPA); "Reddit-no-psych" is NOT "behavior-only GEPA" (it's an Axis-A content arm).

GEPA touches sets 1–4 and 6. GEPA never touches Nemotron (set 5). The *value* signal needs Schwartz (sets 1–4 only); the *behavioral* signals also apply to Reddit-no-psych (set 6).

---

## B. The six persona sets + generation recipes

All Reddit sets are generated from the **same 100 users** (smoke test: 10) so only the vector/optimization varies, not the people. Nemotron matches on *count* and *YAML schema*, not individuals. **The downstream YAML schema (fields the simulator reads) is held constant across all six sets** — this is non-negotiable for clean comparison.

Current data source: Reddit users now start from `reddit/thousand_users_raw.jsonl`
(`user_id`, `subreddits`, `history`). Legacy enriched/archetype files and their old
`target_vector` values should not be treated as source-of-truth vectors. The implemented
path enriches raw users first, then clusters/selects on PVQ-derived vectors.

| # | Set | Schwartz vector? | Reddit posts? | GEPA config | Role |
|---|---|---|---|---|---|
| 1 | Reddit+Schwartz, **unoptimized** | yes (new assignment) | yes | none (seed prompt) | GEPA ladder rung 1 |
| 2 | Reddit+Schwartz, **value-only GEPA** | yes | yes | `W_ALIGN=1, W_TAU=0, W_UTILITY=0`, gate on | GEPA rung 2 (ablation) |
| 3 | Reddit+Schwartz, **behavior-only GEPA** | yes (in input, not rewarded) | yes | `W_ALIGN=0, W_TAU+W_UTILITY on`, gate on | GEPA rung 3 + matched comparator for set 6 |
| 4 | Reddit+Schwartz, **full GEPA** | yes | yes | all signals on, gate on | GEPA rung 4 + **headline system** + comparator to Nemotron |
| 5 | **Nemotron** | no (OCEAN top-down) | no | none | demographic source baseline (already built) |
| 6 | Reddit, **no-psych** | **no — value block removed from prompt** | yes | current Axis-A smoke: none; final E3 comparator: behavior-only GEPA (`W_ALIGN=0, W_TAU+W_UTILITY on`, gate on) | content ablation; matched to set 3 when behavior-only GEPA prompt is available |

### Schwartz assignment — the finalized decision (keep Schwartz; three minimal reviewer-backed fixes)
We are NOT swapping frameworks. We keep Schwartz/PVQ and fix *measurement* and *validation*:

1. **Instrument-faithful assignment (R4 "not LLM-generated profiles").** `reddit/enrich_users.py` now administers **PVQ-40 items** to the model conditioned on heldout-removed USER-labeled text, then scores with the shared key in `reddit/pvq.py`. The LLM becomes a *respondent to a validated instrument*, not a profile generator. (Cite silicon-sampling precedent, e.g. Argyle et al., *Out of One, Many*.)
2. **Shuffle-controlled validation (R3 "arbitrary vs meaningful").** See E4 — this is the experiment that actually defeats the circularity charge.
3. **Report what we already compute + κ (R4 reporting / Cohen's κ).** Surface alignment/grounding/utility distributions; κ on the satisfaction judge (E5).

Plus one subtraction: the **PVQ round-trip (V≈V′) is a consistency/leakage diagnostic, NOT validation**, and must be labeled as such. It may remain a GEPA optimization signal (needed for the value-only ablation), but the headline system is full/behavior-weighted and the *validity* claim rests entirely on E4.

### Evidence-grounding (current code vs planned)
Current code implements attribution, holdout-after-attribution, quote-signal extraction, anti-hallucination grounding judge, register judge, and customer-role judge. It does **not** yet implement a structured per-trait evidence schema.

Planned next step: during persona generation, require per-trait evidence with a **two-tier** design:
- **Attested traits**: verbatim extractive spans, **string-matched against the user's USER-labeled corpus only** (never INTERLOCUTOR/QUOTED). Programmatically verify the span exists; reject/penalize traits whose evidence doesn't match.
- **Inferred traits**: licensed by a *set* of posts, checked by NLI/entailment, not exact match. Do NOT demand a verbatim quote for inferred traits (that's the mis-scoped-judge trap).
- Mark each trait's tier in the output. The provenance-verified attested-trait rate becomes a reported metric (E4).

### Held-constant across all six sets
Same YAML schema; vector kept in a **sidecar column**, never recited in the persona/YAML; demographics dropped from the Reddit arms (fabrication risk); same downstream tasks/seeds when evaluated.

---

## C. Smoke test before scaling
Immediate Axis-A smoke: generate **N=10 each** for Reddit+Schwartz unoptimized, Reddit-no-psych, and Nemotron. This gives the source/content comparison enough output to inspect while GEPA work proceeds separately.

Full six-set smoke before scaling: generate **N=10 per set (60 total)**, run the full generate→score→eval path end-to-end, and confirm: instrument-faithful assignment runs; GEPA configs produce *different* objectives; YAML schema identical across sets; no recitation leakage; no heldout leakage. **Only then** scale to N=100 and only then start STATE-Bench integration (E6).

Current run flow:
```bash
cd /home/yash/multiagent_human_worker/reddit
MAX_USERS=300 python enrich_users.py
python select_pvq_medoids.py --input thousand_users_pvq_enriched.jsonl --k 10 --out selected_users_pvq_k10.jsonl
SEED_INPUT=selected_users_pvq_k10.jsonl SEED_OUTPUT=seed_axis_a_k10.jsonl python build_dd_seed.py
python persona_pipeline_datadesigner.py --arm gepa_unopt --n 10 --seed seed_axis_a_k10.jsonl --out personas_axis_a_reddit_schwartz_unopt.jsonl --resume
python persona_pipeline_datadesigner.py --arm reddit_no_psych --n 10 --seed seed_axis_a_k10.jsonl --out personas_axis_a_reddit_no_psych.jsonl --resume
python persona_pipeline_datadesigner.py --arm nemotron --n 10 --out personas_axis_a_nemotron.jsonl --resume
```

---

## D. Experiments / run matrix

Models (all experiments): GPT-4o-mini, Gemini, Qwen3, GPT-OSS, Kimi (same as the original draft). Benchmarks: τ²-bench (home: telecom/airline/retail) + STATE-Bench (transfer: travel/customer-support/shopping). Matched **N=100** personas per set.

### E1 — Main ranking-shift + baseline table (partner P3, P5; R3 core — GATES THE PAPER)
- **Personas**: set 4 (full-GEPA Reddit+Schwartz).
- **Conditions**: fixed-prompt baseline vs persona-conditioned.
- **Benchmarks**: τ²-bench + STATE-Bench.
- **Metrics**: task success, cumulative + worst-case satisfaction, transfer rate; STATE-Bench native UX score.
- **Stats**: bootstrap CIs over the 100 personas; Kendall τ / Spearman between fixed-prompt and persona rankings; paired task-level deltas; variance across persona samples.
- **Deliverables**: the single clean comparison table (Fixed prompt vs Nemotron vs Reddit vs [Amazon if kept] vs STATE-Bench); **fix the Table 1 caption/text contradiction (L259-260) — separate the fixed-prompt baseline table from the persona-conditioned table.**

### E2 — GEPA ablation (partner P1; R2 exactly)
- **Personas**: sets 1, 2, 3, 4 (unopt / value-only / behavior-only / full), all Reddit+Schwartz.
- **Benchmark**: τ²-bench (subset of domains acceptable to save compute).
- **Metrics**: task success, satisfaction, ranking stability + persona-quality (from E4).
- **Foreground finding**: behavioral signals drive the gains; value-only regresses toward recitation / weaker held-out prediction. (Turns the circularity insight into a result.)

### E3 — Behavioral vs demographic source contrast (R3/R4; "does behavioral data change the errors")
- **Personas**: set 4 (or 3) Reddit · set 6 Reddit-no-psych · set 5 Nemotron. Matched N.
- **Clean single-variable pair**: set 3 vs set 6 (±Schwartz vector, behavior-only GEPA held constant).
- **System pair**: set 4 vs set 5 (best Reddit vs demographic baseline).
- **Metrics**: error-type distribution (Table 6 taxonomy, redefined to concrete agent failure modes), rankings, UX/satisfaction. Separate "safety-refusal/register" errors from substantive ones (profanity confound).

### E4 — Persona validation (partner P2; R3/R4)
- **Per set, before/after GEPA**: behavioral alignment, value alignment (PVQ round-trip — **labeled consistency/leakage diagnostic**), grounding, held-out utility.
- **Shuffle control (the key validity experiment)**: shuffle Schwartz vectors across users, **regenerate personas from shuffled vectors**, measure held-out prediction (`_utility_score`). Report effect size + significance vs real-vector personas. Run persona-mediated (shipped) + optionally vector-direct as robustness. *Honest stance: this can falsify our own claim — that's why it's credible.*
- **External anchor (if feasible)**: agreement of inferred Schwartz with PANDORA (Reddit personality corpus) or a second instrument.
- **Provenance metric**: % attested traits whose evidence string-matches USER spans.
- **Before/after GEPA**: show GEPA improves persona quality, not just that it ran.

### E5 — Human study (partner P4) — OWNED BY FIRST AUTHOR
Design TBD (out of scope here). Will report human-vs-LLM Cohen's κ on the satisfaction judge with a calibrated rubric (the κ=0.31 issue was construct, not sample size — the construct fix is already in the tau judge). Possibly persona-faithfulness as a separate study.

### E6 — STATE-Bench transfer (partner P0; R2/R3 — strongest single move)
- **Personas**: set 4. **Same models.** Integrate our persona-conditioned user simulator into STATE-Bench's user side **without changing our pipeline**.
- **Report**: whether ranking shifts persist; divergence between task success and STATE-Bench's native UX score.
- **Why it's load-bearing**: independent simulator + independent UX judge → the cleanest answer to R3's "could be artifacts of YOUR Gemma sim / GLM critic."
- **Gating check first**: can STATE-Bench's *user* simulator be conditioned on our persona YAML? (Read `state_bench/`, `USE_CUSTOM_CLIENT.md`.) Confirm we *replace* its user behavior, not stack on top. If injection isn't clean, fall back to UserBench.

---

## E. Code-review rubric — reviewer mitigation map

Status: ✅ in code · 🟡 in-progress · 📝 write-up · ⚠️ reframe/partial (do not rubber-stamp). For each: report `PASS / FAIL / NOT FOUND / REFRAME`, cite file+function, quote code/output. No runnable artifact = `FAIL`.

| Rev | Critique | Mitigation | Status |
|---|---|---|---|
| R1 | Ranking shifts hard to interpret; target distribution unjustified | Reframe to stress-test *specifications* + behavioral-vs-demographic contrast (E3) | ⚠️ reframe, not an empirical resolution |
| R2 | GEPA underspecified | Document seed prompt, reflective loop, weighted objective + gate | ✅ code; 📝 write-up |
| R2 | unopt/value-only/behavior-only/full ablations | E2 (sets 1–4) | ✅ verify toggles change the objective (F.4) |
| R2 | Transfer beyond customer-support family | STATE-Bench travel/shopping (E6) | 🟡 contingent on injection (F.6) |
| R2 | Presentation | Craft pass | 📝 last |
| R3 | Synthetic chain unvalidated; shifts may be artifacts | STATE-Bench independent sim+judge (E6); shuffle control (E4); contamination fixes | 🟡 |
| R3 | Human validation weak (κ=0.31) | Construct fix in tau judge (✅) + re-validation (E5) | ✅ / 🟡 |
| R3 | **Central claim unsupported; Table 1 caption contradicts text** | E1 (separate tables, CIs, rank corr, paired deltas) | 🟡 **gating** |
| R3 | Personas not tied to deployment population | Stress-test reframe + shuffle-controlled grounding (E4) | ⚠️/🟡 |
| R4 | Persona quality not shown | Report distributions (E4) | ✅ computed; 📝 surface |
| R4 | Cohen's κ for judge | E5 | 🟡 |
| R4 | **More frameworks, not LLM-generated profiles** | Keep Schwartz + instrument-faithful PVQ assignment + shuffle control + external anchor (B, E4) | ✅ PVQ assignment in code; 🟡 shuffle/external anchor pending |
| R4 | Reads AI-generated; Table 6 unclear | Redefine taxonomy (E3) + craft pass | 📝 last |
| R4 | Source-affects-rankings needs matched N | E3, matched N=100 | 🟡 verify (F.3) |
| R4 | Q: success + transfer ≠ 1? | Define both (not complementary) | 📝 |
| R4 | Q: cumulative vs point-wise deltas? | Justify or switch | 📝 |

---

## F. Verifiable code checks

### F.1 Circularity / construct integrity (HIGHEST)
- `_score_schwartz_alignment` / `_administer_pvq_test`: persona is **administered PVQ-40 item-by-item**, compared to target — NOT reading numbers off the persona. `FAIL` otherwise.
- **Assignment side**: `administer_pvq_to_user_text` (`enrich_users.py`) administers PVQ-40 to heldout-removed USER text and scores with the canonical key in `pvq.py`. If any path returns a direct free-form 10-number profile from a "Quantitative Psychologist" prompt → `FAIL` (this is R4's exact complaint).
- Generation prompts forbid naming values / printing numbers (quote it).
- Grounding is multiplicative (`score = base * grounding`), not additive.

### F.2 Consistency vs validity
- Is the PVQ round-trip used as a GEPA reward? It may be (for the value-only arm) but must be **labeled a diagnostic, not validation**, and must not dominate the headline config.
- **Shuffle control exists?** (E4) Personas regenerated from shuffled vectors + held-out prediction + significance test. Confirm present (previously `NOT FOUND`).
- **External anchor exists?** (PANDORA / 2nd instrument). Confirm present or absent.

### F.3 The six persona sets are runnable as distinct configs
- All four GEPA rungs (sets 1–4) generate from Reddit+Schwartz and differ only by signal weights.
- **Set 6 switch**: `Arm.REDDIT_NO_PSYCH` exists in `persona_pipeline_datadesigner.py`; its prompt must contain no `schwartz_json`, no "Schwartz", and no latent value profile block. Current lightweight test checks this.
- Set 5 (Nemotron): no posts → grounding/utility undefined → downstream-eval only; harness must not crash or silently score 0.
- **YAML schema identical across all six sets** (`FAIL` if fields differ).
- **Matched N**, same downstream tasks/seeds; Reddit sets share the same 100 users.

### F.4 GEPA ablation ladder
- `W_ALIGN/W_TAU/W_UTILITY` and `USE_*` toggles each change the objective (no no-ops). `TRAIN_N` ≥ 8; flag if seed prompt is single-domain (retail) given STATE-Bench transfer.

### F.5 Evidence-grounding
- Current code has USER/INTERLOCUTOR/QUOTED attribution, heldout-after-attribution, quote signals, grounding judge, register judge, and role-check judge.
- Per-trait evidence spans and the two-tier attested/inferred schema are **planned but not implemented**. Mark `NOT FOUND` until a structured evidence field + verifier exists.

### F.6 STATE-Bench feasibility (gates E6)
- Can the **user** simulator take our persona YAML (`state_bench/`, `USE_CUSTOM_CLIENT.md`)? Replace, not stack. Confirm plan uses native UX score.

### F.7 Contamination / leakage
- Interlocutor `>` labeling; USER-vs-QUOTED attribution; **holdout AFTER attribution**; `quote_signals` derived **after** holdout; PVQ assignment on heldout-removed USER corpus.

### F.8 Gating statistics (E1)
- Distinct fixed-prompt vs persona-conditioned tables; bootstrap CIs; Kendall τ/Spearman; paired task-level deltas. `FAIL` if ranking shift is a single point estimate with no uncertainty.

---

## G. Reframes/partials — do NOT mark green
1. **"Which distribution is real" (R1/R3)** → stress-test reframe = rhetoric, not evidence.
2. **"More frameworks" (R4)** → we keep Schwartz and answer via *instrument-faithful assignment + shuffle control + external anchor*. This is a deliberate scope decision, not a framework swap. Verify the three fixes exist (F.1, F.2); the *taxonomy* is unchanged on purpose.
3. **STATE-Bench (R2/R3)** → strongest move but unverified; check injection FIRST (F.6).
4. **Table 1 caption + the two R4 questions** → trivial to fix, trivial to forget; the caption contradiction is what R3 built the rejection around.

---

## H. Priority order (don't let persona generation displace the gating work)
0. Smoke test N=10/set (Section C) — validates the whole pipeline cheaply.
1. **E1** — ranking-shift stats + Table 1 caption (gates everything).
2. **E6** — STATE-Bench transfer (verify injection first).
3. **E2** — GEPA ablation.
4. **E3** — behavioral-vs-demographic contrast, matched N.
5. **E4** — persona validation incl. shuffle control + external anchor.
6. **E5** — human study (first author).
7. Craft pass: prose, table numbering, Table 6 taxonomy, unanswered reviewer questions.

---

## I. Implementation log — current codebase state (2026-06-29)

This section records the implemented changes from the from-scratch Axis-A PVQ work. It is the current handoff for persona generation; older `reddit/handoff.md` and `reddit/ammendment.md` describe earlier states and contain stale items.

### Implemented
- **Canonical raw source**: `reddit/enrich_users.py` now defaults to `reddit/thousand_users_raw.jsonl` and writes `reddit/thousand_users_pvq_enriched.jsonl`. The script no longer defaults to old enriched files or legacy vectors.
- **Instrument-faithful Schwartz assignment**: added `reddit/pvq.py` with PVQ-40 items, Schwartz scoring key, item validation, PVQ value means, and `[0,1]` normalization. `enrich_users.py` now uses `administer_pvq_to_user_text`, not a direct 10-number profile prompt.
- **PVQ sidecars**: enriched records now persist `pvq_item_scores`, `pvq_value_means`, and normalized `target_vector`.
- **Contamination fixes**: enrichment keeps the attribution sequence from the amendment: deterministic pre-parse, `INTERLOCUTOR` labeling for `>` lines, LLM USER/QUOTED attribution, holdout after attribution, quote-signal extraction after holdout, then PVQ assignment only on heldout-removed USER text.
- **Shared PVQ scoring**: `personaAdapter.py` imports shared PVQ scoring for the persona PVQ round-trip. The round-trip is documented as a consistency/leakage diagnostic and GEPA reward signal, not validation.
- **Seed build**: `reddit/build_dd_seed.py` now renders `clean_corpus` labels into `user_corpus`, omits Reddit demographics, carries `quote_signals`, and includes PVQ audit sidecars.
- **PVQ medoid selection**: added `reddit/select_pvq_medoids.py`, which clusters normalized PVQ vectors and selects medoids for `k=10` smoke or `k=100` scale runs. It replaces the legacy `reddit/cluster.py`, which was deleted because it clustered stale direct-inference vectors.
- **Axis-A no-psych arm**: `persona_pipeline_datadesigner.py` now has `Arm.REDDIT_NO_PSYCH`. Its prompt excludes `schwartz_json`, "Schwartz", and latent value-profile language; it also skips the value-alignment judge because there is no value input for that arm.
- **Schema consistency**: Reddit+Schwartz, Reddit-no-psych, and Nemotron all compile into the same `PersonaProfile` / `persona_yaml` schema. Reddit arms leave demographics unset; Nemotron may populate demographics.
- **Stale tests removed**: deleted `reddit/test_offline.py`; added `reddit/test_axis_a_pvq.py` for PVQ scoring, seed schema, heldout leakage, selector sidecars, and prompt-boundary checks.

### Verified locally
- `python reddit/test_axis_a_pvq.py` passes under the repo `.venv/bin/python`.
- `python -m py_compile` passes for `reddit/pvq.py`, `reddit/enrich_users.py`, `reddit/select_pvq_medoids.py`, `reddit/build_dd_seed.py`, `reddit/persona_pipeline_datadesigner.py`, `reddit/test_axis_a_pvq.py`, and `reddit/personaAdapter.py`.
- Data Designer configs build for `gepa_unopt`, `reddit_no_psych`, and `nemotron`; observed columns:
  - `gepa_unopt`: `persona_paragraph`, `persona`, `value_alignment`, `grounding`, `register`, `role_check`
  - `reddit_no_psych`: `persona_paragraph`, `persona`, `grounding`, `register`, `role_check`
  - `nemotron`: `person`, `persona`

### Not yet done
- Live enrichment/persona generation was not run after these changes; it requires model/API calls.
- Shuffle-controlled validation (E4) is still pending.
- External anchor (PANDORA or second instrument) is still pending.
- Structured two-tier evidence output and verifier are still pending.
- Full GEPA value-only / behavior-only / full prompt runs are still pending; the implemented immediate Axis-A arms are Reddit+Schwartz unoptimized, Reddit-no-psych, and Nemotron.
