# Spec: Monte Carlo Persona-Sampled Evaluation (E1 / E2 Main Run)

**Goal.** Replace the exhaustive |P|×|T|×|M| grid with a balanced, paired, persona-sampled design: each task is run K=10 times, each rollout with one persona drawn from the 200-persona pool, such that (a) every persona is used exactly once per block, (b) the persona→task assignment is identical across all models and all arms, and (c) the logged output supports paired statistics, cluster-bootstrap CIs, rank-shift analysis, and the later PPol fingerprint experiment.

End goal is being able to run Taubench and Statebench with this rollout in one script, similar to 
```bash
uv run python -m state_bench.scripts.merge_to_eval \
  --results-dir outputs/travel_persona_k100_bystem \
  --eval-dir ../reddit/Eval/statebench_schwartz_unopt \
  --domain travel
```

or 
```bash
uv run python -m state_bench.scripts.run_all_personas \
  --models gpt-oss \
  --domain travel --num-tasks 20 \
  --sim-model gemma \
  --eval-dir ../reddit/Eval/statebench_baseline_gpt_oss_travel \
  --no-persona --skip-existing
```
for statebench 

or for tau /home/yash/multiagent_human_worker/tau2-bench/scripts/run_all_personas.py or /home/yash/multiagent_human_worker/tau2-bench/scripts/run_all_personas_unique.py 


  # From persona pipeline JSONL (writes cached YAML under reddit/.personas_yaml_cache_*):
  python scripts/run_all_personas.py --models gpt-4o \
    --personas-jsonl ../reddit/personas_gepa_unopt.jsonl --eval-dir ../reddit/Eval/gepa_unopt

  # Baseline (no persona): one run per model, same layout as STATE-Bench --no-persona
  python scripts/run_all_personas.py --models gpt-4o --domain retail --num-tasks 20 \
    --eval-dir ../reddit/Eval/tau2_baseline --no-persona --skip-existing

if possible u can write this script as a wrapper to these 
---


## 1. Parameters (config file, not hardcoded)

| Name | Default | Notes |
|---|---|---|
| `P` | 200 | personas per arm (IDs shared across arms) |
| `T` | 20 | tasks per domain (confirm: per domain, not total) |
| `K` | 10 | rollouts per task per block (P = T × K must hold per block) |
| `B` | 1 | blocks (balanced permutations); harness must support extending to 2–3 without regenerating block 1 |
| `ARMS` | fixed_prompt, unopt, value_only, behavior_only, full_gepa | fixed_prompt = no persona injected (reviewer-demanded baseline) |
| `MODELS` | evaluated agent list | assert: user-sim model ∉ MODELS (hard fail, not warning) |
| `SIM_MODEL` | Gemma (pinned version + temperature + seed policy) | |
| `JUDGE_MODEL` | satisfaction critic (pinned version + frozen prompt hash) | identical across the entire matrix |
| `DOMAINS` | retail, telecom, airline | full design replicates per domain |
| `MASTER_SEED` | int | everything below derives deterministically from this |

## 2. Assignment generation (run once, freeze, commit to repo)

Produce `assignment_block{b}.csv` with columns:
`domain, task_id, slot (0..K-1), persona_id, rollout_seed`

Algorithm per domain per block:
1. Stratify the 200 persona IDs by persona cluster (or dominant Schwartz dimension) into S strata.
2. Assign personas to tasks so that each task receives K distinct personas AND each task's K personas are approximately balanced across strata (e.g., round-robin deal from shuffled strata). Each persona appears exactly once per block.
3. `rollout_seed = H(MASTER_SEED, block, domain, task_id, persona_id)` — deterministic hash, controls user-sim sampling seed.
4. Block b>1 uses a fresh permutation (different persona→task pairing), same persona set.

**Invariants (write asserts + a validation script):**
- Every persona_id appears exactly B times per domain.
- Every task has K distinct personas per block.
- Per-task strata counts differ by ≤1 from uniform.
- The same assignment file is consumed by every (model, arm) run — arms differ ONLY in which persona-text variant is injected for a given persona_id (fixed_prompt injects none).

## 3. Execution

- Total conversations per domain = B × T × K × |MODELS| × |ARMS| (e.g., 1 × 200 × 5 × 5 = 5,000). Print this and estimated cost before launch; require confirmation flag.
- Runs must be resumable: skip (model, arm, domain, task, persona, block) tuples that already have a completed transcript with matching config hash.
- Pin: agent temperature/seeds if the harness supports it; tau-2-bench version; distractor set (frozen); tool schemas.
- Concurrency is fine; determinism requirement applies only to assignment + seeds, not execution order.

### 3a. Pre-flight noise floor (blocking gate, before the full matrix)
Run the complete block-1 panel TWICE on ONE mid-tier model (same assignment, different sim seeds: derive with `H(..., replicate=0|1)`). Compute per-metric deltas between the two replicates (mean satisfaction, task success, P10 satisfaction, transfer rate). 
- If seed-to-seed delta ≥ expected model–model gaps (use gaps from prior runs as reference), set B=2 before launching the matrix.
- Log this as `noise_floor_report.json`; it goes in the paper's appendix.

## 4. Logging schema (JSONL, one record per rollout)

```
run_id, config_hash, block, domain, task_id, persona_id, arm, model,
sim_model, sim_seed, judge_model, judge_prompt_hash,
terminal_state ∈ {success, transfer, failure_no_transfer, max_turns, sim_error},
task_success (bool), transfer (bool), n_turns,
satisfaction_trajectory (per-turn deltas), satisfaction_final, satisfaction_cumulative,
full_transcript (all turns incl. tool calls, role, timestamps including errors),
persona_variant_hash (hash of injected persona text)
```

Notes:
- `terminal_state` is a single exclusive enum — this answers Reviewer 4's "why don't success + transfer sum to 1" definitionally.
- Full transcripts with turn metadata are mandatory: they are the input for the PPol 19-feature fingerprint experiment (tau-usi discriminator) — do not summarize or truncate.
- Store both point-wise and cumulative satisfaction deltas (Reviewer 4 asked about the aggregation choice; keep both so it can be reported either way).

## 5. Statistics module (`stats.py`)

All comparisons are PAIRED on (task_id, persona_id, block).

1. **Point estimates + CIs.** Per (model, arm, domain, metric): mean with 95% CI from cluster bootstrap. Bootstrap resamples clusters, not rollouts. Run two variants — resample persona_ids, and resample task_ids — report the wider interval (or a two-way/hierarchical bootstrap if straightforward). 10,000 resamples, seeded.
2. **Pairwise model differences.** For each model pair within an arm: per-(persona,task) metric deltas → paired bootstrap CI + sign test p-value.
3. **Rank-shift statistics (E1 headline).** Per domain: Kendall τ between the fixed_prompt ranking and each persona-conditioned ranking, with bootstrap CI on τ (recompute both rankings inside each bootstrap resample). Also report per-model rank change.
4. **Convergence analysis.** For n in {10, 25, 50, 100, 150, 200}: subsample n personas (respecting strata) from the completed panel, recompute metrics and rankings, repeat 500×; report estimate spread and P(ranking == full-panel ranking) vs n.
5. **Tail metric.** Replace raw min with 10th-percentile satisfaction (and optionally CVaR@10%), with bootstrap CI. Keep raw min only as a supplementary column.
6. **Arm contrasts (E2).** Within-persona paired deltas across arms (same persona_id, different variant), sign test / paired bootstrap — the set-3-vs-set-6-style causal comparison.
7. Report all p-values with Holm–Bonferroni correction across the model-pair family per domain; state the family explicitly.

## 6. Figures & tables (`plots.py`, matplotlib, one function per artifact, saves PDF + PNG)

1. **Table M1 (main):** per domain — rows = models; column groups = fixed_prompt vs full_gepa persona-conditioned; metrics = task success, mean satisfaction, P10 satisfaction, transfer rate; each cell = estimate [CI].
2. **Fig R1 (bump chart):** model ranks, fixed_prompt → persona-conditioned, one panel per domain; annotate Kendall τ [CI].
3. **Fig C1 (convergence):** metric estimate vs n personas with CI bands, per model, per domain; vertical markers at n=10 and n=200 (directly rebuts the N-sensitivity critique).
4. **Fig D1 (distributions):** per-model satisfaction ECDF (preferred over violins) over personas, P10 marked.
5. **Fig A1 (arm ablation):** paired per-persona deltas across the four arms (dot-plus-CI forest plot per model).
6. **Supp:** persona-cluster × model heatmap of mean satisfaction (which user subpopulations each model fails on).

## 7. Acceptance criteria

- [ ] Assignment validation script passes all invariants; assignment files committed.
- [ ] Sim-model contamination assert present and tested.
- [ ] Noise-floor pre-flight runs end-to-end and emits report before matrix launch.
- [ ] A dry run (2 tasks × 2 personas × 1 model × 2 arms) produces valid JSONL passing a schema check.
- [ ] stats.py reproduces identical numbers from identical JSONL (seeded bootstrap).
- [ ] Resume-from-partial works (kill mid-run, restart, no duplicates).
- [ ] terminal_state enum populated on every record, including error paths.
