# E3 — grounding diagnosticity: run handoff

Goal: run the blind-critic pipeline over the four grounding arms and produce the
composition figure data + outcomes table + permutation control.

## 0. What's in the box
- `score_e3.py` — the pipeline (imports `score_value_fidelity.py`, which imports `pvq.py`; all three live here in `reddit/`).
- `SourceAB/` — one JSONL per arm. **Arm is taken from the filename**, not the in-file `arm` field:
  | file | arm |
  |---|---|
  | `nemotron.jsonl` | nemotron |
  | `randomreddit.jsonl` | reddit_generic ← **you add this one** |
  | `domain_specificreddit.jsonl` | reddit_matched |
  | `amazon.jsonl` | amazon |
- Policy text is read from `../tau2-bench/data/tau2/domains/retail/policy.md` (in the repo, keep the tree intact).

The filename **must be exactly `randomreddit.jsonl`** or that arm is silently skipped.

## 1. One-time setup
```bash
cd reddit
pip install -r requirements.txt          # openai, numpy, python-dotenv (+ project deps)
# API key — .env is git-ignored, so it did NOT come with the push.
# Put YOUR OpenRouter key in the REPO-ROOT .env (one level above reddit/):
echo 'OPEN_ROUTER_API_KEY=sk-or-...' >> ../.env
```
Without the key the script errors on import.

## 2. Drop in the last arm
Put `randomreddit.jsonl` into `reddit/SourceAB/`. That makes all four arms present.

## 3. Run
```bash
cd reddit
E3_MAX_WORKERS=16 python score_e3.py all
```
`all` = label (failure critic) → satisfy (per-user-turn satisfaction) → score
(metrics + permutation control). Critic model is `qwen/qwen3.7-flash` via OpenRouter.

**Runtime:** critic ≈23s/call, satisfaction ≈8s/call. ~800 traces:
- 16 workers → ~25 min · 8 workers (default) → ~50 min.

It's **resumable** — every trace is cached by content hash
(`e3_critic_cache.jsonl`, `e3_satisfaction_cache.jsonl`). Re-running skips
finished traces, so an interrupted run just continues. These caches regenerate;
no need to commit them.

## 4. Output
- Prints the composition table, the compact 5×N table
  (`task_succ | cum_sat | worst_sat | sat_spread(SD) | JS_to_nemotron`),
  paired task-difficulty deltas, and the permutation p-values.
- Writes everything to `e3_aggregate.json` (composition vectors, JS-to-nemotron
  with cluster-bootstrap CIs, full JS matrix, per-category prevalence, outcomes,
  spread, paired deltas, control).

## Notes / gotchas
- **Don't mix a MOCK run with the real one.** `MOCK_LLM=1` writes fake results
  into the same caches. If you want a free wiring check, use a throwaway dir so
  it can't poison real caches:
  ```bash
  E3_SOURCE_DIR=/tmp/e3mock MOCK_LLM=1 python score_e3.py mock-data
  E3_SOURCE_DIR=/tmp/e3mock MOCK_LLM=1 python score_e3.py all
  ```
- Re-run a single arm without touching the rest (cache handles the others):
  ```bash
  python score_e3.py label reddit_generic
  python score_e3.py satisfy reddit_generic
  python score_e3.py score
  ```
- User turns carry `<internal_monologue>...</internal_monologue>`; it is stripped
  before the critic ever sees the transcript (keeps labeling arm-blind and judges
  only spoken dialogue). Agent = `assistant` turns; `tool` turns are context only.
- All paths are anchored to the script location, so cwd doesn't matter as long as
  the repo tree is intact.
