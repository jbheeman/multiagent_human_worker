# Commands

Run from the repo root unless a block says otherwise:

```bash
cd /home/yash/multiagent_human_worker
```

## Axis A Persona Generation

### 1. Enrich raw Reddit users with PVQ

```bash
cd /home/yash/multiagent_human_worker/reddit
MAX_USERS=300 python enrich_users.py
```

Output:

```text
reddit/thousand_users_pvq_enriched.jsonl
```

### 2. Select smoke/eval/GEPA splits

First select the 10-user smoke set:

```bash
cd /home/yash/multiagent_human_worker/reddit
python select_pvq_medoids.py \
  --input thousand_users_pvq_enriched.jsonl \
  --k 10 \
  --out selected_users_pvq_k10.jsonl
```

Select the 100-user eval set, forcing the smoke users into it:

```bash
python select_pvq_medoids.py \
  --input thousand_users_pvq_enriched.jsonl \
  --k 100 \
  --include selected_users_pvq_k10.jsonl \
  --out selected_users_pvq_eval_k100.jsonl
```

Select disjoint GEPA train/test users:

```bash
python select_pvq_medoids.py \
  --input thousand_users_pvq_enriched.jsonl \
  --k 50 \
  --exclude selected_users_pvq_eval_k100.jsonl \
  --out selected_users_pvq_gepa_train_k50.jsonl

python select_pvq_medoids.py \
  --input thousand_users_pvq_enriched.jsonl \
  --k 25 \
  --exclude selected_users_pvq_eval_k100.jsonl \
  --exclude selected_users_pvq_gepa_train_k50.jsonl \
  --out selected_users_pvq_gepa_test_k25.jsonl
```

### 3. Build Data Designer seeds

Smoke seed:

```bash
SEED_INPUT=selected_users_pvq_k10.jsonl \
SEED_OUTPUT=seed_axis_a_k10.jsonl \
python build_dd_seed.py
```

Eval seed:

```bash
SEED_INPUT=selected_users_pvq_eval_k100.jsonl \
SEED_OUTPUT=seed_axis_a_eval_k100.jsonl \
python build_dd_seed.py
```

GEPA train/test seeds:

```bash
SEED_INPUT=selected_users_pvq_gepa_train_k50.jsonl \
SEED_OUTPUT=train_reddit_pvq_k50.jsonl \
python build_dd_seed.py

SEED_INPUT=selected_users_pvq_gepa_test_k25.jsonl \
SEED_OUTPUT=test_reddit_pvq_k25.jsonl \
python build_dd_seed.py
```

### 4. Generate N=10 Axis A smoke personas

```bash
python persona_pipeline_datadesigner.py \
  --arm gepa_unopt \
  --n 10 \
  --seed seed_axis_a_k10.jsonl \
  --out personas_axis_a_reddit_schwartz_unopt.jsonl \
  --resume

python persona_pipeline_datadesigner.py \
  --arm reddit_no_psych \
  --n 10 \
  --seed seed_axis_a_k10.jsonl \
  --out personas_axis_a_reddit_no_psych.jsonl \
  --resume

python persona_pipeline_datadesigner.py \
  --arm nemotron \
  --n 10 \
  --out personas_axis_a_nemotron.jsonl \
  --resume
```

### 5. Generate N=100 Axis A eval personas

Use after the N=10 smoke set looks good:

```bash
python persona_pipeline_datadesigner.py \
  --arm gepa_unopt \
  --n 100 \
  --seed seed_axis_a_eval_k100.jsonl \
  --out personas_axis_a_reddit_schwartz_unopt_k100.jsonl \
  --resume

python persona_pipeline_datadesigner.py \
  --arm reddit_no_psych \
  --n 100 \
  --seed seed_axis_a_eval_k100.jsonl \
  --out personas_axis_a_reddit_no_psych_k100.jsonl \
  --resume

python persona_pipeline_datadesigner.py \
  --arm nemotron \
  --n 100 \
  --out personas_axis_a_nemotron_k100.jsonl \
  --resume
```

## Validation Checks

```bash
cd /home/yash/multiagent_human_worker
.venv/bin/python reddit/test_axis_a_pvq.py
.venv/bin/python -m py_compile \
  reddit/pvq.py \
  reddit/enrich_users.py \
  reddit/select_pvq_medoids.py \
  reddit/build_dd_seed.py \
  reddit/persona_pipeline_datadesigner.py \
  reddit/personaAdapter.py
```

## STATE-Bench Review Notes

Do not treat `[TERMINAL: success]` as benchmark pass/fail. Check scorer fields:

```bash
python - <<'PY'
import json
path = "/home/yash/multiagent_human_worker/STATE-Bench/outputs/travel_persona/persona_001/1-cancel_economy_domestic.json"
data = json.load(open(path))
for key in [
    "terminal_state",
    "task_completion_pass",
    "state_requirements_met",
    "task_requirements_met",
    "ux_score",
    "scoring",
]:
    print(key, data.get(key))
print("state_diff:", data.get("state_diff"))
PY
```

For official comparisons, make sure STATE-Bench scoring populates:

```text
state_requirements_met
task_requirements_met
task_completion_pass
ux_score
ux_reasoning
```

---

# Fresh-machine setup + STATE-Bench overnight run

Clone the repo on a new machine, (re)generate personas, and leave STATE-Bench running
overnight on a fixed **20 tasks per persona** (same 20 across all personas). tau2 is assumed
already installed. Everything runs on **Nautilus** (free, OpenAI-compatible) — no Azure.
Paths assume `~/multiagent_human_worker`; adjust if cloned elsewhere.

## A. Prereqs (one-time per machine)

STATE-Bench needs **Python 3.12+**; `uv` fetches it automatically.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## B. Recreate secrets — `.env` files are gitignored and DO NOT come with the clone

STATE-Bench reads `STATE-Bench/.env`; the reddit pipeline reads the root `.env`.

```bash
cat > ~/multiagent_human_worker/STATE-Bench/.env <<'EOF'
NAUT_API_KEY=<your-nautilus-key>
NAUT_API_BASE=https://ellm.nrp-nautilus.io/v1
NAUT_VERIFY_SSL=false
NAUT_MODEL=gpt-oss
# STATE_BENCH_EVAL_* only needed for the official Azure GPT-5.4 judge — we do NOT use it.
EOF

echo 'NAUT_API_KEY=<your-nautilus-key>' > ~/multiagent_human_worker/.env
```

## C. Install STATE-Bench

```bash
cd ~/multiagent_human_worker/STATE-Bench
uv sync
# sanity: custom classes + runner import
uv run python -c "from state_bench.agents.loader import load_root_client_class as L; print('client ok:', hasattr(L('NautilusClient'),'complete_json'))"
uv run python -m state_bench.scripts.run_unofficial --help >/dev/null && echo "runner ok"
```

## D. Generate personas

Use the **Axis A Persona Generation** section at the top of this file (needs the repo-root
`.venv`; create it with `python3 -m venv .venv && pip install -r requirements.txt` if the
fresh clone lacks it). The `gepa_unopt` smoke arm writes the YAML cache the benchmark reads:

```
reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt/*.yaml
```

If you already have persona YAMLs, skip generation and point `--persona-dir` at any folder
of persona `*.yaml`.

## E. Run STATE-Bench overnight — 20 tasks × all personas

The fixed 20 tasks = task numbers **1–20** in the domain (same set for every persona because
`--task` is explicit). Derive them to avoid typos:

```bash
cd ~/multiagent_human_worker/STATE-Bench
set -a && source .env && set +a

DOMAIN=travel
TASKS=$(ls state_bench/domains/$DOMAIN/tasks/*.json | xargs -n1 basename | sed 's/.json//' \
        | sort -t- -k1 -n | head -20)
echo "$TASKS"    # expect 1-... through 20-...
```

Launch detached (100 personas × 20 tasks = 2000 sequential runs → many hours, ideal
overnight):

```bash
nohup uv run python -m state_bench.scripts.run_unofficial \
  --domain $DOMAIN \
  --task $TASKS \
  --persona-dir ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt_k100 \
  --sim-model gemma \
  --agent-model gpt-oss \
  --no-satisfaction \
  --skip-existing \
  --output-dir outputs/${DOMAIN}_persona_k100_bystem \
  > outputs/run_${DOMAIN}_k100_bystem.log 2>&1 &

echo "PID $! — tail -f outputs/run_${DOMAIN}_k100_bystem.log"
```

Each task writes, at completion: `terminal_state` (persona satisfaction), the deterministic
`state_requirements_met` (free), and (unless `--no-satisfaction`) turn-level `satisfaction_*`
aggregates. **No judge needed.** Per-task failures (e.g. agent tool-loops) are caught and the
batch continues.

- **Personas are keyed by the YAML filename stem (reddit `user_id`), NOT `persona_profile.id`.**
  The id inside the YAML collides across users (k100 → only ~35 unique ids), so keying by it
  silently overwrites runs. Output dirs are `<user_id>/`, matching tau2 `run_all_personas.py`.
- **`--skip-existing`** skips `(persona, task)` pairs whose JSON already exists → safe resume.
- **`--sim-model gemma`** is the user simulator; `--agent-model` is the agent under test.
- **More domains (E6 transfer):** repeat with `DOMAIN=customer_support` and
  `DOMAIN=shopping_assistant` (tasks 1–20 exist in each).
- **More persona arms:** point `--persona-dir` at `..._reddit_no_psych`, `..._nemotron`, etc.

### E.1 Reddit-style aggregate output (one file per persona, all 20 tasks)

To get the **same layout as the reddit/tau2 Eval folders** —
`<eval-dir>/<model>/<user_id>_<model>_<domain>.json`, one JSON holding all 20 tasks — use
`run_all_personas`. Same CLI shape as the tau2 runner; materializes personas straight from
the pipeline JSONL (keyed by `user_id`):

```bash
uv run python -m state_bench.scripts.run_all_personas \
  --models gpt-oss \
  --domain travel --num-tasks 20 \
  --sim-model gemma \
  --personas-jsonl ../reddit/personas_axis_a_reddit_schwartz_unopt_k100.jsonl \
  --eval-dir ../reddit/Eval/statebench_schwartz_unopt \
  --skip-existing            # add --dry-run first to print planned paths, write nothing
```

The envelope matches reddit (`{timestamp, info, tasks, simulations}`); each `simulations`
entry carries STATE-Bench's real metrics (`terminal_state`, `state_requirements_met`,
`satisfaction_*`), NOT tau2 `reward_info`. Note `--skip-existing` here skips a whole
`(model, persona)` file if it exists, so it won't top up a persona that's only partially done —
for incremental filling prefer `run_unofficial` (per-task skip) + `merge_to_eval` below.

### E.2 Merge per-task runs → reddit-style aggregates

If you collected per-task files with `run_unofficial` (section E), fold them into the
aggregate layout afterwards (idempotent — re-run any time to refresh):

```bash
uv run python -m state_bench.scripts.merge_to_eval \
  --results-dir outputs/travel_persona_k100_bystem \
  --eval-dir ../reddit/Eval/statebench_schwartz_unopt \
  --domain travel
```

Model/sim labels are read from the per-task JSONs, so no need to re-specify them.

### E.3 No-persona baseline

`--no-persona` runs the **stock, task-driven STATE-Bench simulator with no persona layer
injected** — the fixed-prompt baseline (code_review.md K.1) that isolates the effect of
persona injection. One run per model on the same 20 tasks, written as `baseline_<model>_<domain>.json`:

```bash
uv run python -m state_bench.scripts.run_all_personas \
  --models gpt-oss \
  --domain travel --num-tasks 20 \
  --sim-model gemma \
  --eval-dir ../reddit/Eval/statebench_baseline_gpt_oss_travel \
  --no-persona --skip-existing
```

Do NOT pass `--personas-jsonl` / `--persona-dir` with `--no-persona` (it errors). Same
envelope as the persona runs, so it's directly comparable.

**Caveat:** `terminal_state` relies on the persona convention of emitting `[TASK_DONE]` /
`[TERMINAL:…]` tags. The stock simulator emits none, so baseline runs mostly show
`terminal_state=incomplete` and run to max turns. For baseline-vs-persona comparison the
meaningful axis is the deterministic `state_requirements_met` (+ `satisfaction_*` if enabled),
not `terminal_state`.

## F. Monitor / resume

```bash
tail -f outputs/run_travel_k100_bystem.log
grep -c '\[ok\]'   outputs/run_travel_k100_bystem.log      # completed
grep -c '\[skip\]' outputs/run_travel_k100_bystem.log      # already-done (with --skip-existing)
grep    '\[ERR\]'  outputs/run_travel_k100_bystem.log      # failed (agent loops etc.)
```

Re-launch with `--skip-existing` to resume: existing `(persona, task)` transcripts are
skipped, missing ones are filled. Without it, a re-launch overwrites and redoes everything.

## G. Scoring (mostly inline already)

`satisfaction_*` and `state_requirements_met` are written during the run. Later:

- **Swap in the first author's calibrated critic prompt:** edit
  `state_bench/scripts/satisfaction_critic.py` (`SATISFACTION_SYSTEM` / `SATISFACTION_USER`),
  then re-score in place:
  ```bash
  uv run python -m state_bench.scripts.score_satisfaction \
    --results-dir outputs/travel_persona_k100_bystem --satisfaction-model qwen3-small --num-workers 4
  ```
- **Optional** native task/UX judge (NOT required; slow). Use a judge model ≠ agent model:
  ```bash
  uv run python -m state_bench.scripts.score_unofficial \
    --domain travel --results-dir outputs/travel_persona_k100_bystem --judge-model kimi --num-workers 4 --task $TASKS
  ```

## H. Output layout

Per-task (`run_unofficial`), keyed by reddit `user_id`:

```
STATE-Bench/outputs/<domain>_persona_k100_bystem/<user_id>/<task_id>.json
```

Reddit-style aggregate (`run_all_personas` or `merge_to_eval`), one file per persona:

```
<eval-dir>/<model>/<user_id>_<model>_<domain>.json   # {timestamp, info, tasks, simulations}
```

Two orthogonal axes per (task) record: `terminal_state` (persona satisfaction, NOT a pass) +
`state_requirements_met` (0/1 objective) + `satisfaction_cumulative/_worst_case/_mean/_final`
+ `satisfaction_per_turn[]`. See `STATE-Bench/instruction.md` for the full design.



PARNTER PROMPT: _SYSTEM_PROMPT = “”"\
You are evaluating how a user’s emotion changes after a single interaction step.
The user’s INTERNAL MONOLOGUE reflects their true thoughts and feelings.
---
# INPUT
You are given:
- the user’s goal
- the assistant’s previous response
- the user’s INTERNAL MONOLOGUE after that response
---
# TASK
Determine how the user’s emotion changed as a result of the assistant’s response.
---
# OUTPUT
Return:
emotion_delta ∈ [-10, +10]
---
# GUIDELINES
Base your judgment primarily on the INTERNAL MONOLOGUE.
Interpret the monologue as the user’s reaction to the assistant’s response.
Use the following scale consistently:
- No emotional change → 0
- Mild reaction (slight concern, mild satisfaction) → ±1 to ±3
- Moderate reaction (noticeable concern, annoyance, appreciation) → ±4 to ±6
- Strong emotion (frustration, relief, disappointment) → ±7 to ±9
- Very strong emotion (e.g., “unacceptable”, escalation, demanding supervisor) → ±9 to ±10
CRITICAL:
- Strong emotional language MUST produce large magnitude deltas (≥ |8|)
- Do NOT underestimate frustration or escalation signals
- If the monologue expresses increasing frustration, assign a negative delta
- If it expresses relief or satisfaction, assign a positive delta
- If the monologue contains only planning or neutral reasoning, assign 0 or a very small delta
---
# IMPORTANT
- Do NOT consider any future conversation
- Do NOT try to maintain or infer an absolute score
- Only estimate the CHANGE in emotion at this step
---
# RESPONSE FORMAT
{
  “emotion_delta”: <number>,
  “feeling”: “<brief explanation grounded in the internal monologue>”
}“”"


<!-- 
uv run python -m state_bench.scripts.run_unofficial \
  --domain $DOMAIN \
  --task $TASKS \
  --persona-dir ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt_k100 \
  --sim-model gemma \
  --agent-model gpt-oss \
  --no-satisfaction \
  --output-dir outputs/${DOMAIN}_persona_k100  -->