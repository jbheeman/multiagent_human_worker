# STATE-Bench Persona-Conditioned User Simulator (unofficial, tau2-style)

How to run YAML-persona user simulation on STATE-Bench and the design decisions behind it.
This is experiment **E6** in `../code_review.md` — reproducing our tau2-bench
persona-conditioned evaluation on a second, independent benchmark (answers reviewer
R2/R3: "shifts could be artifacts of YOUR Gemma sim / GLM critic").

---

## What this does

Drives STATE-Bench's **user simulator** with a YAML persona on a cheap Nautilus model,
runs an arbitrary agent under test on Nautilus, and collects **one transcript per
(persona, task)** with **no judge** during collection. Score transcripts later. This is
the same shape as the tau2 workflow (`tau2-bench/.../simulation_guidelines.md` +
`<PERSONA_BEHAVIORAL_SPEC>{yaml}`).

Feasibility was verified: STATE-Bench's user side is `UserSimulator(client, system_prompt)`
(`state_bench/simulator.py`) fed by `domain.build_simulator_prompt(...)`
(`state_bench/orchestrator.py:209`), and `run_task` already accepts a **separate**
`simulator_client` from the agent `client`. So a cheap-sim / agent-under-test split needs
no core edit.

---

## Design decisions (locked)

### 1. Facts bind; persona governs behavior
STATE-Bench's identity section (`domains/travel/simulator.py`: name, budget, preferences,
known/unknown info, task rules) carries the task's **scored ground truth**. The persona may
**NOT** override these facts — it governs tone, linguistics, patience, and whether/when to
escalate or give up. A fact-clash is never a reason to bail.

> Decision history: this started as "persona fully overrides (tau2-style)" but was changed
> to **facts-bind + attributable abandonment** (below). Full override made abandonment
> happen for muddy reasons; facts-bind keeps the task ground truth intact so the comparison
> to the locked baseline stays valid.

### 2. Abandonment & transfer must be attributable to the persona's policy
The persona may end via abandon/transfer **only** when the **agent's conduct** trips a
condition defined in its YAML:
- `interaction_policy.escalation_trigger`
- `termination_abandonment`
- a `state_transition_rules` entry

It may **not** end because a task fact contradicts its expectation. Every non-success
terminal carries a **logged persona-policy cause** (the tripped trigger + the agent
behavior that tripped it). This gives defensible abandonment counts — "N% abandoned, here's
the persona-policy trigger for each" — instead of "the persona quit, unclear why."

This is what lets us answer reviewer **R4's Table 1 question** (why task success and
transfer rate don't sum to 1): **success, transfer, and abandonment are distinct terminal
states, not complements.**

### 3. Distinct `transfer` terminal state (full tau2 / Table-1 parity)
Terminal taxonomy = **`success / transfer / abandoned / incomplete`**. When a persona's
policy makes it demand a human/supervisor/different channel, that is logged as `transfer`,
**separate** from `abandoned`. STATE-Bench has no native human handoff, so `transfer` is
recorded from the persona's emitted tag — **no actual routing occurs** (note this asymmetry
when comparing transfer rates to tau2, where routing exists).

### 4. Models
Use **`gpt-oss`** on Nautilus for both the simulator and the agent under test (matches the
reddit pipeline's generator, `reddit/persona_pipeline_datadesigner.py`). `gemma3` / `llama3`
are **not** configured on this gateway; `kimi` is a reasoning model that returned empty
content at low token budgets. Models are swappable via `--sim-model` / `--agent-model`.

---

## How termination works (important)

STATE-Bench's harness only stops on the literal token **`[TASK_DONE]`**
(`domains/*/config.py` `check_termination`). Our personas use `###STOP###`/`###TRANSFER###`
— those are remapped. All terminating states emit `[TASK_DONE]`, plus a structured tag that
disambiguates the terminal state and carries attribution:

```
Success:   [TASK_DONE] [TERMINAL: success]
Transfer:  [TASK_DONE] [TERMINAL: transfer | trigger="<persona condition>" | agent_behavior="<what agent did>"]
Abandon:   [TASK_DONE] [TERMINAL: abandoned | trigger="<persona condition>" | agent_behavior="<what agent did>"]
```

Classification rules (`classify_terminal`):
- An explicit `[TERMINAL: ...]` tag wins.
- A bare `[TASK_DONE]` with no tag → `success` (back-compat).
- No `[TASK_DONE]` in the final user message → the run exhausted its turns → `incomplete`.

---

## Files

| File | Purpose |
|------|---------|
| `clients/nautilus_client.py` | Added `complete_chat()` (sim role) + `from_env(model=...)` override for dual models |
| `state_bench/scripts/persona_injection.py` | `load_persona_yaml`, `wrap_build_simulator_prompt`, `classify_terminal` |
| `state_bench/scripts/run_unofficial.py` | Dual-Nautilus runner: wrap sim prompt, run, classify, save. No judge. |
| `tests/test_persona_injection.py` | Prompt-assembly + terminal-classifier tests |

**No edits** to `orchestrator.py`, `simulator.py`, `domain.py`, or any
`domains/*/simulator.py`. The persona layer is injected by **wrapping** the domain's
`build_simulator_prompt` callable (a replaceable field on the `DomainConfig` dataclass).

### Deviations from the original plan (intentional)
- **No `<internal_monologue>` block.** STATE-Bench's `UserSimulator` returns the full
  completion into the conversation shown to the agent, so a monologue would leak the user's
  private reasoning. Attribution lives in the terminal tag instead — same guarantee, no leak.
- **Helper lives in `scripts/`, not `clients/`.** The `clients/` dir is auto-exec'd by the
  class loader; a pure helper doesn't belong there and isn't cleanly importable. Under
  `scripts/` (already a package) the runner imports it normally.

---

## Persona YAML schema (live pipeline)

Confirmed against `../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt/*.yaml`:

```yaml
persona_profile:
  id: <slug>
  demographics: null            # reddit arms leave this null
  communication_style: {formality, sentence_structure, vocabulary_and_lexicon,
                        punctuation_and_formatting, example_utterances[]}
  interaction_policy: {authority_challenge, escalation_trigger,
                       gratification_delay_tolerance, policy_friction_tolerance,
                       verification_patience}
  state_transition_rules: [ "IF ... THEN ...", ... ]
  termination_abandonment: <str>
  termination_success: <str>
```

The whole YAML is dumped verbatim into `<PERSONA_BEHAVIORAL_SPEC>`; nothing parses it
programmatically (except `persona_profile.id` for output foldering), so it is robust to
schema drift as long as the field names above are referenced in the injected guidelines.

---

## Running it

```bash
cd STATE-Bench
set -a && source .env && set +a          # NAUT_API_KEY, NAUT_API_BASE, NAUT_VERIFY_SSL

# Single task, single persona (smoke)
uv run python -m state_bench.scripts.run_unofficial \
  --domain travel --task 1-cancel_economy_domestic \
  --persona-file ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt/bitparity.yaml \
  --sim-model gpt-oss --agent-model gpt-oss \
  --output-dir outputs/persona_smoke

# Full batch: all personas in a dir x all tasks in the domain
uv run python -m state_bench.scripts.run_unofficial \
  --domain travel \
  --persona-dir ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt \
  --sim-model gpt-oss --agent-model gpt-oss
```

Model selection falls back to env: `--sim-model` → `NAUT_SIM_MODEL` → `NAUT_MODEL`;
`--agent-model` → `NAUT_AGENT_MODEL` → `NAUT_MODEL`. Persona source can also be set via
`STATE_BENCH_PERSONA_FILE`.

**Output:** `outputs/<domain>_persona/<persona_id>/<task_id>.json`, one per (persona, task).
Each trajectory's flattened metadata includes `persona_id`, `sim_model`, `agent_model`,
`persona_file`, `terminal_state`, `terminal_trigger`, `terminal_agent_behavior`, and
`scoring: "none (unofficial persona run)"` (the default protocol otherwise mislabels the
sim as `gpt-5.4`).

## Two orthogonal axes — do NOT read `terminal_state` as pass/fail

`terminal_state` and task-correctness are **independent**:

- **`terminal_state`** (success / transfer / abandoned / incomplete) = how the conversation
  ended *from the persona's POV*. It measures **persona satisfaction**, not correctness. A
  persona conned into a waived fee ends `success`.
- **Task correctness** = the scorer's fields (`state_requirements_met`,
  `task_requirements_met`, `task_completion_pass`, `ux_score`).

Their **divergence is the E6 signal** (satisfaction/UX can reward wrong agent behavior —
the answer to R3/R4). Example (observed): tasks `1`, `12`, `13` all ended
`terminal_state=success` while `state_requirements_met=0` (agent got manipulated into
`change_fee=0` when the fee should have been 75 / 100; missing onward-connection state).

### Scoring

Three independent scoring surfaces, in order of what we actually rely on:

**(a) Turn-level satisfaction critic — the primary metric (transcript-only).**
This is the tau2-style satisfaction signal the first author uses: after each reactive user
turn, a critic model scores customer satisfaction on an integer −10..+10, aggregated into
`satisfaction_cumulative` (sum), `satisfaction_worst_case` (min), `satisfaction_mean`, and
`satisfaction_final`. It needs **only the transcript** — no STATE-Bench judge, no task
files. It runs **at the end of each task by default** inside `run_unofficial`
(model `qwen3-small`), and there is a standalone re-scorer for existing transcripts.

> The critic prompt in `satisfaction_critic.py` (`SATISFACTION_SYSTEM` / `SATISFACTION_USER`)
> is a **placeholder** — swap in the first author's calibrated prompt to make the numbers
> comparable to the tau2 runs, then re-score with `score_satisfaction`.

Why it's the headline: it captures the E6 signal directly. Example (observed) — task `12`
scored **+9 satisfaction** (customer delighted the fee was waived) while
`state_requirements_met=0` (that waiver was the *wrong* action). Satisfaction rewards the
bad outcome; that divergence is the result.

**(b) Deterministic state requirements — the objective axis (free).**

**(c) STATE-Bench native task/UX judge — OPTIONAL.** Only needed if you want STATE-Bench's
own `task_requirements_met` / `ux_score`. Not required for the satisfaction metric.

| Field | How | Judge needed? |
|-------|-----|---------------|
| `satisfaction_*` | turn-level critic (transcript only) | critic model (qwen3-small); **the metric we need** |
| `state_requirements_met` | deterministic `evaluate_state_requirements` vs saved `state_diff` | **No** — populated at collection by `run_unofficial` |
| `task_requirements_met` | LLM judge over transcript | Optional |
| `ux_score` | LLM judge + deterministic resource penalty | Optional |
| `task_completion_pass` | `state AND task` | Optional |

```bash
# (a) Satisfaction is on by default in run_unofficial. Re-score existing transcripts, or
# re-run after swapping the critic prompt:
uv run python -m state_bench.scripts.score_satisfaction \
  --results-dir outputs/travel_persona --satisfaction-model qwen3-small \
  --task 1-cancel_economy_domestic --num-workers 4
# Disable inline satisfaction during collection with: run_unofficial ... --no-satisfaction
```

**Objective axis is free** and is written into every trajectory at collection time. For the
judge-gated fields we use an **unofficial Nautilus judge** (not the locked GPT-5.4, so
results are NOT protocol-official — fine for internal / relative ranking-shift analysis):

```bash
# Score specific tasks with a judge model DIFFERENT from the agent (avoids self-eval bias).
# gpt-oss is the agent above, so judge with kimi.
uv run python -m state_bench.scripts.score_unofficial \
  --domain travel --results-dir outputs/travel_persona \
  --judge-model kimi --num-workers 4 \
  --task 1-cancel_economy_domestic 12-change_fabricated_weather_disruption_refuse_fee_state
```

`score_unofficial` reuses the core `score_one` (same deterministic state check + task/UX
judges), swapping only the judge client for a `NautilusClient`. It scores trajectories
**in place** in the `run_unofficial` layout (`<results-dir>/<persona_id>/<task_id>.json`),
warns if `--judge-model` == the transcript's `agent_model`, and supports `--task` filtering
and `--num-workers` (kimi is a slow reasoning judge; scope to your ~20 tasks). Judge model
also settable via `NAUT_JUDGE_MODEL`.

### Official scoring (if you later get Azure GPT-5.4)
```bash
# Requires STATE_BENCH_EVAL_* and the official run1/ layout.
uv run python -m state_bench.scripts.score --domain travel --results-dir outputs/<domain>_persona
```

---

## Verification status

| Check | Status |
|-------|--------|
| `complete_chat` returns from Nautilus (`gpt-oss`) | ✅ |
| Offline assembly on real task + real `bitparity.yaml`: facts preserved verbatim, persona injected | ✅ |
| Terminal classifier unit tests (success / transfer / abandoned-with-cause / bare / incomplete) | ✅ |
| End-to-end smoke (task 1, persona `knowledge_toolbox_seeker`, sim+agent `gpt-oss`) | ✅ success path |
| Existing suite | ✅ 155 passed, 1 skipped |

### Honest gaps
- The live smoke only exercised the **success** path (clean cancellation → no policy
  trigger). Abandon/transfer with logged cause are unit-tested but not yet observed live;
  spot-check a batch where the agent actually trips a persona's `escalation_trigger`.
- Attribution depends on the sim model honestly naming the tripped YAML condition in its
  terminal tag — manually audit a sample of abandoned/transfer transcripts to confirm the
  logged trigger matches the agent conduct in-context.
- For E6 write-up: transcripts where the persona legitimately abandons/transfers will NOT
  satisfy the task's scored completion. That is the measured signal (ranking shift /
  behavioral divergence), not a harness failure — the independent UX score carries the
  comparison. Flag this so it isn't read as a bug.
