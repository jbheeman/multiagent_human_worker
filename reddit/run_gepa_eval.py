"""tau2 glue for the GEPA behavioral signal.

Replaces the old external `tau_bench.run_gepa_eval` (gone with the tau_bench ->
tau2 migration). Runs ONE tau2 simulation in-process per call (cheap enough for the
GEPA loop) with the GEPA-generated persona injected as the user simulator's
behavioral spec, and returns a cleaned transcript for the behavioral judge.

Design choices (all env-overridable):
  * Domain: the lightweight `mock` domain by default — a fast, deterministic
    customer-support environment, so the tau signal measures behavioral
    consistency without depending on retail/airline task realism.
  * Task selection: deterministic hash of `user_id` over the mock task pool
    excluding `impossible_task_*` (agent transfers on turn one → 2-turn stubs,
    systematically compressed/noisy tau scores). Every candidate prompt for the
    same user is compared on the identical persona–task pair; diversity lives
    across users, not as noise within candidate comparisons. Held-out transfer
    is retail/telecom/airline — mock is the training signal, not a transfer claim.
  * In-process `tau2.run.run_task` (not the `tau2 run` CLI) so there is no
    subprocess / results-file overhead per metric call.
  * Models routed to the NRP endpoint via litellm (api_base/api_key in llm_args).
  * Reward eval set to ENV (state-only, no extra LLM calls) — we only need the
    transcript; the persona-behavior scoring happens in personaAdapter's judge.

Persona injection: `run_task` reads the module-global `tau2.run.yaml_content`
(loaded once at import from TAU2_PERSONA_FILE) at call time and interpolates it
verbatim into the user simulator's <PERSONA_BEHAVIORAL_SPEC>. We override that
global per call so each persona drives its own simulation. Because that global is
read mid-`run_task` (agent/user construction AND the post-sim persona critic), a
`ThreadPoolExecutor`-parallel caller would race on it and mis-pair personas with
transcripts, so the whole set-global + `run_task` section runs under a lock. Tau
sims therefore serialize; the caller's other signals still parallelize.
"""

from __future__ import annotations

import hashlib
import os
import threading

import litellm
import tau2.run as _tau2_run
from tau2.run import get_tasks, run_task
from tau2.evaluator.evaluator import EvaluationType

# tau2 drives its sims through litellm.completion(), which verifies SSL by default.
# The NRP/Nautilus endpoint has cert issues (personaAdapter's persona/teacher models
# use httpx verify=False for the same reason), so without this every tau sim's LLM
# call fails with an SSL error. Match the codebase's existing verify=False handling.
litellm.ssl_verify = False


NRP_ENDPOINT = os.getenv("NRP_ENDPOINT", "https://ellm.nrp-nautilus.io/v1")
NAUT_API_KEY = os.getenv("NAUT_API_KEY")

TAU_DOMAIN = os.getenv("TAU_DOMAIN", "mock")
TAU_AGENT_LLM = os.getenv("TAU_AGENT_LLM", "openai/kimi")
TAU_USER_LLM = os.getenv("TAU_USER_LLM", "openai/gemma")
TAU_MAX_STEPS = int(os.getenv("TAU_MAX_STEPS", "30"))
TAU_SEED = int(os.getenv("TAU_SEED", "300"))

# litellm's openai provider also reads these; set them as a fallback to the
# explicit api_base/api_key we pass through llm_args.
if NAUT_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", NAUT_API_KEY)
os.environ.setdefault("OPENAI_API_BASE", NRP_ENDPOINT)

_LLM_ARGS = {"temperature": 0.0, "api_base": NRP_ENDPOINT, "api_key": NAUT_API_KEY}

# Tasks are loaded once (pure JSON read, no LLM). Drop impossible_task_*: dry-run
# showed the fixed agent transfers on the first user turn, leaving ~2 user turns
# and nothing for the behavioral judge to verify (noise, not construct-validity).
_ALL_TASKS = get_tasks(TAU_DOMAIN)
_TASKS = [t for t in _ALL_TASKS if not str(t.id).startswith("impossible_task_")]
if not _TASKS:
    raise RuntimeError(f"No usable tasks in domain {TAU_DOMAIN!r} after excluding impossible_task_*")

# run_task reads the module-global persona mid-call, so concurrent sims must not
# clobber each other's persona. Serialize the set-global + sim critical section.
_TAU_LOCK = threading.Lock()


def select_task(user_id: str):
    """Deterministic user_id → task assignment over the usable mock pool."""
    if not user_id:
        raise ValueError("user_id is required for deterministic tau task assignment")
    idx = int(hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16) % len(_TASKS)
    return _TASKS[idx]


def run_evaluation(persona_yaml: str, user_id: str, *, task_id: str | None = None):
    """Run one tau2 simulation with `persona_yaml` as the user simulator's
    behavioral spec (the same PersonaProfile YAML the production sims consume).

    Task is `hash(user_id) % len(pool)` unless `task_id` is passed (dry-runs /
    forced probes). Seed stays fixed so candidate comparisons for the same
    persona–task pair are not RNG-noisy.
    Returns the tau2 SimulationRun.
    """
    if task_id is not None:
        matches = [t for t in _ALL_TASKS if t.id == task_id]
        if not matches:
            raise ValueError(f"Unknown task_id={task_id!r} in domain {TAU_DOMAIN}")
        task = matches[0]
    else:
        task = select_task(user_id)
    with _TAU_LOCK:
        # Inject the persona into the global the user simulator reads via run_task,
        # then run the whole sim before another thread can overwrite it.
        _tau2_run.yaml_content = persona_yaml
        return run_task(
            domain=TAU_DOMAIN,
            task=task,
            agent="llm_agent",
            user="user_simulator",
            llm_agent=TAU_AGENT_LLM,
            llm_args_agent=dict(_LLM_ARGS),
            llm_user=TAU_USER_LLM,
            llm_args_user=dict(_LLM_ARGS),
            max_steps=TAU_MAX_STEPS,
            evaluation_type=EvaluationType.ENV,  # state-only; no extra judge LLM calls
            seed=TAU_SEED,
        )


def clean_transcript_for_judge(result) -> str:
    """Flatten a SimulationRun into a role-labeled transcript string, dropping
    tool-call turns with no text content."""
    if result is None:
        return ""
    messages = getattr(result, "messages", None) or []
    lines = []
    for msg in messages:
        role = msg["role"] if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            lines.append(f"{str(role).upper()}: {content}")
    return "\n\n".join(lines)
