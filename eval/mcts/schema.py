"""Core rollout JSONL schema (Phase 1 — no satisfaction_* fields)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TERMINAL_STATES = frozenset(
    {
        "success",
        "transfer",
        "failure_no_transfer",
        "max_turns",
        "sim_error",
    }
)

CORE_REQUIRED_KEYS = (
    "run_id",
    "config_hash",
    "block",
    "domain",
    "task_id",
    "persona_id",
    "arm",
    "model",
    "sim_model",
    "sim_seed",
    "judge_model",
    "judge_prompt_hash",
    "terminal_state",
    "task_success",
    "transfer",
    "n_turns",
    "full_transcript",
    "persona_variant_hash",
    "bench",
)

# Identity tuple used for run_id / resume / enrichment joins.
IDENTITY_KEYS = (
    "bench",
    "block",
    "domain",
    "task_id",
    "persona_id",
    "arm",
    "model",
    "sim_seed",
)


def make_run_id(
    *,
    bench: str,
    block: int,
    domain: str,
    task_id: str,
    persona_id: str,
    arm: str,
    model: str,
    sim_seed: int,
) -> str:
    payload = "|".join(
        [
            str(bench),
            str(block),
            str(domain),
            str(task_id),
            str(persona_id),
            str(arm),
            str(model),
            str(sim_seed),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def make_config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def hash_persona_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate_core_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors (empty => OK)."""
    errors: list[str] = []
    for key in CORE_REQUIRED_KEYS:
        if key not in record:
            errors.append(f"missing key: {key}")
    if errors:
        return errors

    if record["terminal_state"] not in TERMINAL_STATES:
        errors.append(f"invalid terminal_state: {record['terminal_state']!r}")
    if not isinstance(record["task_success"], bool):
        errors.append("task_success must be bool")
    if not isinstance(record["transfer"], bool):
        errors.append("transfer must be bool")
    if not isinstance(record["n_turns"], int) or record["n_turns"] < 0:
        errors.append("n_turns must be non-negative int")
    if not isinstance(record["full_transcript"], list):
        errors.append("full_transcript must be a list")
    if record["bench"] not in ("tau2", "statebench"):
        errors.append(f"invalid bench: {record['bench']!r}")

    # Exclusive enum consistency helpers (not hard requirements beyond enum).
    if record["terminal_state"] == "success" and not record["task_success"]:
        errors.append("terminal_state=success requires task_success=True")
    if record["terminal_state"] == "transfer" and not record["transfer"]:
        errors.append("terminal_state=transfer requires transfer=True")

    expected = make_run_id(
        bench=record["bench"],
        block=int(record["block"]),
        domain=record["domain"],
        task_id=record["task_id"],
        persona_id=record["persona_id"],
        arm=record["arm"],
        model=record["model"],
        sim_seed=int(record["sim_seed"]),
    )
    if record["run_id"] != expected:
        errors.append(f"run_id mismatch: got {record['run_id']}, expected {expected}")

    return errors


def build_core_record(
    *,
    bench: str,
    config_hash: str,
    block: int,
    domain: str,
    task_id: str,
    persona_id: str,
    arm: str,
    model: str,
    sim_model: str,
    sim_seed: int,
    judge_model: str | None,
    judge_prompt_hash: str | None,
    terminal_state: str,
    task_success: bool,
    transfer: bool,
    n_turns: int,
    full_transcript: list[Any],
    persona_variant_hash: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": make_run_id(
            bench=bench,
            block=block,
            domain=domain,
            task_id=task_id,
            persona_id=persona_id,
            arm=arm,
            model=model,
            sim_seed=sim_seed,
        ),
        "config_hash": config_hash,
        "block": block,
        "domain": domain,
        "task_id": task_id,
        "persona_id": persona_id,
        "arm": arm,
        "model": model,
        "sim_model": sim_model,
        "sim_seed": sim_seed,
        "judge_model": judge_model,
        "judge_prompt_hash": judge_prompt_hash,
        "terminal_state": terminal_state,
        "task_success": task_success,
        "transfer": transfer,
        "n_turns": n_turns,
        "full_transcript": full_transcript,
        "persona_variant_hash": persona_variant_hash,
        "bench": bench,
    }
    if extra:
        # Do not allow overwriting identity/schema keys silently.
        for key, value in extra.items():
            if key in CORE_REQUIRED_KEYS:
                continue
            record[key] = value
    errors = validate_core_record(record)
    if errors:
        raise ValueError("Invalid core record: " + "; ".join(errors))
    return record
