"""Shared backend helpers."""

from __future__ import annotations

from pathlib import Path


def to_fs_label(value: str) -> str:
    return value.replace("/", "__")


def artifact_dir(eval_dir: Path, *, model: str, domain: str, arm: str) -> Path:
    """``eval_dir/{model}/{domain}/{arm}/`` — domain-segregated artifact root."""
    return eval_dir / to_fs_label(model) / to_fs_label(domain) / arm


def artifact_filename(
    *,
    persona_key: str,
    model: str,
    domain: str,
    task_id: str,
) -> str:
    model_label = to_fs_label(model)
    return f"{persona_key}_{model_label}_{domain}_{task_id}.json"


def artifact_path(
    eval_dir: Path,
    *,
    model: str,
    domain: str,
    arm: str,
    persona_key: str,
    task_id: str,
) -> Path:
    return artifact_dir(eval_dir, model=model, domain=domain, arm=arm) / artifact_filename(
        persona_key=persona_key,
        model=model,
        domain=domain,
        task_id=task_id,
    )
