"""Resolve NAUT_* endpoints the same way STATE-Bench NautilusClient does.

Convention (matches STATE-Bench/clients/nautilus_client.py):
  - agent  → NAUT_AGENT_API_BASE or NAUT_API_BASE
  - sim    → NAUT_API_BASE

Typical split for this project (same as STATE-Bench CLI prefix):
  NAUT_API_BASE=http://localhost:8000/v1              # local gemma user-sim
  NAUT_AGENT_API_BASE=https://ellm.nrp-nautilus.io/v1 # remote gpt-oss agent
  NAUT_API_KEY=...

Load priority (highest wins):
  1. Shell / process env already set when Python starts
  2. tau2-bench/.env  (BASE split — overrides STATE-Bench's remote-only default)
  3. STATE-Bench/.env / repo .env  (key + defaults)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_NAUT_BASE = "https://ellm.nrp-nautilus.io/v1"

# Vars that define the agent/sim endpoint split.
_BASE_VARS = (
    "NAUT_API_BASE",
    "NAUT_AGENT_API_BASE",
    "NAUT_VERIFY_SSL",
    "OPENAI_API_BASE",
)


def load_naut_dotenv(
    *roots: Path,
    tau2_env: Path | None = None,
) -> list[Path]:
    """Load NAUT_* from .env files.

    ``tau2_env`` (typically ``tau2-bench/.env``) is applied with override for
    BASE vars so local-sim / remote-agent wins over STATE-Bench/.env's remote
    ``NAUT_API_BASE``. Any of those vars already set in the process environment
    before this call (shell prefix) are restored afterward so they still win.
    """
    loaded: list[Path] = []
    try:
        from dotenv import load_dotenv
    except ImportError:
        return loaded

    # Snapshot shell / pre-existing process env for split vars.
    preexisting = {k: os.environ[k] for k in _BASE_VARS if k in os.environ}

    for root in roots:
        if root is None:
            continue
        path = Path(root) / ".env"
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)

    if tau2_env is not None and Path(tau2_env).is_file():
        # Override BASE split after STATE-Bench defaults.
        load_dotenv(Path(tau2_env), override=True)
        loaded.append(Path(tau2_env))

    # Shell prefix always wins.
    for k, v in preexisting.items():
        os.environ[k] = v

    return loaded


def naut_verify_ssl() -> bool:
    raw = os.environ.get("NAUT_VERIFY_SSL", "false").strip().lower()
    return raw in {"1", "true", "yes"}


def naut_api_key() -> str | None:
    return os.environ.get("NAUT_API_KEY") or os.environ.get("OPENAI_API_KEY")


def naut_base_for_role(role: str) -> str:
    """role: 'agent' | 'sim' (or anything else → sim/default)."""
    default_base = os.environ.get("NAUT_API_BASE") or os.environ.get(
        "OPENAI_API_BASE", DEFAULT_NAUT_BASE
    )
    if role == "agent":
        return os.environ.get("NAUT_AGENT_API_BASE") or default_base
    return default_base


def litellm_openai_args(role: str, *, temperature: float = 0.0) -> dict[str, Any]:
    """Build kwargs for litellm openai/* models (tau2 --*-llm-args).

    Does **not** embed ``api_key`` (avoids leaking into tau2 simulation JSON /
    resume diffs). Set ``OPENAI_API_KEY`` / ``NAUT_API_KEY`` in the subprocess
    env instead.
    """
    args: dict[str, Any] = {
        "temperature": temperature,
        "api_base": naut_base_for_role(role),
    }
    if not naut_verify_ssl():
        args["ssl_verify"] = False
    return args


def apply_naut_key_to_environ(env: dict[str, str]) -> dict[str, str]:
    """Ensure subprocess sees an OpenAI-compat API key for litellm."""
    out = dict(env)
    key = naut_api_key()
    if key:
        out.setdefault("OPENAI_API_KEY", key)
        out.setdefault("NAUT_API_KEY", key)
    else:
        out.setdefault("OPENAI_API_KEY", "EMPTY")
    return out
