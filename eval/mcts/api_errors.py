"""Detect external/provider API failures that should not be persisted as rollouts."""

from __future__ import annotations

# OpenAI SDK / gateway / transport failures — retryable infra, not task outcomes.
_API_ERROR_PREFIXES = (
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "APIError",
    "RateLimitError",
    "InternalServerError",
    "NotFoundError",  # e.g. Nautilus gateway: model route missing
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "ConflictError",
    "UnprocessableEntityError",
)


def is_external_api_error(err: str | None) -> bool:
    """True when ``err`` is an external API/provider failure (not agent logic)."""
    if not err or not isinstance(err, str):
        return False
    text = err.strip()
    if any(text.startswith(p) for p in _API_ERROR_PREFIXES):
        return True
    # httpx / transport phrasing sometimes surfaces without the OpenAI wrapper name
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "request timed out",
            "connection error",
            "no matching route found",
            "not configured in the gateway",
            "connecterror",
            "readtimeout",
            "connecttimeout",
        )
    )
