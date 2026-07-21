"""OpenAI-compatible client for Nautilus (ellm.nrp-nautilus.io)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

from state_bench.client import BaseLLMClient


def _extract_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating ```json fences and prose."""
    text = content.strip()
    if text.startswith("```"):
        # strip a leading fence line (``` or ```json) and any trailing fence
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON object from model output: {content[:200]!r}")


@dataclass(slots=True)
class NautilusToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class NautilusUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(slots=True)
class NautilusResponse:
    text: str = ""
    tool_calls: list[NautilusToolCall] = field(default_factory=list)
    usage: NautilusUsage | None = None


class NautilusClient(BaseLLMClient):
    """Chat-completions client for Nautilus OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://ellm.nrp-nautilus.io/v1",
        verify_ssl: bool = False,
    ) -> None:
        http_client = httpx.Client(verify=verify_ssl)
        self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self._model = model

    @classmethod
    def from_env(cls, *, model: str | None = None, role: str = "default") -> NautilusClient:
        """Build a client from NAUT_* env vars.

        role:
          - "agent": prefer NAUT_AGENT_API_BASE (e.g. local vLLM), else NAUT_API_BASE
          - "default" / "sim" / anything else: NAUT_API_BASE (sim, judge, critic)
        """
        verify_raw = os.environ.get("NAUT_VERIFY_SSL", "false").strip().lower()
        verify_ssl = verify_raw in {"1", "true", "yes"}
        api_key = os.environ.get("NAUT_API_KEY")
        if not api_key:
            raise ValueError("NAUT_API_KEY is required for NautilusClient")
        default_base = os.environ.get("NAUT_API_BASE", "https://ellm.nrp-nautilus.io/v1")
        if role == "agent":
            base_url = os.environ.get("NAUT_AGENT_API_BASE") or default_base
        else:
            base_url = default_base
        return cls(
            api_key=api_key,
            model=model or os.environ.get("NAUT_MODEL", "kimi"),
            base_url=base_url,
            verify_ssl=verify_ssl,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> NautilusResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=conversation,
            tools=tools or None,
        )
        message = response.choices[0].message

        tool_calls: list[NautilusToolCall] = []
        for call in message.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(NautilusToolCall(name=call.function.name, arguments=arguments))

        usage = None
        if response.usage is not None:
            usage = NautilusUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return NautilusResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
        )

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """Plain chat completion (no tools) for the user-simulator role.

        Matches the LLMClient.complete_chat interface the UserSimulator calls, but uses
        chat completions instead of the Responses API (Nautilus is OpenAI-compatible chat).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        reasoning_effort: str | None = None,  # noqa: ARG002 - accepted for judge interface parity
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """JSON completion for the metric judges (unofficial Nautilus judge path).

        Matches the complete_json interface the TaskRequirementsJudge / UXQualityJudge call.
        Nautilus is OpenAI-compatible chat; reasoning_effort has no chat-completions analog
        and is ignored. Requests a JSON object and robustly parses it from the content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        base_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        try:
            response = self._client.chat.completions.create(
                response_format={"type": "json_object"}, **base_kwargs
            )
        except Exception:
            # Not all Nautilus-served models support response_format; fall back to plain.
            response = self._client.chat.completions.create(**base_kwargs)
        content = response.choices[0].message.content or ""
        return _extract_json_object(content)
