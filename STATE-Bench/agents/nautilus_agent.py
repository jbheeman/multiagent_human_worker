"""STATE-Bench agent that calls Nautilus via chat completions + tool calling."""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


def _to_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _to_chat_messages(system_prompt: str, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    call_id_counter = 0
    index = 0
    while index < len(conversation):
        msg = conversation[index]
        role = msg.get("role")
        if role == "user":
            messages.append({"role": "user", "content": msg.get("content", "")})
            index += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                index += 1
                continue

            openai_tool_calls: list[dict[str, Any]] = []
            pending_results: list[tuple[str, Any]] = []
            for tool_call in tool_calls:
                call_id = f"call_{call_id_counter}"
                call_id_counter += 1
                openai_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }
                )
                if "result" in tool_call:
                    pending_results.append((call_id, tool_call["result"]))

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": openai_tool_calls,
                }
            )
            for call_id, result in pending_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            if index + 1 < len(conversation) and conversation[index + 1].get("role") == "tool":
                index += 2
            else:
                index += 1
            continue

        index += 1

    return messages


class NautilusAgent(BaseAgent):
    def __init__(
        self,
        client,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        runtime_context=None,
        **kwargs,
    ) -> None:
        super().__init__(runtime_context=runtime_context)
        self.client = client

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        provider_tools = _to_chat_tools(tools)
        response = self.client.generate(
            system_prompt=system_prompt,
            conversation=_to_chat_messages(system_prompt, conversation),
            tools=provider_tools,
        )

        if response.usage is not None:
            self.add_token_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_input_tokens=response.usage.cached_input_tokens,
            )

        return AgentTurnResponse(
            text=response.text,
            tool_calls=[
                AgentToolCallRequest(name=call.name, arguments=call.arguments) for call in response.tool_calls
            ],
        )
