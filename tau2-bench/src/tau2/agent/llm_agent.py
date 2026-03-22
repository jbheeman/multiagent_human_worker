import json
import re
from copy import deepcopy
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

CHECKLIST_GENERATION_PROMPT = """
You are preparing to review a customer service agent's responses for policy
compliance. Read the following policy and produce a numbered checklist of
specific rules that must be verified in every agent response.

Each item should be a concrete, verifiable rule — not a vague guideline.
Write each rule so that a reviewer can check yes/no whether a draft response
satisfies it.

Consider:
- What must happen BEFORE the agent takes any action? (e.g., confirmations,
  listing details for the user, verifying eligibility)
- What is the agent NOT allowed to do or offer?
- What information must be collected from the user before proceeding?
- When should the agent transfer to a specialist vs handle directly?
- Are there rules about how to handle transfers? (e.g., confirming the user
  has no other needs before transferring)
- Any domain-specific constraints (amounts, timeframes, eligibility criteria)

<policy>
{policy}
</policy>

Output ONLY the numbered checklist. No preamble or commentary.
""".strip()

TEXT_CRITIC_PROMPT = """
You are a compliance reviewer for a customer service agent. Your job is to catch
policy violations and factual errors that the agent missed, then output a corrected
version of the agent's response.

You will receive:
- A CHECKLIST of policy rules derived from the company policy
- The CONVERSATION history so far
- The agent's DRAFT response

Review the draft against EVERY item in the checklist. Also verify:

- FACTUAL CONSISTENCY: Cross-check any names, IDs, dates, amounts, or
  reference numbers in the draft against what was actually discussed in the
  conversation or returned by tools. Fix any mismatches.

- COMPLETENESS: If the user asked multiple questions, ensure all are addressed.

- PREMATURE CLOSING: The draft must NOT end the conversation, say goodbye,
  offer to transfer, or suggest the interaction is complete UNLESS the user's
  request has been fully resolved AND the agent has confirmed with the user
  that there is nothing else they need help with. If the draft wraps up
  prematurely, rewrite it to continue addressing the user's needs or ask if
  there is anything else the agent can assist with.

Output ONLY the corrected response text. No commentary, no explanation.
If the draft is already correct, output it unchanged.
""".strip()

TOOL_CALL_CRITIC_PROMPT = """
You are a tool-call reviewer for a customer service agent. Your job is to verify
that the agent's intended tool call has correct arguments before it executes.

You will receive:
- A CHECKLIST of policy rules derived from the company policy
- The CONVERSATION history (including prior tool calls and their results)
- The TOOL SCHEMAS (available functions and their parameter specifications)
- The agent's DRAFT tool call (function name + arguments as JSON)

Review the draft tool call against the checklist. Also verify:

- CORRECT FUNCTION: Is this the right tool for what the agent is trying to do?
- ARGUMENT VALUES: Cross-check every argument value against the conversation
  (IDs, dates, amounts, names must match exactly).
- REQUIRED ARGUMENTS: All required parameters must be present per the tool schema.
- PREMATURE CLOSING: If this tool call would end the conversation (e.g.,
  transferring to a human agent, closing a ticket), CANCEL it UNLESS:
  (a) The user's request has been fully resolved, AND
  (b) The agent has already asked if there is anything else the user needs,
      and the user confirmed there is not.
  If either condition is not met, cancel by outputting a plain-text message
  instead. The message should continue helping the user or ask if there is
  anything else the agent can assist with today.

Output EITHER:
  - A JSON object {{"name": "<function_name>", "arguments": {{<args>}}}} to approve
    or correct the tool call.
  - A plain-text message (containing no JSON braces) to send to the user,
    cancelling the tool call entirely. Use this when the tool call must not
    execute. Write the message the agent should send to the user instead.
Do not mix JSON and prose.
""".strip()


def _format_conversation_for_critic(messages):
    """Format message list as readable transcript for critic context."""
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        role = msg.role.upper()
        if hasattr(msg, "content") and msg.content:
            lines.append(f"{role}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                lines.append(
                    f"{role} [tool_call]: {tc.name}({json.dumps(tc.arguments)})"
                )
        if isinstance(msg, ToolMessage):
            lines.append(f"TOOL ({msg.id}): {msg.content}")
    return "\n".join(lines)


def _merge_usage(usage1, usage2):
    """Merge two usage dicts by summing numeric values."""
    if usage1 is None:
        return usage2
    if usage2 is None:
        return usage1
    merged = dict(usage1)
    for key, val in usage2.items():
        if isinstance(val, (int, float)) and key in merged:
            merged[key] = merged[key] + val
        else:
            merged[key] = val
    return merged


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        critic_model: Optional[str] = None,
        critic_for_tool_calls: bool = False,
    ):
        """
        Initialize the LLMAgent.

        Args:
            critic_model: LLM model for one-shot critic. None disables the critic.
            critic_for_tool_calls: Whether to also critique tool call arguments.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.critic_model = critic_model
        self.critic_for_tool_calls = critic_for_tool_calls
        self._critic_checklist: Optional[str] = None

    def _get_critic_checklist(self) -> str:
        """Generate and cache a policy-derived checklist for the critic.

        Called lazily on first critic invocation. The checklist is generated
        once per agent instance (i.e., once per task) since the policy is
        static within a domain.
        """
        if self._critic_checklist is not None:
            return self._critic_checklist
        logger.info("Generating critic checklist from domain policy...")
        messages = [
            SystemMessage(role="system", content=CHECKLIST_GENERATION_PROMPT.format(
                policy=self.domain_policy
            )),
            UserMessage(
                role="user",
                content="Produce the checklist now.",
            ),
        ]
        result = generate(
            model=self.critic_model, messages=messages, tools=None, max_tokens=4096
        )
        self._critic_checklist = result.content or ""
        logger.info(
            f"Critic checklist (cost=${result.cost or 0:.4f}):\n{self._critic_checklist}"
        )
        return self._critic_checklist

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def _critique_text(self, draft, messages):
        """Critique and rewrite a text response using a second LLM pass."""
        try:
            conversation = _format_conversation_for_critic(messages)
            checklist = self._get_critic_checklist()
            critic_messages = [
                SystemMessage(role="system", content=TEXT_CRITIC_PROMPT),
                UserMessage(
                    role="user",
                    content=(
                        f"<checklist>\n{checklist}\n</checklist>\n\n"
                        f"<conversation>\n{conversation}\n</conversation>\n\n"
                        f"<draft_response>\n{draft.content}\n</draft_response>"
                    ),
                ),
            ]
            rewrite = generate(
                model=self.critic_model, messages=critic_messages, tools=None
            )
            logger.debug(
                f"Text critic: draft='{draft.content[:80]}...' -> "
                f"rewrite='{rewrite.content[:80]}...'"
            )
            return AssistantMessage(
                role="assistant",
                content=rewrite.content,
                cost=(draft.cost or 0.0) + (rewrite.cost or 0.0),
                usage=_merge_usage(draft.usage, rewrite.usage),
                raw_data={
                    "critic_draft": draft.content,
                    "critic_cost": rewrite.cost,
                    "critic_changed": (
                        draft.content.strip() != rewrite.content.strip()
                    ),
                },
            )
        except Exception as e:
            logger.warning(f"Text critic failed, using original draft: {e}")
            return draft

    def _critique_tool_call(self, draft, messages):
        """Critique and rewrite tool call arguments using a second LLM pass.

        The critic may:
        - Return JSON  → rewrite the tool call arguments/name.
        - Return prose → cancel the tool call entirely and send that text to
                         the user instead (e.g. to stop a premature transfer).
        """
        if not draft.tool_calls:
            return draft
        try:
            conversation = _format_conversation_for_critic(messages)
            checklist = self._get_critic_checklist()
            tool_schemas = "\n".join(
                json.dumps(t.openai_schema, indent=2) for t in self.tools
            )
            rewritten_calls = []
            total_critic_cost = 0.0
            original_calls = []
            cancel_text = None  # set when critic wants to cancel via prose

            for tc in draft.tool_calls:
                tc_json = json.dumps(
                    {"name": tc.name, "arguments": tc.arguments}, indent=2
                )
                original_calls.append({"name": tc.name, "arguments": tc.arguments})
                critic_messages = [
                    SystemMessage(role="system", content=TOOL_CALL_CRITIC_PROMPT),
                    UserMessage(
                        role="user",
                        content=(
                            f"<checklist>\n{checklist}\n</checklist>\n\n"
                            f"<conversation>\n{conversation}\n</conversation>\n\n"
                            f"<tool_schemas>\n{tool_schemas}\n</tool_schemas>\n\n"
                            f"<draft_tool_call>\n{tc_json}\n</draft_tool_call>"
                        ),
                    ),
                ]
                rewrite = generate(
                    model=self.critic_model, messages=critic_messages, tools=None
                )
                total_critic_cost += rewrite.cost or 0.0
                content = rewrite.content or ""

                if "{" not in content:
                    # No JSON braces at all → intentional cancellation
                    cancel_text = content.strip()
                    logger.debug(
                        f"Tool call critic cancelled '{tc.name}': {cancel_text[:80]}"
                    )
                    break

                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    rewritten_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=parsed.get("name", tc.name),
                            arguments=parsed.get("arguments", tc.arguments),
                            requestor=tc.requestor,
                        )
                    )
                else:
                    # Contains { but invalid JSON → parse error → keep original
                    logger.warning(
                        f"Tool call critic returned malformed JSON, keeping original: "
                        f"{content[:100]}"
                    )
                    rewritten_calls.append(tc)

            if cancel_text is not None:
                return AssistantMessage(
                    role="assistant",
                    content=cancel_text,
                    tool_calls=None,
                    cost=(draft.cost or 0.0) + total_critic_cost,
                    usage=draft.usage,
                    raw_data={
                        "critic_cancelled_tool_call": True,
                        "critic_draft_tool_calls": original_calls,
                        "critic_cost": total_critic_cost,
                    },
                )

            logger.debug(
                f"Tool call critic: {len(draft.tool_calls)} call(s) reviewed"
            )
            return AssistantMessage(
                role="assistant",
                content=draft.content,
                tool_calls=rewritten_calls,
                cost=(draft.cost or 0.0) + total_critic_cost,
                usage=draft.usage,
                raw_data={
                    "critic_draft_tool_calls": original_calls,
                    "critic_cost": total_critic_cost,
                },
            )
        except Exception as e:
            logger.warning(f"Tool call critic failed, using original draft: {e}")
            return draft

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = deepcopy(state.system_messages + state.messages)
        # Strip internal monologue from UserMessages before passing to Agent LLM
        for msg in messages:
            if isinstance(msg, UserMessage) and msg.content and "<internal_monologue>" in msg.content:
                msg.content = re.sub(r"<internal_monologue>.*?</internal_monologue>\s*", "", msg.content, flags=re.DOTALL).strip()

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )

        # One-shot critic pass
        if self.critic_model is not None:
            if assistant_message.has_text_content() and not assistant_message.is_tool_call():
                assistant_message = self._critique_text(assistant_message, messages)
            elif assistant_message.is_tool_call() and self.critic_for_tool_calls:
                assistant_message = self._critique_tool_call(
                    assistant_message, messages
                )

        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(LocalAgent[LLMAgentState]):
    """
    An GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.provide_function_args = provide_function_args

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
            
        messages = deepcopy(state.system_messages + state.messages)
        # Strip internal monologue from UserMessages before passing to Agent LLM
        for msg in messages:
            if isinstance(msg, UserMessage) and msg.content and "<internal_monologue>" in msg.content:
                msg.content = re.sub(r"<internal_monologue>.*?</internal_monologue>\s*", "", msg.content, flags=re.DOTALL).strip()

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{stop_function_name}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


class LLMSoloAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = llm_args if llm_args is not None else {}
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""

        def done() -> str:
            """Call this function when you are done with the task."""
            return self.STOP_TOKEN

        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Task should contain a ticket and evaluation criteria.
        If the task contains an initial state, the message history should only contain tool calls and responses.
        """
        if task.initial_state is not None:
            message_history = task.initial_state.message_history or []
            for message in message_history:
                if isinstance(message, UserMessage):
                    return False
                if isinstance(message, AssistantMessage) and not message.is_tool_call():
                    return False
            return True
        if task.ticket is None:
            return False
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        agent_instruction = AGENT_SOLO_INSTRUCTION.format(
            stop_function_name=self.STOP_FUNCTION_NAME,
            stop_token=self.STOP_TOKEN,
        )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        """
        is_stop = False
        for tool_call in message.tool_calls:
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, UserMessage):
            raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            tool_choice="required",
            **self.llm_args,
        )
        if not assistant_message.is_tool_call():
            raise ValueError("LLMSoloAgent only supports tool calls.")
        message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
