from unittest.mock import patch

import pytest

from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage


@pytest.fixture
def agent(get_environment) -> LLMAgent:
    return LLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
    )


@pytest.fixture
def solo_agent(get_environment, base_task) -> LLMSoloAgent:
    return LLMSoloAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        task=base_task,
    )


@pytest.fixture
def first_user_message():
    return UserMessage(content="Hello can you help me create a task?", role="user")


def test_agent(agent: LLMAgent, first_user_message: UserMessage):
    agent_state = agent.get_init_state()
    assert agent_state is not None
    agent_msg, agent_state = agent.generate_next_message(
        first_user_message, agent_state
    )
    # Check the response is an assistant message
    assert isinstance(agent_msg, AssistantMessage)
    # Check the state is updated
    assert agent_state is not None
    assert len(agent_state.messages) == 2
    # Check the messages are of the correct type
    assert isinstance(agent_state.messages[0], UserMessage)
    assert isinstance(agent_state.messages[1], AssistantMessage)
    assert agent_state.messages[0].content == first_user_message.content
    assert agent_state.messages[1].content == agent_msg.content


def test_agent_set_state(agent: LLMAgent, first_user_message: UserMessage):
    _ = agent.get_init_state(
        message_history=[
            UserMessage(content="Hello, can you help me find a flight?", role="user"),
            AssistantMessage(
                content="Hello, I can help you find a flight.", role="assistant"
            ),
        ]
    )


def test_agent_text_critic(get_environment, first_user_message):
    """Text critic rewrites a draft response and stores it in raw_data."""
    agent = LLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        critic_model="gpt-4o-mini",
    )
    draft_msg = AssistantMessage(role="assistant", content="Draft reply from agent.")
    checklist_msg = AssistantMessage(role="assistant", content="1. Always confirm before actions.")
    critic_msg = AssistantMessage(role="assistant", content="Critic-rewritten reply.")

    with patch("tau2.agent.llm_agent.generate", side_effect=[draft_msg, checklist_msg, critic_msg]):
        state = agent.get_init_state()
        result, _ = agent.generate_next_message(first_user_message, state)

    assert result.content == "Critic-rewritten reply."
    assert result.raw_data is not None
    assert result.raw_data["critic_draft"] == "Draft reply from agent."
    assert result.raw_data["critic_changed"] is True


def test_agent_tool_call_critic(get_environment, first_user_message):
    """Tool call critic rewrites argument values and stores originals in raw_data."""
    agent = LLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        critic_model="gpt-4o-mini",
        critic_for_tool_calls=True,
    )
    draft_tc = ToolCall(id="tc1", name="create_task", arguments={"title": "old"}, requestor="assistant")
    draft_msg = AssistantMessage(role="assistant", tool_calls=[draft_tc])
    checklist_msg = AssistantMessage(role="assistant", content="1. Always confirm before actions.")
    critic_msg = AssistantMessage(
        role="assistant",
        content='{"name": "create_task", "arguments": {"title": "corrected"}}',
    )

    with patch("tau2.agent.llm_agent.generate", side_effect=[draft_msg, checklist_msg, critic_msg]):
        state = agent.get_init_state()
        result, _ = agent.generate_next_message(first_user_message, state)

    assert result.tool_calls is not None
    assert result.tool_calls[0].arguments == {"title": "corrected"}
    assert result.raw_data is not None
    assert result.raw_data["critic_draft_tool_calls"][0]["arguments"] == {"title": "old"}


def test_agent_tool_call_critic_cancels(get_environment, first_user_message):
    """Tool call critic cancels a tool call by returning prose, switching to text message."""
    agent = LLMAgent(
        llm="gpt-4o-mini",
        tools=get_environment().get_tools(),
        domain_policy=get_environment().get_policy(),
        critic_model="gpt-4o-mini",
        critic_for_tool_calls=True,
    )
    draft_tc = ToolCall(
        id="tc1", name="transfer_to_human_agents",
        arguments={"summary": "user wants X"}, requestor="assistant"
    )
    draft_msg = AssistantMessage(role="assistant", tool_calls=[draft_tc])
    checklist_msg = AssistantMessage(role="assistant", content="1. Always confirm before actions.")
    # Critic returns prose (no braces) → cancellation signal
    cancel_msg = AssistantMessage(
        role="assistant",
        content="I can help you with that directly. Let me look into it for you.",
    )

    with patch("tau2.agent.llm_agent.generate", side_effect=[draft_msg, checklist_msg, cancel_msg]):
        state = agent.get_init_state()
        result, _ = agent.generate_next_message(first_user_message, state)

    assert result.tool_calls is None
    assert result.content == "I can help you with that directly. Let me look into it for you."
    assert result.raw_data["critic_cancelled_tool_call"] is True
    assert result.raw_data["critic_draft_tool_calls"][0]["name"] == "transfer_to_human_agents"


def test_solo_agent(solo_agent: LLMSoloAgent):
    agent_state = solo_agent.get_init_state()
    assert agent_state is not None
    agent_msg, agent_state = solo_agent.generate_next_message(None, agent_state)
    assert isinstance(agent_msg, AssistantMessage)
    assert agent_state is not None
    assert len(agent_state.messages) == 1
