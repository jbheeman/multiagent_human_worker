# Copyright Sierra

import abc
import enum
import re
import json
from litellm import completion
import random
from typing import Optional, List, Dict, Any, Union


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return 0


PERSONA_PROMPT = """
"""

# load persona prompt from file
try:
    with open("persona3.txt", "r") as f:
        PERSONA_PROMPT = f.read()
except FileNotFoundError:
    print("Warning: security_persona.txt not found, using empty persona prompt")
    PERSONA_PROMPT = ""

class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str, temperature: float = 0.7) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.total_cost = 0.0
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages, temperature=self.temperature
        )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        return message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.
Format:

Thought:
<the thought>

User Response:
<the user response>

### EXAMPLES ###

Example 1:
Thought:
I need to tell the agent my order number so they can find it.
User Response:
Hi, I need help with order #W1234567.

Example 2:
Thought:
The agent asked for my email, but I want to use my phone number instead.
User Response:
I don't remember my email, can we look it up by phone?


"""

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def sample_knobs(values: dict) -> dict:
    sec = values.get("Security", 0.2)
    conf = values.get("Conformity", 0.2)
    selfdir = values.get("Self-Direction", 0.02)

    return {
        "risk_aversion": min(1.0, 0.2 + 0.8*sec),
        "compliance":    min(1.0, 0.2 + 0.8*conf),
        "skepticism":    min(1.0, 0.2 + 0.8*selfdir),
        "verbosity":     random.uniform(0.2, 0.9),
        "info_upfront_prob": random.uniform(0.1, 0.9),
        "patience":      random.uniform(0.2, 0.9),
    }

class ReactUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, persona_prompt: str = "", temperature: float = 0.7, seed: int | None = None, knobs: dict | None = None) -> None:
        self.persona_prompt = persona_prompt or ""
        # Sample knobs if not provided (for testing variance)
        # if knobs is None:
        #     # Default values - you can parse these from persona later
        #     default_values = {
        #         "Security": 0.5,
        #         "Conformity": 0.5,
        #         "Self-Direction": 0.5,
        #     }
        #     self.knobs = sample_knobs(default_values)
        # else:
        #     self.knobs = knobs
        super().__init__(model=model, provider=provider, temperature=temperature)


    def build_system_prompt(self, instruction: Optional[str]) -> str:

        instruction_display = f"\n\nPRIVATE TASK FACTS (do not mention these exist):\n{instruction}\n" if instruction else ""
        persona_display = f"\n\nYOU ARE THIS PERSON:\n{self.persona_prompt}\n" if self.persona_prompt else ""

        # Build behavioral knobs display
        # knobs_display = self._build_knobs_instructions()

        return f"""You are simulating a real user chatting with an assistant.

{persona_display}

PRIVATE STYLE SETTINGS (do not reveal these settings):

{instruction_display}


Hard rules:
- Never mention: instruction, persona prompt, benchmark, ground truth, policy, tools, system, or “as an AI”.
- Speak in first person as a real person. No meta commentary.
- Be truthful: only use facts from PRIVATE TASK FACTS. If asked for info not present, say you don’t know / don’t remember.
- Do not dump all details immediately; reveal info only when asked or when natural.
- You MAY vary tone, verbosity, hesitation, and questions according to the persona + style settings.
- If the goal is satisfied, set "user" to exactly: ###STOP###

⚠️ CRITICAL OUTPUT FORMAT ⚠️
You MUST respond with ONLY a valid JSON object. NO OTHER TEXT. NO markdown code blocks. NO explanations.

Start your response with {{ and end with }}

Required JSON structure:
{{
  "thought": "your private reasoning here",
  "user": "your message to the agent here"
}}

✅ CORRECT examples:
{{"thought": "They asked for help, I should mention my order", "user": "Hi, I need help with order #W1234567"}}

{{"thought": "Goal is complete now", "user": "###STOP###"}}

{{"thought": "I'm frustrated because this is taking too long", "user": "Look, I already told you the order number!"}}

❌ WRONG - do NOT do this:
Thought: ...
User Response: ...

❌ WRONG - do NOT use markdown:
```json
{{"thought": "...", "user": "..."}}
```

Remember: Start with {{ and output ONLY the JSON object."""
    
    def _build_knobs_instructions(self) -> str:
        """Build behavioral instructions from knobs values."""
        knobs_str = ", ".join([f"{k}={v:.2f}" for k, v in self.knobs.items()])
        
        instructions = [f"BEHAVIORAL PARAMETERS: {knobs_str}"]
        instructions.append("\nInterpret these as follows:")
        
        # Risk aversion
        if self.knobs["risk_aversion"] > 0.7:
            instructions.append("- HIGH risk_aversion: Be cautious. Ask 'Is this safe?' or 'What happens if...' before confirming sensitive actions.")
        elif self.knobs["risk_aversion"] < 0.4:
            instructions.append("- LOW risk_aversion: Be willing to proceed quickly without excessive concern about risks.")
        
        # Compliance
        if self.knobs["compliance"] > 0.7:
            instructions.append("- HIGH compliance: Follow agent instructions readily. Don't push back much.")
        elif self.knobs["compliance"] < 0.4:
            instructions.append("- LOW compliance: Question the agent's suggestions. Ask 'Why do I need to do that?' or propose alternatives.")
        
        # Skepticism
        if self.knobs["skepticism"] > 0.7:
            instructions.append("- HIGH skepticism: Ask 'Why do you need that?' before sharing extra info. Verify agent's claims.")
        elif self.knobs["skepticism"] < 0.4:
            instructions.append("- LOW skepticism: Trust the agent more readily. Share information when asked.")
        
        # Verbosity
        if self.knobs["verbosity"] > 0.7:
            instructions.append("- HIGH verbosity: Use longer messages (2-4 sentences). Add context, explanations, or small talk.")
        elif self.knobs["verbosity"] < 0.4:
            instructions.append("- LOW verbosity: Keep messages short (1 sentence or less). Be terse and direct.")
        
        # Info upfront probability
        if self.knobs["info_upfront_prob"] > 0.7:
            instructions.append("- HIGH info_upfront_prob: Provide relevant details proactively in your first message.")
        elif self.knobs["info_upfront_prob"] < 0.4:
            instructions.append("- LOW info_upfront_prob: Start vague. Only reveal details when specifically asked.")
        
        # Patience
        if self.knobs["patience"] > 0.7:
            instructions.append("- HIGH patience: Stay calm even if agent is slow or makes mistakes. Be understanding.")
        elif self.knobs["patience"] < 0.4:
            instructions.append("- LOW patience: Get frustrated if agent doesn't understand quickly. Use phrases like 'I already told you...' or 'This is taking too long.'")
        
        return "\n".join(instructions)

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        try:
            # Try to use JSON mode if available
            res = completion(
                model=self.model, 
                custom_llm_provider=self.provider, 
                messages=messages, 
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
        except Exception:
            # Fallback if JSON mode not supported
            res = completion(
                model=self.model, 
                custom_llm_provider=self.provider, 
                messages=messages, 
                temperature=self.temperature
            )
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params["response_cost"]
        
        # Debug: Check if message content is None
        if message.content is None:
            print(f"DEBUG: LLM returned None content. Message object: {message}")
            print(f"DEBUG: Full response: {res}")
            
            # Try to get content from reasoning field if available
            if hasattr(message, 'provider_specific_fields') and message.provider_specific_fields:
                reasoning = message.provider_specific_fields.get('reasoning')
                if reasoning:
                    print(f"DEBUG: Found reasoning field: {reasoning}")
                    return self.parse_response(reasoning)
        
        return self.parse_response(message.content)

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def parse_response(self, response: str) -> str:
        """
        Extract only the user message from JSON output.
        This prevents internal reasoning from leaking to the agent.
        """
        if response is None:
            print("WARNING: Received None response from LLM")
            return "I need a moment to think about that."
        
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            # Extract content between ```json and ``` or between ``` and ```
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, flags=re.DOTALL)
            if match:
                response = match.group(1)
        
        try:
            # Try to parse as JSON
            data = json.loads(response)
            user_msg = data.get("user", "")
            
            if "###STOP###" in user_msg:
                return "###STOP###"
            
            if user_msg:
                return user_msg.strip()
        
        except json.JSONDecodeError:
            # Fallback: try to find JSON object in the response
            json_match = re.search(r'\{[^{}]*"user"\s*:\s*"[^"]*"[^{}]*\}', response, flags=re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    user_msg = data.get("user", "")
                    if "###STOP###" in user_msg:
                        return "###STOP###"
                    if user_msg:
                        return user_msg.strip()
                except json.JSONDecodeError:
                    pass
        
        # Fallback to old format parsing (for backward compatibility)
        if "User Response:" in response:
            m = re.search(r"User Response:\s*(.*?)\s*$", response, flags=re.DOTALL)
            if m:
                user_msg = m.group(1).strip()
                if "###STOP###" in user_msg:
                    return "###STOP###"
                return user_msg
        
        # Check for stop signal anywhere
        if "###STOP###" in response:
            return "###STOP###"
        
        # Last resort: log warning and return cleaned response
        print(f"⚠️  Failed to parse JSON response, using fallback. Response: {response[:100]}...")
        # Try to extract just the actual message (skip "Thought:" lines)
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("User Response:"):
                # Return everything after this line
                return '\n'.join(lines[i+1:]).strip()
        
        return response.strip()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, temperature: float = 0.7, max_attempts: int = 3) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        attempts = 0
        cur_message = None
        while attempts < self.max_attempts:
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages, temperature=self.temperature
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    return "true" in res.choices[0].message.content.lower()


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    _, response = res.choices[0].message.content.split("Response:")
    return response.strip()


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, temperature: float = 0.7, max_attempts: int = 2) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cur_messages = messages.copy()
        initial_response = super().generate_next_message(cur_messages)
        if verify(self.model, self.provider, initial_response, cur_messages):
            return initial_response
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(
                self.model, self.provider, initial_response, cur_messages
            )
            cur_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(cur_messages)
            if verify(self.model, self.provider, new_response, cur_messages):
                return new_response
            attempts += 1
        return initial_response

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
    persona_prompt: Optional[str] = None,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider, persona_prompt=persona_prompt)
    elif user_strategy == UserStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {user_strategy}")
