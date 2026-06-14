# NOTE (AAAI rework): the scale loop in this file's __main__ (paragraph -> LLM YAML
# compile with invented keys + k=3 conformance retries) is SUPERSEDED by
# persona_pipeline_datadesigner.py, which uses a FIXED PersonaProfile schema (no
# invented keys), deterministic YAML, and copies the Schwartz vector verbatim.
# REDDIT_PROMPT below has been de-leaked to match personaAdapter; YAML_PROMPT and
# CONFORMANCE_PROMPT are retained for reference only.
from smolagents.models import OpenAIServerModel
import os
from typing import Any
import json
import re
import time
from functools import wraps
from dataclasses import dataclass
from cleanpersona import _clean_persona

# HTTP timeouts: NRP/ellm can be slow or stall; without a read timeout the client may hang
# for a very long time. Tune with env vars. (Your physical location does not matter —
# requests go from *this machine* (the VM) to the API; laptop "internet working" is unrelated.)
# Large models (e.g. Qwen 397B) + YAML compile often exceed 5–10 min; 300s default caused frequent timeouts.
_connect_s = float(os.getenv("LLM_CONNECT_TIMEOUT_SEC", "30"))
_read_s = float(os.getenv("LLM_READ_TIMEOUT_SEC", "1200"))  # 20 min default; raise if you still see timeouts
try:
    import httpx

    _llm_http_timeout: Any = httpx.Timeout(connect=_connect_s, read=_read_s, write=120.0, pool=60.0)
except Exception:
    _llm_http_timeout = max(_connect_s, _read_s)

# Qwen3.5 on NRP may return chain-of-thought in `reasoning` with `content` null.
_LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16384"))

persona_model = OpenAIServerModel(
    model_id="gpt-oss",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.getenv("NAUT_API_KEY"),
    max_tokens=_LLM_MAX_TOKENS,
    # OpenAI SDK retries + long default timeouts can look like an infinite hang; cap explicitly.
    client_kwargs={
        "timeout": _llm_http_timeout,
        "max_retries": int(os.getenv("OPENAI_CLIENT_MAX_RETRIES", "0")),
    },
)


def _assistant_text(msg: Any) -> str:
    """Extract assistant text from smolagents ChatMessage (content and/or reasoning fields)."""
    if msg is None:
        return ""
    c = getattr(msg, "content", None)
    if c:
        return str(c)
    raw = getattr(msg, "raw", None)
    if raw is not None:
        try:
            m = raw.choices[0].message
            for attr in ("reasoning", "reasoning_content"):
                if getattr(m, attr, None):
                    return str(getattr(m, attr))
            if hasattr(m, "model_dump"):
                d = m.model_dump()
                for k in ("reasoning", "reasoning_content", "content"):
                    if d.get(k):
                        return str(d[k])
        except Exception:
            pass
    return ""


def _maybe_truncate_persona(text: str, user_id: str) -> str:
    """Optionally cap persona size — very long inputs make compile slow and can hit context limits."""
    max_chars = os.getenv("MAX_PERSONA_CHARS")
    if not max_chars:
        return text
    limit = int(max_chars)
    if len(text) <= limit:
        return text
    print(
        f"  WARNING: persona for {user_id} is {len(text)} chars; truncating to {limit} (set MAX_PERSONA_CHARS to change).",
        flush=True,
    )
    return text[:limit] + "\n\n[... truncated by MAX_PERSONA_CHARS ...]\n"



def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator to retry a function with exponential backoff on timeout or connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, Exception) as e:
                    error_str = str(e).lower()
                    is_timeout = "timeout" in error_str or "timed out" in error_str
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the exception
                        raise
                    
                    if is_timeout or "connection" in error_str:
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Non-timeout error, don't retry
                        raise
            return None
        return wrapper
    return decorator



REDDIT_PROMPT = """
You are an expert Psychological Profiler.
Generate a persona definition that is self-explanatory. The persona description must be so coherent and psychologically vivid that an AI acting as this person will naturally deduce how to behave in any situation purely by reading the description.

Do not write specific rules (e.g., 'Do not give zip code'). Instead, write the psychological reasoning (e.g., 'He is deeply skeptical of digital surveillance and treats personal data as a currency to be hoarded').

=== INPUT DATA ===
1. SUBREDDITS:
{subreddits}

2. PSYCHOLOGICAL SHIFT:
{target_vector}

3. BEHAVIORAL SAMPLES:
{history_str}

=== OUTPUT FORMAT ===
You must output the persona in the following strict format:

### 1. CORE IDENTITY
(A first-person introduction: "I am a [Age] year old [Job]...")

### 2. PSYCHOLOGICAL DRIVERS
(A narrative explanation of *why* they act the way they do. Connect their background and lived
experience to their priorities — in plain language, never as named psychological values or numbers.)

### 3. INTERNAL MONOLOGUE STYLE
Describe how this person thinks and reasons under pressure. Then give exactly two example internal
thoughts, prefixed "Example 1: " and "Example 2: ", set in a support interaction. The examples must
reveal the person's priorities IMPLICITLY through concrete reactions — they must NOT name any
psychological value or cite any number.

=== YOUR RESPONSE ===
"""


YAML_PROMPT = """
You are an expert Persona Compiler for Multi-Agent Simulation Environments. Your objective is to ingest raw, narrative-heavy human personas and compile them into strict, machine-readable YAML behavioral specifications. 

These YAML specifications will be used to govern the behavior of a simulated user interacting with a target agent in an objective, state-tracking benchmark (e.g., Dec-POMDP environments like Tau-bench).

YOUR DIRECTIVES:
1. Strip all narrative fluff, backstory, and "internal monologue" rules.
2. CRITICAL - DYNAMIC KEYS: For `cognitive_profile` and `interaction_policy`, you MUST invent custom, highly specific keys tailored to the specific Schwartz values of the persona. DO NOT use generic keys like "technical_competence", "patience_level", or "adaptability" for everyone. Invent keys like "authority_defiance", "novelty_seeking", "bureaucracy_tolerance", or "empathy_capacity".
3. Translate abstract values into concrete, observable Interaction Policies.
4. Define strict State-Transition Rules (If/Then heuristics) that dictate exactly how the persona reacts to specific agent behaviors.
5. Output ONLY valid YAML. Do not include any conversational filler, preamble, or postscript.

YAML SCHEMA TO POPULATE:

persona_profile:
  id: [Generate a descriptive string, e.g., "28_High_Stimulation"]
  demographics: [Brief string summarizing age, role, region]
  communication_style:
    formality: [Low/Medium/High] # Provide specific, actionable writing instructions instead of abstract traits
    sentence_structure: "[Describe sentence length, pacing, and whether they use run-ons, fragments, or perfect grammar]"
    vocabulary_and_lexicon: "[Describe specific word choices, industry jargon, slang, or idioms they rely on]"
    punctuation_and_formatting: "[Describe their typing habits: do they ignore capitalization? Overuse ellipses? Use emojis or ALL CAPS?]"
    example_utterances:
        - "[Invent a highly specific quote demonstrating the above 3 traits perfectly]"
        - "[Invent a second quote showing how they ask for help]"


cognitive_profile:
  [INVENT_CUSTOM_TRAIT_1]: [Low/Medium/High/Specific Descriptor]
  [INVENT_CUSTOM_TRAIT_2]: [Low/Medium/High/Specific Descriptor]
  [INVENT_CUSTOM_TRAIT_3]: [Low/Medium/High/Specific Descriptor]

interaction_policy:
  [INVENT_CUSTOM_POLICY_1]: [Specific Descriptor]
  [INVENT_CUSTOM_POLICY_2]: [Specific Descriptor]
  escalation_trigger: [Specific condition that causes them to demand a human/supervisor]

state_transition_rules:
  - "IF the agent asks you to wait or delays, THEN [Specific behavioral reaction]"
  - "IF the agent denies a request based on strict policy, THEN [Specific behavioral reaction]"
  - "IF the agent makes a mistake, THEN [Specific behavioral reaction]"
  - "[Add 1-2 more IF/THEN rules specific to this persona's dominant Schwartz values]"

"""


# termination_conditions:
#   success: "The final database state matches the initial goal."
#   abandonment: "[Specific condition where this persona gives up, e.g., 'Agent repeats the same question 3 times' or 'Task takes more than 5 turns']"

CONFORMANCE_PROMPT = """
You are an Automated Conformance Evaluator for Multi-Agent Behavioral Specs.
Your task is to verify that a compiled YAML behavioral profile strictly aligns with its source Schwartz Values vector.

INPUT A (Source Values JSON):
{schwartz_json}

INPUT B (Compiled YAML):
{yaml_file}

EVALUATION CRITERIA:
Analyze the YAML against the source values and output a JSON evaluation with binary pass/fail scores.

1. Dominant Values Check: Identify the top 2 highest Schwartz values in INPUT A. Does the YAML explicitly codify behavioral policies, triggers, or tone that manifest these specific dominant values? 
2. Inferior Values Check: Identify the 1 lowest Schwartz value in INPUT A. Does the YAML explicitly show a lack of concern, or resistance, related to this lowest value?
3. State-Transition Verifiability: Do the state_transition_rules contain strict "IF/THEN" behavioral heuristics rather than vague narrative guidelines?

OUTPUT FORMAT (output ONLY valid JSON, no preamble):
{{
  "Dominant_Values_Check": {{"pass": true/false, "identified_values": "[List the 2 values]", "reason": "..."}},
  "Inferior_Value_Check": {{"pass": true/false, "identified_value": "[List the 1 value]", "reason": "..."}},
  "State_Transition_Check": {{"pass": true/false, "reason": "..."}},
  "OVERALL_STATUS": "APPROVE or REJECT"
}}
"""
# 4. Termination Bounds Check: Are there strict, numerical bounds on when the persona abandons the task (e.g., specific turn limits, repetition limits)?
#   "Termination_Check": {{"pass": true/false, "reason": "..."}},

@dataclass
class PersonaDataInst_Synthetic:
    id: str
    persona: dict
    schwartz_values: dict  # for conformance check (e.g. CONFORMITY, SECURITY, ...)

@dataclass
class PersonaDataInst_Scaled:
    user_id: str
    persona: dict

@dataclass
class PersonaDataInst_GEPA:
    user_id: str             # "lumenation"
    subreddits: list[str]           # "r/KotakuInAction" (The Context)
    
    # INPUTS FOR THE AGENT
    history: str         # The 5 posts from THIS subreddit only
    target_vector: dict       # {"POWER": 0.8, ...} (extracted locally from these posts)
    
def format_history_for_prompt(history_list: list[dict]) -> str:
    formatted_str = ""
    for entry in history_list:
        sub = entry['subreddit']
        posts = "\n- ".join(entry['posts'])
        formatted_str += f"\n[Subreddit: r/{sub}]\n- {posts}\n"
    return formatted_str


def _persona_dict_to_description(persona: dict) -> str:
    """Format synthetic persona dict as a string for the YAML compiler."""
    parts = []
    if "core_identity" in persona:
        ci = persona["core_identity"]
        parts.append("### 1. CORE IDENTITY")
        parts.append(
            f"I am a {ci.get('age', '?')} year old {ci.get('occupation', '')} in {ci.get('location', '')}. "
            f"{ci.get('worldview_and_personality', '')}"
        )
    if "psychological_drivers" in persona:
        parts.append("\n### 2. PSYCHOLOGICAL DRIVERS")
        parts.append(persona["psychological_drivers"])
    if "schwartz_values" in persona:
        parts.append("\n### 3. SCHWARTZ VALUES (JSON)")
        parts.append("```json\n" + json.dumps(persona["schwartz_values"], indent=2) + "\n```")
    if "internal_monologue_style" in persona:
        parts.append("\n### 4. INTERNAL MONOLOGUE STYLE")
        parts.append(persona["internal_monologue_style"])
    return "\n".join(parts) if parts else json.dumps(persona, indent=2)


def load_persona_dataset_synthetic(path: str) -> list[PersonaDataInst_Synthetic]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst_Synthetic] = []
        for row in f:
            row = row.strip()
            if not row:
                continue
            try:
                data = json.loads(row)
                persona = data["persona"]
                schwartz = persona.get("schwartz_values", {}) if isinstance(persona, dict) else {}
                examples.append(
                    PersonaDataInst_Synthetic(
                        id=data["id"],
                        persona=persona,
                        schwartz_values=schwartz,
                    )
                )
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples

def load_persona_dataset(path: str) -> list[PersonaDataInst_GEPA]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst_GEPA] = []
        for row in f:
            try:
                data = json.loads(row)
                history = data["history"]
                formatted_history = format_history_for_prompt(history)
                target_vector = data["target_vector"]

                examples.append(PersonaDataInst_GEPA(user_id=data["user_id"], subreddits=data["subreddits"], history=formatted_history, target_vector=str(target_vector)))
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples


def load_scaled_persona_dataset(path: str) -> list[PersonaDataInst_Scaled]:
    with open(path, "r") as f:
        examples: list[PersonaDataInst_Scaled] = []
        for row in f:
            try:
                data = json.loads(row)
                examples.append(PersonaDataInst_Scaled(user_id=data["user_id"], persona=data["persona"]))
            except Exception as e:
                print(f"Error loading row: {e}")
                continue
    return examples


def load_existing_user_ids(path: str) -> set[str]:
    """Load user_ids from an existing JSONL output file."""
    existing_ids: set[str] = set()
    if not os.path.exists(path):
        return existing_ids

    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            try:
                data = json.loads(row)
                user_id = data.get("user_id")
                if user_id:
                    existing_ids.add(user_id)
            except Exception:
                # Ignore malformed rows to keep resume robust.
                continue
    return existing_ids

@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_persona_model(prompt: str) -> str:
    response_message = persona_model([{"role": "user", "content": prompt}])
    return _assistant_text(response_message) or ""


def extract_schwartz_json(persona_text: str) -> dict:
    """Parse the ```json block from the raw REDDIT_PROMPT output."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: grab any bare JSON object in the text
    match = re.search(r'\{[^{}]*"[A-Za-z]+":\s*[\d.]+.*?\}', persona_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def compile_persona(persona_description: str, critique: str = None) -> str:
    prompt = YAML_PROMPT
    if critique:
        prompt += (
            "\n\n=== REVISION CONSTRAINTS (must address before outputting YAML) ===\n"
            f"{critique}\n"
            "Address every constraint above. Do NOT rewrite from scratch; only adjust the fields that failed.\n"
        )
    prompt += f"\n\n=== RAW PERSONA ===\n{persona_description}\n\n=== YOUR YAML OUTPUT ==="
    response = persona_model([{"role": "user", "content": prompt}])
    return response.content or ""


@retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0)
def call_conformance_model(prompt: str) -> dict:
    response = persona_model([{"role": "user", "content": prompt}])
    text = _assistant_text(response) or ""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def extract_critique(conformance_result: dict) -> str:
    """Build a targeted critique from the failed checks in a conformance result."""
    lines = []
    for key, val in conformance_result.items():
        if key == "OVERALL_STATUS":
            continue
        if isinstance(val, dict) and not val.get("pass", True):
            lines.append(f"- {key}: {val.get('reason', 'No reason provided.')}")
    return "\n".join(lines) if lines else "General conformance failure — tighten IF/THEN rules and termination bounds."


#Steps:
#1. Generate a persona using the REDDIT_PROMPT
#2. Compile the persona into a YAML file using the CLAUDE_PROMPT
#3. Evaluate the YAML file using the CONFORMANCE_PROMPT - which needs schwartz values as input from original persona <- can parse from original persona or from the data we use for persona generation
#4. If the YAML file is not approved, go back to step 2 and generate a new persona with critique Pass the Critique as a Direct Constraint, not a Rewrite from Scratch
    #Implement a Hard Cutoff ($k=3$) LLMs can occasionally get stuck in a loop where fixing one constraint breaks another. To prevent infinite API calls, implement a max_retries counter. If a persona fails the conformance check 3 times, kick it out of the automated pipeline and log it as an "edge-case failure" for your methodology section. Documenting why certain value combinations fail to compile cleanly is actually a great finding for the paper.


if __name__ == "__main__":

    #validation_personas.jsonl - write userID+generated persona as {userID: persona}
    # synthetic_personas = load_persona_dataset_synthetic("/home/pgen/personagen/multiagent_human_worker/reddit/SyntheticPersonas/fakepersonas.jsonl")
    # total_users = len(synthetic_personas)
    # print(f"Loaded {total_users} users from fakepersonas.jsonl")

    # trainset = load_persona_dataset("train_gdelt_enriched.jsonl")
   
   
    testset = load_scaled_persona_dataset(
        "/home/pgen/personagen/multiagent_human_worker/reddit/ScaledPersonas/GEPAprompted.jsonl"
    )
    total_users = len(testset)
    print(f"Loaded {total_users} users from GEPAprompted.jsonl")
    # print(f"Loaded {total_users} users from GEPAprompted.jsonl")
    
   # # Load existing user_ids from output file to skip already processed users
    # output_file = "/home/pgen/personagen/multiagent_human_worker/reddit/ClusteredPersonas/pipelinedpersona.jsonl"
    # eval_folder = "/home/pgen/personagen/multiagent_human_worker/reddit/ClusteredPersonas/EvalYaml"
    
    output_file = "/home/pgen/personagen/multiagent_human_worker/reddit/ScaledPersonas/pipelinedpersona.jsonl"
    eval_folder = "/home/pgen/personagen/multiagent_human_worker/reddit/ScaledPersonas/YAML"
    os.makedirs(eval_folder, exist_ok=True)
    existing_user_ids = load_existing_user_ids(output_file)
    print(f"Found {len(existing_user_ids)} existing users in pipelinedpersona.jsonl")

    
    
    print(f"Starting pipeline...\n")
    print(
        f"LLM timeouts: connect={_connect_s}s read={_read_s}s (env LLM_CONNECT_TIMEOUT_SEC / LLM_READ_TIMEOUT_SEC)\n",
        flush=True,
    )
    
    MAX_CONFORMANCE_RETRIES = 3

    successful = 0
    failed = 0
    skipped = 0
    edge_case_failures = []

# user_id: str             # "lumenation"
#     subreddits: list[str]           # "r/KotakuInAction" (The Context)
    
#     # INPUTS FOR THE AGENT
#     history: str         # The 5 posts from THIS subreddit only
#     target_vector: dict       # {"POWER": 0

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(len(testset)):
            inst = testset[i]
            persona_id = inst.user_id
            if persona_id in existing_user_ids:
                skipped += 1
                print(f"\n[{i+1}/{total_users}] Skipping existing user: {persona_id}", flush=True)
                continue
            schwartz_json = extract_schwartz_json(str(inst.persona))
        #     subreddits = inst.subreddits
        #     history = inst.history
        #     target_vector = inst.target_vector
        #     schwartz_json = json.dumps(inst.target_vector, indent=2)
            persona_description = str(inst.persona)
            persona_description = _maybe_truncate_persona(persona_description, persona_id)
            persona = persona_description
            yaml_output_path = os.path.join(eval_folder, f"{persona_id}.yaml")
            _raw_len = len(str(inst.persona))
            _prompt_len = len(YAML_PROMPT) + _raw_len + 200
            print(
                f"\n[{i+1}/{total_users}] Processing: {persona_id} "
                f"(persona ~{_raw_len} chars, full compile prompt ~{_prompt_len} chars; "
                f"~{_prompt_len // 4} tok rough guess)",
                flush=True,
            )

            try:
                # k=3 compile → conformance → critique retry loop
                critique = None
                conformance_result = None
                approved = False

                for attempt in range(MAX_CONFORMANCE_RETRIES):
                    print(
                        f"  → compile_persona attempt {attempt + 1}/{MAX_CONFORMANCE_RETRIES} ...",
                        flush=True,
                    )
                    _t0 = time.perf_counter()
                    yaml_content = compile_persona(persona_description, critique=critique)
                    print(f"  → compile_persona finished in {time.perf_counter() - _t0:.1f}s", flush=True)

                    print(f"  → conformance check ...", flush=True)
                    conformance_result = call_conformance_model(
                        CONFORMANCE_PROMPT.format(schwartz_json=schwartz_json, yaml_file=yaml_content)
                    )
                    if not conformance_result:
                        print(f"\n  WARNING: Empty conformance result on attempt {attempt + 1}")
                        break

                    print(f"\n  Conformance attempt {attempt + 1}: {conformance_result.get('OVERALL_STATUS')}")

                    if conformance_result.get("OVERALL_STATUS") == "APPROVE":
                        approved = True
                        break

                    critique = extract_critique(conformance_result)
                    print(f"  Critique: {critique}")

                if not approved:
                    print(f"  EDGE CASE: {persona_id} failed conformance after {MAX_CONFORMANCE_RETRIES} attempts — logging.")
                    edge_case_failures.append({
                        "user_id": persona_id,
                        "last_conformance": conformance_result,
                    })
                    failed += 1
                    continue

                # Write the yaml file to the eval_folder so we can evaluate the personas after the pipeline
                with open(yaml_output_path, "w", encoding="utf-8") as file2:
                    file2.write(yaml_content)

                f.write(json.dumps({"user_id": persona_id, "persona": persona, "yaml": yaml_output_path}, ensure_ascii=False) + "\n")
                f.flush()
                existing_user_ids.add(persona_id)

                successful += 1
                print(f"  ✓ Success")

            except Exception as e:
                failed += 1
                print(f"✗ Failed: {e}")
                continue

    if edge_case_failures:
        with open("edge_case_failures.jsonl", "a", encoding="utf-8") as ef:
            for rec in edge_case_failures:
                ef.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Successful: {successful}/{total_users}")
    print(f"Skipped: {skipped}/{total_users}")
    print(f"Failed: {failed}/{total_users}")
    print(f"Edge-case failures logged: {len(edge_case_failures)}")
    print(f"{'='*50}")
