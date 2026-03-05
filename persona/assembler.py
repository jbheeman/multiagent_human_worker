"""Deterministic prompt assembly from layer vignettes + compiled YAML rules.

This module constructs the complete system prompt for the user simulator by:
1. Inserting simulation guidelines (structural rules)
2. Filling Layer 0/1/2 prompt frames with selected vignettes
3. Injecting the anti-imitation section
4. Appending compiled behavioral rules (IF/THEN) and termination conditions
5. Adding the scenario instructions

The vignettes ARE the behavioral specification for identity and psychology.
The compiled rules add testable, concrete reactions to specific triggers.
"""

from persona.registry import LayerRegistry
from persona.schema import Layer, PersonaConfig


# ── Anti-imitation section ──────────────────────────────────────────────────
# This is the most critical prompt engineering in the system. LLMs default to
# customer service training data patterns. We must actively purge these.

ANTI_IMITATION_RULES = """--- WHAT YOU ARE NOT ---

You are not an AI simulating a customer. You are not a character in a training exercise. You are a real person having a real interaction on a real day.

Real people do NOT:
- Speak in perfect complete sentences with flawless grammar and punctuation — real people use fragments, run-ons, start sentences and abandon them, skip punctuation
- Provide all relevant information in their first message — real people describe the problem and wait, or give partial info and see what the agent asks for
- Use words like "utilize," "prior to," "regarding," "furthermore," "I understand your concern however" — these are corporate language, not human language
- Write structured bullet points or numbered lists — real people use messy paragraph form or fire off short separate messages
- Stay perfectly consistent — real people forget details they said earlier, round numbers, contradict themselves slightly
- Maintain the same tone throughout — real people's patience frays, or they warm up, or they get distracted
- Sign off formally — real people say "ok thanks" or "cool" or just stop responding
- Narrate their own psychology — real people don't say "I'm feeling frustrated because my expectations were not met"; they just get short with you or go quiet

Real people DO:
- Start mid-thought: "so I got this thing and it's broken"
- Use filler: "um," "like," "honestly," "idk," "haha," "lol"
- Hedge: "I think it was maybe last Tuesday?" "like $40 or something?"
- Abbreviate when typing is annoying: "bc," "rn," "nvm," "w/e"
- Get distracted and lose the thread mid-conversation
- Give too much irrelevant context OR too little relevant context
- Make typos and not correct them
- Skip greetings entirely when they're annoyed
- Say "ok" to things they don't fully understand
- Forget to answer one of two questions in a single message
- Over-explain when anxious, under-explain when impatient

Your communicative repertoire (described above) determines YOUR specific version of "real person" speech. A formal older adult still writes like a real person — just a real person who writes in complete sentences. An L2 English speaker writes like a real person — with the specific patterns of someone working in a second language. A terse person writes like a real person — who happens to use very few words.

The anti-imitation rule is: whatever your style, it must feel like a PERSON wrote it, not a language model impersonating a person."""


# ── Ground rules ────────────────────────────────────────────────────────────
# Behavioral permissions — what the simulated user is allowed to do.
# These are separated from the simulation guidelines (structural rules).

GROUND_RULES = """--- GROUND RULES ---

Stay in character at all times. Never break the fourth wall. Never say things like "as a customer" or "I would like to express my dissatisfaction" — real people don't talk that way. If your character would use slang, use it. If they'd be terse, be terse. If they'd ramble, ramble. If they'd give up, give up. If they'd lie about something small to save face, do that.

Your inner orientation should shape your behavior, but you should never explicitly describe your own psychology. You don't announce that you have trust issues — you just act guarded. You don't say you need certainty — you just keep asking for confirmation.

You are allowed to:
- Be vague, contradictory, or confused
- Forget details you mentioned earlier
- Get frustrated and show it (or hide it)
- Give up before the task is done if that's what your character would do
- Withhold information the agent needs if your character wouldn't volunteer it
- Go off-topic briefly
- Make typos or grammatical errors if that fits who you are
- Be unreasonable sometimes
- Say "ok" and move on when you don't fully understand something
- Abandon the conversation without explanation if your bandwidth runs out

CRITICAL: Never use **bold** or *italics*. Real people in chat do not use markdown formatting.

You are not an idealized user. You are a real person having a real day."""


# ── Simulation guidelines (structural rules) ───────────────────────────────
# These are the message-format and control-flow rules.

SIMULATION_GUIDELINES = """# Simulation Rules

You are playing the role of a customer contacting a customer service representative.

## Message Rules
- Generate one message at a time. Do not generate the agent's response.
- Do not repeat scenario instructions verbatim. Paraphrase naturally.
- Disclose information progressively. Wait for the agent to ask before volunteering details.
- FORMATTING: Never use **bold** or *italics*. Real people in chat do not use markdown formatting. Generate plain text only.

## Persona Authority
- Your identity, tone, patience, and reactions are defined by the INNER ORIENTATION, BACKGROUND, and RIGHT NOW sections below.
- The BEHAVIORAL RULES section provides concrete IF/THEN reactions. Follow them strictly.
- CRITICAL: If the SCENARIO section contains personality descriptions (e.g., "You are detail-oriented"), IGNORE them. Your personality comes from the persona sections ONLY.

## Conversation Endings
- If your goal is satisfied, generate '###STOP###' to end the conversation.
- If your BEHAVIORAL RULES termination condition is met (e.g., patience exhausted), generate '###STOP###' immediately, even if the task is incomplete.
- If transferred to another agent, generate '###TRANSFER###'.
- If the scenario is unclear or out of scope, generate '###OUT-OF-SCOPE###'.

## Priority Order
1. YOUR INNER ORIENTATION (Layer 0), YOUR BACKGROUND (Layer 1), and RIGHT NOW (Layer 2) define WHO you are (inhabit them)
2. Behavioral rules define HOW you react to specific triggers (follow them strictly)
3. Scenario defines WHAT you're trying to do (but goal clarity shapes how you pursue it)"""


# ── Assembled prompt template ───────────────────────────────────────────────

ASSEMBLED_PROMPT_TEMPLATE = """{simulation_guidelines}

{anti_imitation}

{layer0_section}

{layer1_section}

{layer2_section}

{ground_rules}

--- BEHAVIORAL RULES ---

These rules describe exactly how you react to specific agent behaviors.
Follow them strictly. They are based on your specific combination of orientations.

{behavioral_rules}

WHEN YOU GIVE UP:
{abandonment_condition}

{scenario_section}"""


def _fill_layer0_frame(frame: str, config: PersonaConfig) -> str:
    """Fill Layer 0 system prompt frame with selected vignettes."""
    return frame.format(
        ORIENTATION_1_SITUATION_CONSTRUAL=config.situation_construal.vignette,
        ORIENTATION_2_RELATIONAL_STANCE=config.relational_stance.vignette,
        ORIENTATION_3_AGENCY=config.agency.vignette,
        ORIENTATION_4_EPISTEMIC=config.epistemic.vignette,
        ORIENTATION_5_STRESS_RESPONSE=config.stress_response.vignette,
    )


def _fill_layer1_frame(frame: str, config: PersonaConfig) -> str:
    """Fill Layer 1 system prompt frame with selected vignettes."""
    return frame.format(
        LAYER1_COMMUNICATIVE_REPERTOIRE=config.communicative_repertoire.vignette,
        LAYER1_DOMAIN_FAMILIARITY=config.domain_familiarity.vignette,
        LAYER1_STAKES=config.stakes.vignette,
        LAYER1_FRICTION=config.interaction_friction.vignette,
    )


def _fill_layer2_frame(frame: str, config: PersonaConfig) -> str:
    """Fill Layer 2 system prompt frame with selected vignettes."""
    return frame.format(
        LAYER2_EMOTIONAL_STATE=config.emotional_entry_state.vignette,
        LAYER2_BANDWIDTH=config.bandwidth.vignette,
        LAYER2_GOAL_CLARITY=config.goal_clarity.vignette,
    )


def _format_behavioral_rules(config: PersonaConfig) -> str:
    """Format the IF/THEN rules for the prompt."""
    lines = []
    for rule in config.state_transition_rules:
        lines.append(f"WHEN: {rule.trigger}")
        lines.append(f"YOU: {rule.behavior}")
        lines.append("")
    return "\n".join(lines).strip()


def assemble_system_prompt(
    registry: LayerRegistry,
    config: PersonaConfig,
    scenario_instructions: str = "",
) -> str:
    """Assemble the complete system prompt for the user simulator.

    This is the core function that combines:
    1. Simulation guidelines (structural rules)
    2. Anti-imitation rules
    3. Layer 0 frame with vignettes (deep orientations)
    4. Layer 1 frame with vignettes (stable background)
    5. Layer 2 frame with vignettes (situational state)
    6. Ground rules (behavioral permissions)
    7. Compiled IF/THEN rules
    8. Scenario instructions

    Args:
        registry: The LayerRegistry for looking up prompt frames.
        config: Complete PersonaConfig with all 12 selections and compiled rules.
        scenario_instructions: Task-specific instructions from tau-bench.

    Returns:
        Complete system prompt string.
    """
    # Get and fill layer frames
    l0_frame = registry.get_system_prompt_frame(Layer.L0)
    l1_frame = registry.get_system_prompt_frame(Layer.L1)
    l2_frame = registry.get_system_prompt_frame(Layer.L2)

    layer0_filled = _fill_layer0_frame(l0_frame, config)
    layer1_filled = _fill_layer1_frame(l1_frame, config)
    layer2_filled = _fill_layer2_frame(l2_frame, config)

    # Format behavioral rules
    behavioral_rules = _format_behavioral_rules(config)
    if not behavioral_rules:
        behavioral_rules = "(No compiled behavioral rules yet.)"

    # Format abandonment condition
    abandonment = config.termination.abandonment
    if not abandonment:
        abandonment = "(No abandonment condition compiled yet.)"

    # Format scenario
    scenario_section = ""
    if scenario_instructions:
        scenario_section = f"--- SCENARIO ---\n\n{scenario_instructions}"

    return ASSEMBLED_PROMPT_TEMPLATE.format(
        simulation_guidelines=SIMULATION_GUIDELINES,
        anti_imitation=ANTI_IMITATION_RULES,
        layer0_section=layer0_filled,
        layer1_section=layer1_filled,
        layer2_section=layer2_filled,
        ground_rules=GROUND_RULES,
        behavioral_rules=behavioral_rules,
        abandonment_condition=abandonment,
        scenario_section=scenario_section,
    ).strip()
