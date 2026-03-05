"""Lightweight IF/THEN rule compiler.

Takes a PersonaConfig (with vignettes already populated) and generates:
1. Behavioral responses for the 6 core triggers
2. 1-3 persona-specific extra triggers
3. Termination/abandonment condition

This is the only LLM call in the persona assembly pipeline (aside from
extraction for Reddit-grounded personas). The vignettes themselves go
directly into the system prompt without LLM processing.
"""

import json
import re
from typing import Any, Callable, Optional

from persona.schema import (
    CORE_TRIGGERS,
    Layer,
    PersonaConfig,
    StateTransitionRule,
    TerminationConditions,
)


COMPILER_PROMPT = """You are generating behavioral rules for a simulated customer service user.

Below is a complete persona defined by psychological layers. Your job is to generate:
1. Specific behavioral responses for 6 standard triggers
2. 1-3 additional triggers unique to this persona's layer combination
3. An abandonment condition (when this person gives up and leaves)

=== PERSONA LAYERS ===

DEEP ORIENTATIONS (Layer 0):
- Situation Construal [{l0_sc_code}]: {l0_sc_label}
  → {l0_sc_vignette}

- Relational Stance [{l0_rs_code}]: {l0_rs_label}
  → {l0_rs_vignette}

- Agency [{l0_ag_code}]: {l0_ag_label}
  → {l0_ag_vignette}

- Epistemic [{l0_ep_code}]: {l0_ep_label}
  → {l0_ep_vignette}

- Stress Response [{l0_sr_code}]: {l0_sr_label}
  → {l0_sr_vignette}

STABLE BACKGROUND (Layer 1):
- Communicative Repertoire [{l1_cr_code}]: {l1_cr_label}
  → {l1_cr_vignette}

- Domain Familiarity [{l1_df_code}]: {l1_df_label}
  → {l1_df_vignette}

- Stakes [{l1_st_code}]: {l1_st_label}
  → {l1_st_vignette}

- Interaction Friction [{l1_if_code}]: {l1_if_label}
  → {l1_if_vignette}

SITUATIONAL STATE (Layer 2):
- Emotional Entry [{l2_es_code}]: {l2_es_label}
  → {l2_es_vignette}

- Bandwidth [{l2_bw_code}]: {l2_bw_label}
  → {l2_bw_vignette}

- Goal Clarity [{l2_gc_code}]: {l2_gc_label}
  → {l2_gc_vignette}

Demographics: {demographics}

=== RULES FOR WRITING BEHAVIORS ===

Each behavior MUST be:
- OBSERVABLE: describe what they SAY or DO, not what they feel.
- LAYER-SPECIFIC: must reflect THIS specific combination. If you swapped one layer, the behavior should change.
- COMMUNICATION-TEXTURED: Dialogue MUST strictly match the "Communicative Repertoire" (Layer 1).
- VOICE-ANCHORED: When writing dialogue, write it exactly as the character would.
- NO-MARKDOWN: Never use **bold** or *italics*. Real people in chat do not use markdown. Generate plain text only.
- NO-DIALECT-HALLUCINATION: Do not introduce slang, regional dialects, or linguistic markers (e.g., "abeg", "mate", "finna") UNLESS they are explicitly mentioned in the provided vignettes. Use only the tools provided in the layers.
- NO-TIME-CONSTRAINTS: Do not generate rules based on time or response speed (e.g., "if agent takes more than 2 minutes"). Simulations are turn-based and instantaneous. Focus on the CONTENT and TONE of the agent's messages.
- CONCRETE: specific enough to verify in a transcript.

BAD example: "Gets frustrated and pushes back"
GOOD example: "Your messages get shorter. You drop the pleasantries. 'so whats the actual timeline here'"

BAD example: "Feels anxious about the delay"
GOOD example: "'ok and youre sure thats going to work? sorry I just want to make sure bc last time...'"

=== OUTPUT FORMAT (JSON only, no other text) ===
{{
  "core_rules": [
    {{
      "trigger": "{trigger_1}",
      "behavior": "...",
      "derived_from": "list which 2-4 layer codes interact to produce this behavior"
    }},
    {{
      "trigger": "{trigger_2}",
      "behavior": "...",
      "derived_from": "..."
    }},
    {{
      "trigger": "{trigger_3}",
      "behavior": "...",
      "derived_from": "..."
    }},
    {{
      "trigger": "{trigger_4}",
      "behavior": "...",
      "derived_from": "..."
    }},
    {{
      "trigger": "{trigger_5}",
      "behavior": "...",
      "derived_from": "..."
    }},
    {{
      "trigger": "{trigger_6}",
      "behavior": "...",
      "derived_from": "..."
    }}
  ],
  "persona_specific_rules": [
    {{
      "trigger": "A trigger unique to this persona's layer combination",
      "behavior": "...",
      "derived_from": "..."
    }}
  ],
  "abandonment": "When and why THIS specific person gives up. Be concrete — what specific condition causes them to leave? Reference their agency, stakes, bandwidth, and stress response."
}}"""


def _build_compiler_prompt(config: PersonaConfig) -> str:
    """Build the compiler prompt from a PersonaConfig."""
    return COMPILER_PROMPT.format(
        # Layer 0
        l0_sc_code=config.situation_construal.code,
        l0_sc_label=config.situation_construal.label,
        l0_sc_vignette=config.situation_construal.vignette,
        l0_rs_code=config.relational_stance.code,
        l0_rs_label=config.relational_stance.label,
        l0_rs_vignette=config.relational_stance.vignette,
        l0_ag_code=config.agency.code,
        l0_ag_label=config.agency.label,
        l0_ag_vignette=config.agency.vignette,
        l0_ep_code=config.epistemic.code,
        l0_ep_label=config.epistemic.label,
        l0_ep_vignette=config.epistemic.vignette,
        l0_sr_code=config.stress_response.code,
        l0_sr_label=config.stress_response.label,
        l0_sr_vignette=config.stress_response.vignette,
        # Layer 1
        l1_cr_code=config.communicative_repertoire.code,
        l1_cr_label=config.communicative_repertoire.label,
        l1_cr_vignette=config.communicative_repertoire.vignette,
        l1_df_code=config.domain_familiarity.code,
        l1_df_label=config.domain_familiarity.label,
        l1_df_vignette=config.domain_familiarity.vignette,
        l1_st_code=config.stakes.code,
        l1_st_label=config.stakes.label,
        l1_st_vignette=config.stakes.vignette,
        l1_if_code=config.interaction_friction.code,
        l1_if_label=config.interaction_friction.label,
        l1_if_vignette=config.interaction_friction.vignette,
        # Layer 2
        l2_es_code=config.emotional_entry_state.code,
        l2_es_label=config.emotional_entry_state.label,
        l2_es_vignette=config.emotional_entry_state.vignette,
        l2_bw_code=config.bandwidth.code,
        l2_bw_label=config.bandwidth.label,
        l2_bw_vignette=config.bandwidth.vignette,
        l2_gc_code=config.goal_clarity.code,
        l2_gc_label=config.goal_clarity.label,
        l2_gc_vignette=config.goal_clarity.vignette,
        # Metadata
        demographics=config.demographics or "Not specified",
        # Core triggers
        trigger_1=CORE_TRIGGERS[0],
        trigger_2=CORE_TRIGGERS[1],
        trigger_3=CORE_TRIGGERS[2],
        trigger_4=CORE_TRIGGERS[3],
        trigger_5=CORE_TRIGGERS[4],
        trigger_6=CORE_TRIGGERS[5],
    )


def _parse_compiler_response(response_text: str) -> dict:
    """Parse the JSON response from the compiler LLM."""
    # Try to extract JSON from the response
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def compile_rules(
    config: PersonaConfig,
    llm_call: Callable[[str], str],
) -> PersonaConfig:
    """Compile IF/THEN rules for a persona using an LLM.

    Args:
        config: PersonaConfig with all 12 layer selections populated.
        llm_call: Function that takes a prompt string and returns a response string.
                  This abstracts the specific LLM backend (Qwen3, OpenAI, etc.)

    Returns:
        The same PersonaConfig with state_transition_rules and termination populated.
    """
    prompt = _build_compiler_prompt(config)
    response = llm_call(prompt)
    parsed = _parse_compiler_response(response)

    if not parsed:
        return config

    # Parse core rules
    rules = []
    for rule_data in parsed.get("core_rules", []):
        rules.append(StateTransitionRule(
            trigger=rule_data.get("trigger", ""),
            behavior=rule_data.get("behavior", ""),
            derived_from=rule_data.get("derived_from", ""),
            is_core=True,
        ))

    # Parse persona-specific rules
    for rule_data in parsed.get("persona_specific_rules", []):
        rules.append(StateTransitionRule(
            trigger=rule_data.get("trigger", ""),
            behavior=rule_data.get("behavior", ""),
            derived_from=rule_data.get("derived_from", ""),
            is_core=False,
        ))

    config.state_transition_rules = rules
    config.termination = TerminationConditions(
        abandonment=parsed.get("abandonment", "")
    )

    return config


def get_compiler_prompt(config: PersonaConfig) -> str:
    """Expose the compiler prompt for debugging/inspection."""
    return _build_compiler_prompt(config)
