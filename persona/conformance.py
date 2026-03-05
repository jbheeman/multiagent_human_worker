"""Layer-based conformance checking for compiled persona rules.

Validates that the IF/THEN rules and termination conditions are consistent
with the selected layer vignettes. Replaces the old Schwartz-based
conformance check.
"""

import json
import re
from typing import Callable

from persona.schema import PersonaConfig


CONFORMANCE_PROMPT = """You are a Conformance Evaluator for layered persona behavioral rules.
Verify that the compiled IF/THEN rules are consistent with and specific to the source layer vignettes.

=== SOURCE LAYERS ===

Situation Construal [{sc_code}]: {sc_label}
Relational Stance [{rs_code}]: {rs_label}
Agency [{ag_code}]: {ag_label}
Epistemic [{ep_code}]: {ep_label}
Stress Response [{sr_code}]: {sr_label}
Communicative Repertoire [{cr_code}]: {cr_label}
Domain Familiarity [{df_code}]: {df_label}
Stakes [{st_code}]: {st_label}
Interaction Friction [{if_code}]: {if_label}
Emotional Entry [{es_code}]: {es_label}
Bandwidth [{bw_code}]: {bw_label}
Goal Clarity [{gc_code}]: {gc_label}

=== COMPILED RULES ===
{rules_text}

ABANDONMENT: {abandonment}

=== EVALUATION CRITERIA ===

1. Stress Response Consistency: Does the escalation/de-escalation behavior in the rules match the stress response vignette ({sr_code} = {sr_label})? A "{sr_label}" stress pattern should NOT produce behaviors from a different pattern (e.g., escalation rules for a withdrawal persona).

2. Agency-Communication Alignment: Do the rules reflect BOTH the agency level ({ag_code} = {ag_label}) AND communicative repertoire ({cr_code} = {cr_label})? A terse person with high agency behaves differently than a verbose person with high agency.

3. Specificity Check: Are the rules specific enough to distinguish this persona from a different layer combination? Would swapping ONE layer code produce meaningfully different rules? Generic rules like "gets frustrated" fail this check.

4. Communication Texture: Do the behavior descriptions reflect the communicative repertoire? If the persona is L2 English, do the example phrases show L2 patterns? If terse, are behaviors described tersely?

5. Termination Bounds: Is the abandonment condition concrete and testable? Does it reference specific layer interactions (agency + stakes + bandwidth)?

=== OUTPUT FORMAT (JSON only) ===
{{
  "stress_consistency": {{"pass": true, "reason": "..."}},
  "agency_communication": {{"pass": true, "reason": "..."}},
  "specificity_check": {{"pass": true, "reason": "..."}},
  "communication_texture": {{"pass": true, "reason": "..."}},
  "termination_check": {{"pass": true, "reason": "..."}},
  "OVERALL_STATUS": "APPROVE or REJECT"
}}"""


def _build_conformance_prompt(config: PersonaConfig) -> str:
    """Build the conformance check prompt."""
    # Format rules as text
    rules_lines = []
    for rule in config.state_transition_rules:
        prefix = "[CORE]" if rule.is_core else "[PERSONA-SPECIFIC]"
        rules_lines.append(f"{prefix} WHEN: {rule.trigger}")
        rules_lines.append(f"  YOU: {rule.behavior}")
        rules_lines.append(f"  (derived from: {rule.derived_from})")
        rules_lines.append("")
    rules_text = "\n".join(rules_lines).strip() or "(No rules compiled)"

    return CONFORMANCE_PROMPT.format(
        sc_code=config.situation_construal.code,
        sc_label=config.situation_construal.label,
        rs_code=config.relational_stance.code,
        rs_label=config.relational_stance.label,
        ag_code=config.agency.code,
        ag_label=config.agency.label,
        ep_code=config.epistemic.code,
        ep_label=config.epistemic.label,
        sr_code=config.stress_response.code,
        sr_label=config.stress_response.label,
        cr_code=config.communicative_repertoire.code,
        cr_label=config.communicative_repertoire.label,
        df_code=config.domain_familiarity.code,
        df_label=config.domain_familiarity.label,
        st_code=config.stakes.code,
        st_label=config.stakes.label,
        if_code=config.interaction_friction.code,
        if_label=config.interaction_friction.label,
        es_code=config.emotional_entry_state.code,
        es_label=config.emotional_entry_state.label,
        bw_code=config.bandwidth.code,
        bw_label=config.bandwidth.label,
        gc_code=config.goal_clarity.code,
        gc_label=config.goal_clarity.label,
        rules_text=rules_text,
        abandonment=config.termination.abandonment or "(none)",
    )


def _parse_conformance_response(response_text: str) -> dict:
    """Parse the conformance check JSON response."""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def extract_critique(result: dict) -> str:
    """Build a targeted critique from failed conformance checks.

    Used to feed back into the compiler for retry.
    """
    lines = []
    for key, val in result.items():
        if key == "OVERALL_STATUS":
            continue
        if isinstance(val, dict) and not val.get("pass", True):
            lines.append(f"- {key}: {val.get('reason', 'No reason provided.')}")
    return "\n".join(lines) if lines else "General conformance failure — make rules more specific to the layer combination."


def check_conformance(
    config: PersonaConfig,
    llm_call: Callable[[str], str],
) -> dict:
    """Run conformance check on a compiled persona.

    Args:
        config: PersonaConfig with compiled rules.
        llm_call: LLM call function.

    Returns:
        Conformance result dict with pass/fail per criterion and OVERALL_STATUS.
    """
    prompt = _build_conformance_prompt(config)
    response = llm_call(prompt)
    return _parse_conformance_response(response)


def get_conformance_prompt(config: PersonaConfig) -> str:
    """Expose the conformance prompt for debugging."""
    return _build_conformance_prompt(config)
