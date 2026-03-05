"""Data models for the layered persona system.

Defines PersonaConfig, LayerSelection, and related types for the 3-layer
persona architecture (Layer 0: Deep Substrate, Layer 1: Stable Background,
Layer 2: Situational State).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GenerationMode(Enum):
    """How a persona's layer codes were determined."""
    FULLY_GROUNDED = "fully_grounded"       # L0 + L1 from Reddit, L2 sampled
    PARTIALLY_GROUNDED = "partially_grounded"  # L0 from Reddit, L1 + L2 sampled
    SYNTHETIC = "synthetic"                 # All layers sampled


class Layer(Enum):
    """Which layer a dimension belongs to."""
    L0 = 0  # Deep Substrate
    L1 = 1  # Stable Background
    L2 = 2  # Situational State


# Dimension metadata: (dimension_key, layer, code_prefix, num_options)
DIMENSIONS = {
    "situation_construal":      (Layer.L0, "1", 10),  # 1A-1J
    "relational_stance":        (Layer.L0, "2", 10),  # 2A-2J
    "agency":                   (Layer.L0, "3", 8),   # 3A-3H
    "epistemic":                (Layer.L0, "4", 8),   # 4A-4H
    "stress_response":          (Layer.L0, "5", 10),  # 5A-5J
    "communicative_repertoire": (Layer.L1, "C", 8),   # C1-C8
    "domain_familiarity":       (Layer.L1, "D", 8),   # D1-D8
    "stakes":                   (Layer.L1, "S", 8),   # S1-S8
    "interaction_friction":     (Layer.L1, "F", 10),  # F1-F10
    "emotional_entry_state":    (Layer.L2, "E", 10),  # E1-E10
    "bandwidth":                (Layer.L2, "B", 8),   # B1-B8
    "goal_clarity":             (Layer.L2, "G", 10),  # G1-G10
}

# Letter suffix for code generation (1-based index → letter)
_CODE_LETTERS = "ABCDEFGHIJKL"


def code_for(prefix: str, index: int) -> str:
    """Generate a code like '1A', 'C12', 'E10' from prefix and 1-based index.

    For numeric prefixes (1-5), uses letters: 1A, 1B, ...
    For letter prefixes (C, D, S, F, E, B, G), uses numbers: C1, C2, ...
    """
    if prefix.isdigit():
        return f"{prefix}{_CODE_LETTERS[index - 1]}"
    else:
        return f"{prefix}{index}"


def all_codes_for(dimension: str) -> list[str]:
    """Return all valid codes for a dimension."""
    _, prefix, num_options = DIMENSIONS[dimension]
    return [code_for(prefix, i) for i in range(1, num_options + 1)]


@dataclass
class LayerSelection:
    """A single dimension selection with its code, label, and vignette text."""
    dimension: str   # e.g., "situation_construal"
    code: str        # e.g., "1A"
    label: str       # e.g., "Rights exercise"
    vignette: str    # Full prose text from the spec


@dataclass
class StateTransitionRule:
    """A single IF/THEN behavioral rule."""
    trigger: str       # The WHEN condition
    behavior: str      # The THEN behavior (observable, layer-specific)
    derived_from: str   # Which layer codes produced this rule
    is_core: bool = True  # False for persona-specific extras


@dataclass
class TerminationConditions:
    """When the persona ends the conversation."""
    success: str = "The final database state matches the initial goal."
    abandonment: str = ""  # LLM-generated, persona-specific


@dataclass
class PersonaConfig:
    """Complete 3-layer persona configuration.

    A PersonaConfig holds the 12 layer selections, metadata about how
    it was generated, and the compiled behavioral rules.
    """

    # Layer 0: Deep Substrate
    situation_construal: LayerSelection = field(default=None)
    relational_stance: LayerSelection = field(default=None)
    agency: LayerSelection = field(default=None)
    epistemic: LayerSelection = field(default=None)
    stress_response: LayerSelection = field(default=None)

    # Layer 1: Stable Background
    communicative_repertoire: LayerSelection = field(default=None)
    domain_familiarity: LayerSelection = field(default=None)
    stakes: LayerSelection = field(default=None)
    interaction_friction: LayerSelection = field(default=None)

    # Layer 2: Situational State
    emotional_entry_state: LayerSelection = field(default=None)
    bandwidth: LayerSelection = field(default=None)
    goal_clarity: LayerSelection = field(default=None)

    # Compiled behavioral rules (populated by compiler)
    state_transition_rules: list[StateTransitionRule] = field(default_factory=list)
    termination: TerminationConditions = field(default_factory=TerminationConditions)

    # Metadata
    mode: GenerationMode = GenerationMode.SYNTHETIC
    demographics: str = ""
    source_user_id: Optional[str] = None
    source_subreddit: Optional[str] = None
    extraction_justification: Optional[dict] = None

    def get_layer_selections(self, layer: Layer) -> list[LayerSelection]:
        """Return all selections for a given layer."""
        mapping = {
            Layer.L0: [self.situation_construal, self.relational_stance,
                       self.agency, self.epistemic, self.stress_response],
            Layer.L1: [self.communicative_repertoire, self.domain_familiarity,
                       self.stakes, self.interaction_friction],
            Layer.L2: [self.emotional_entry_state, self.bandwidth,
                       self.goal_clarity],
        }
        return mapping[layer]

    def get_all_selections(self) -> list[LayerSelection]:
        """Return all 12 selections in layer order."""
        selections = []
        for layer in Layer:
            selections.extend(self.get_layer_selections(layer))
        return selections

    @property
    def code_string(self) -> str:
        """Compact representation: '1A-2E-3C-4D-5J/C8-D3-S5-F3/E8-B3-G4'"""
        l0 = "-".join(s.code for s in self.get_layer_selections(Layer.L0))
        l1 = "-".join(s.code for s in self.get_layer_selections(Layer.L1))
        l2 = "-".join(s.code for s in self.get_layer_selections(Layer.L2))
        return f"{l0}/{l1}/{l2}"

    @property
    def layer_codes_dict(self) -> dict[str, str]:
        """Return dict of dimension → code for YAML serialization."""
        return {s.dimension: s.code for s in self.get_all_selections()}

    def get_core_rules(self) -> list[StateTransitionRule]:
        """Return only the 6 core state transition rules."""
        return [r for r in self.state_transition_rules if r.is_core]

    def get_persona_specific_rules(self) -> list[StateTransitionRule]:
        """Return only the persona-specific extra rules."""
        return [r for r in self.state_transition_rules if not r.is_core]

    def is_complete(self) -> bool:
        """Check if all 12 dimensions have been assigned."""
        return all(s is not None for s in self.get_all_selections())

    def to_yaml_dict(self) -> dict:
        """Serialize to the lightweight YAML schema (layer codes + rules + termination)."""
        return {
            "persona_profile": {
                "id": self.code_string,
                "demographics": self.demographics,
                "mode": self.mode.value,
                "source_user_id": self.source_user_id,
                "source_subreddit": self.source_subreddit,
                "layer_codes": self.layer_codes_dict,
            },
            "state_transition_rules": {
                "core": [
                    {
                        "trigger": r.trigger,
                        "behavior": r.behavior,
                        "derived_from": r.derived_from,
                    }
                    for r in self.get_core_rules()
                ],
                "persona_specific": [
                    {
                        "trigger": r.trigger,
                        "behavior": r.behavior,
                        "derived_from": r.derived_from,
                    }
                    for r in self.get_persona_specific_rules()
                ],
            },
            "termination_conditions": {
                "success": self.termination.success,
                "abandonment": self.termination.abandonment,
            },
        }


# The 6 core triggers (fixed for all personas)
CORE_TRIGGERS = [
    "Agent delays or asks you to wait",
    "Agent denies your request citing policy",
    "Agent makes an error or gives you wrong information",
    "Agent asks for information you don't have handy right now",
    "The process gets complicated or involves multiple steps",
    "Agent is warm, competent, and acknowledges your situation",
]
