"""Canonical valence mapping for all persona dimension codes.

Every code across all 12 dimensions has exactly one valence on a 5-point scale.
This is the single source of truth — both the sampler (for biased sampling) and
the categorizer (for difficulty scoring) derive from this map.

Valence semantics (from the agent's perspective):
    POSITIVE           — clearly makes the interaction easier
    SLIGHTLY_POSITIVE  — leans easier but not dramatically
    NEUTRAL            — neither helps nor hinders
    SLIGHTLY_NEGATIVE  — leans harder but manageable
    NEGATIVE           — clearly makes the interaction harder

The mapping is derived from reading each vignette in specs/layer{0,1,2}_persona_spec.md
and asking: "Does this trait make it easier or harder for a customer-service agent
to resolve the user's issue effectively?"
"""

from enum import Enum

from persona.schema import DIMENSIONS


class Valence(Enum):
    POSITIVE = "positive"
    SLIGHTLY_POSITIVE = "slightly_positive"
    NEUTRAL = "neutral"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEGATIVE = "negative"


# ---------------------------------------------------------------------------
# Canonical code → valence mapping
# ---------------------------------------------------------------------------

CODE_VALENCE: dict[str, Valence] = {
    # ── Layer 0: Deep Persona Substrate ────────────────────────────────────
    #
    # Orientation 1: Situation Construal
    "1A": Valence.SLIGHTLY_POSITIVE,  # Rights exercise — clear framing
    "1B": Valence.NEUTRAL,            # Asking a favor
    "1C": Valence.POSITIVE,           # Rational problem-solving
    "1D": Valence.NEGATIVE,           # Threat response
    "1E": Valence.POSITIVE,           # Collaborative partnership
    "1F": Valence.NEUTRAL,            # Information gathering
    "1G": Valence.SLIGHTLY_NEGATIVE,  # Status assertion
    "1H": Valence.NEUTRAL,            # Routine transaction
    "1I": Valence.SLIGHTLY_NEGATIVE,  # Emotional processing
    "1J": Valence.NEUTRAL,            # Advocacy for another

    # Orientation 2: Relational Stance
    "2A": Valence.POSITIVE,           # Default trust
    "2B": Valence.NEUTRAL,            # Cautious-pragmatic
    "2C": Valence.SLIGHTLY_NEGATIVE,  # Testing-evaluative
    "2D": Valence.NEGATIVE,           # History of bad experiences
    "2E": Valence.POSITIVE,           # Empathetic peer
    "2F": Valence.SLIGHTLY_POSITIVE,  # Authority deference
    "2G": Valence.NEGATIVE,           # Adversarial negotiator
    "2H": Valence.NEGATIVE,           # Bot-frustrated
    "2I": Valence.POSITIVE,           # Grateful and surprised
    "2J": Valence.SLIGHTLY_POSITIVE,  # Consumer authority (assertive but fair)

    # Orientation 3: Agency
    "3A": Valence.POSITIVE,           # Full initiative
    "3B": Valence.SLIGHTLY_POSITIVE,  # Collaborative partner
    "3C": Valence.SLIGHTLY_NEGATIVE,  # Tentative
    "3D": Valence.SLIGHTLY_POSITIVE,  # Learned persistence
    "3E": Valence.NEGATIVE,           # Passive / fatalistic
    "3F": Valence.NEUTRAL,            # Procedural follower
    "3G": Valence.NEGATIVE,           # Dependent
    "3H": Valence.NEUTRAL,            # Opportunistic

    # Orientation 4: Epistemic
    "4A": Valence.POSITIVE,           # Just-fix-it
    "4B": Valence.NEUTRAL,            # Full-landscape explorer
    "4C": Valence.SLIGHTLY_POSITIVE,  # Trust-authority
    "4D": Valence.SLIGHTLY_NEGATIVE,  # Skeptic-verifier
    "4E": Valence.SLIGHTLY_NEGATIVE,  # Social validator
    "4F": Valence.NEGATIVE,           # Anxious verifier (repeats questions)
    "4G": Valence.NEUTRAL,            # Experiential learner
    "4H": Valence.SLIGHTLY_POSITIVE,  # Pattern matcher

    # Orientation 5: Stress Response
    "5A": Valence.NEGATIVE,           # Escalation
    "5B": Valence.POSITIVE,           # Withdrawal (backs off)
    "5C": Valence.SLIGHTLY_NEGATIVE,  # Tangential venting
    "5D": Valence.SLIGHTLY_NEGATIVE,  # Hyper-focus on detail
    "5E": Valence.POSITIVE,           # Resigned compliance
    "5F": Valence.NEUTRAL,            # Humor deflection
    "5G": Valence.NEGATIVE,           # Repeated insistence
    "5H": Valence.POSITIVE,           # Constructive feedback
    "5I": Valence.NEGATIVE,           # Delayed fuse (patient then snaps)
    "5J": Valence.SLIGHTLY_POSITIVE,  # Self-blame

    # ── Layer 1: Stable Background ─────────────────────────────────────────
    #
    # Communicative Repertoire
    "C1": Valence.POSITIVE,           # Standard fluent (SAE)
    "C2": Valence.NEUTRAL,            # Southern / regional dialect
    "C3": Valence.SLIGHTLY_POSITIVE,  # Formal register
    "C4": Valence.POSITIVE,           # Conversational / warm
    "C5": Valence.SLIGHTLY_POSITIVE,  # L2 high competence
    "C6": Valence.NEGATIVE,           # L2 working competence
    "C7": Valence.NEGATIVE,           # Fragmented digital native
    "C8": Valence.NEUTRAL,            # Technical expert jargon

    # Domain Familiarity
    "D1": Valence.POSITIVE,           # Veteran customer
    "D2": Valence.SLIGHTLY_POSITIVE,  # Generally competent consumer
    "D3": Valence.NEGATIVE,           # First-timer (no vocabulary)
    "D4": Valence.NEUTRAL,            # Moderate familiarity
    "D5": Valence.SLIGHTLY_POSITIVE,  # Adjacent domain expert
    "D6": Valence.POSITIVE,           # Experienced with this company
    "D7": Valence.NEGATIVE,           # Confused about scope
    "D8": Valence.NEGATIVE,           # Actively misinformed

    # Stakes Context
    "S1": Valence.POSITIVE,           # Trivial (will drop if complicated)
    "S2": Valence.SLIGHTLY_POSITIVE,  # Moderate and proportional
    "S3": Valence.NEUTRAL,            # Financially significant
    "S4": Valence.NEGATIVE,           # Time-urgent deadline
    "S5": Valence.NEGATIVE,           # Financially critical
    "S6": Valence.NEGATIVE,           # Principle dispute
    "S7": Valence.SLIGHTLY_NEGATIVE,  # Gift / 3rd-party
    "S8": Valence.NEGATIVE,           # Repeat problem

    # Interaction Friction
    "F1": Valence.POSITIVE,           # No significant friction
    "F2": Valence.SLIGHTLY_NEGATIVE,  # Phone typing (typos, brevity)
    "F3": Valence.SLIGHTLY_NEGATIVE,  # Slow typist
    "F4": Valence.NEGATIVE,           # Attention issues (ADHD-like)
    "F5": Valence.NEGATIVE,           # Vision difficulty
    "F6": Valence.SLIGHTLY_NEGATIVE,  # Voice-to-text artifacts
    "F7": Valence.SLIGHTLY_NEGATIVE,  # Multitasking
    "F8": Valence.NEUTRAL,            # Unfamiliar with chat interface
    "F9": Valence.NEGATIVE,           # Memory / tracking difficulty
    "F10": Valence.NEGATIVE,          # Language processing lag

    # ── Layer 2: Situational State ─────────────────────────────────────────
    #
    # Emotional Entry State
    "E1": Valence.POSITIVE,           # Neutral / baseline
    "E2": Valence.SLIGHTLY_NEGATIVE,  # Mildly annoyed
    "E3": Valence.NEGATIVE,           # Frustrated
    "E4": Valence.SLIGHTLY_NEGATIVE,  # Anxious
    "E5": Valence.NEUTRAL,            # Resigned / tired
    "E6": Valence.NEGATIVE,           # Angry
    "E7": Valence.POSITIVE,           # Cheerful / easy-going
    "E8": Valence.SLIGHTLY_NEGATIVE,  # Stressed / overwhelmed
    "E9": Valence.NEUTRAL,            # Embarrassed
    "E10": Valence.POSITIVE,          # Hopeful

    # Bandwidth
    "B1": Valence.POSITIVE,           # Full capacity
    "B2": Valence.SLIGHTLY_POSITIVE,  # Normal / moderate
    "B3": Valence.SLIGHTLY_NEGATIVE,  # Rushed
    "B4": Valence.NEGATIVE,           # Exhausted / depleted
    "B5": Valence.SLIGHTLY_NEGATIVE,  # Distracted
    "B6": Valence.POSITIVE,           # Hyper-focused
    "B7": Valence.NEGATIVE,           # Stolen moment (hard time limit)
    "B8": Valence.NEUTRAL,            # Winding down

    # Goal Clarity
    "G1": Valence.POSITIVE,           # Fully crystallized
    "G2": Valence.SLIGHTLY_POSITIVE,  # Clear goal, flexible on details
    "G3": Valence.NEUTRAL,            # Partially formed
    "G4": Valence.NEGATIVE,           # Conflicting goals
    "G5": Valence.NEUTRAL,            # Evolving
    "G6": Valence.NEGATIVE,           # Wrong mental model
    "G7": Valence.NEUTRAL,            # Delegating
    "G8": Valence.NEGATIVE,           # Emotionally driven
    "G9": Valence.SLIGHTLY_NEGATIVE,  # Testing / exploring
    "G10": Valence.SLIGHTLY_NEGATIVE, # Vague / exploratory
}


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Dimensions whose relationship to agent performance is an empirical
# measurement target, not a prior assumption.  Still in CODE_VALENCE
# (sampler uses them) but excluded from difficulty scoring.
EXCLUDED_DIMENSIONS = frozenset({"situation_construal", "agency", "epistemic"})

# The 9 dimensions that contribute to difficulty scoring.
SCORED_DIMENSIONS = frozenset(
    dim for dim in DIMENSIONS if dim not in EXCLUDED_DIMENSIONS
)

# Numeric weights for scoring (maps 5-level valence to [-2, +2]).
VALENCE_WEIGHT: dict[Valence, int] = {
    Valence.POSITIVE: -2,
    Valence.SLIGHTLY_POSITIVE: -1,
    Valence.NEUTRAL: 0,
    Valence.SLIGHTLY_NEGATIVE: 1,
    Valence.NEGATIVE: 2,
}
