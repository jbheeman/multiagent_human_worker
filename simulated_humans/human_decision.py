from typing import Literal, TypedDict, Dict, Any

class HumanDecision(TypedDict, total=False):
    """Normalized decision format for all human interactions.
    
    Decision types:
    - approve: Go ahead with the proposed action (plan/guard phases)
    - deny: Block the action, don't proceed (plan/guard phases)
    - revise: Change the approach (plan/guard phases, revisions contains corrections)
    - suggest: Provide help/next steps (help phase, revisions contains hints)
    - verify_ok: Result is acceptable (verify phase)
    - verify_fail: Result needs work (verify phase, message contains issues)
    """
    decision: Literal["approve", "deny", "revise", "suggest", "verify_ok", "verify_fail"]
    message: str                 # brief rationale
    revisions: Dict[str, Any]    # optional minimal edits, corrections, or next-step hints
