"""Evaluation metrics for persona-conditioned agent evaluation.

Implements the metrics from the paper:
  "Persona-Conditioned Evaluation of Agentic Systems" (COLM 2026)

Two metric families:

1. **Objective metrics** (no LLM judge):
   - Task success (tau-bench reward)
   - Robustness: cross-persona variance, tail risk, persona gap (Section 2.3)
   - Trait sensitivity: E[M(A,p) | t_j = v] per dimension (Section 2.3)
   - Table 2 proxies: goal achievement, cognitive effort, intent alignment
   - Bootstrap confidence intervals (Section 2.5)

2. **Calibrated LLM judge** (for Trust + Use Again only):
   - Anchor-calibrated prompts
   - k=3 multi-call agreement (median / majority vote)
   - Judge reliability (Fleiss' kappa)

Internal QA metrics (humanness patterns, transcript diversity) are kept
but separated from paper metrics.
"""

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
# 1. TASK EXTRACTION (from tau-bench SimulationRun dicts)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskMetrics:
    """Task-level metrics extracted from a simulation run."""
    task_id: str
    reward: float                   # 0.0 or 1.0
    termination_reason: str
    num_messages: int
    duration: float
    persona_id: str = ""            # code_string or "baseline"
    persona_codes: dict = field(default_factory=dict)  # {dim: code} for trait sensitivity
    condition: str = ""             # "layer", "none", etc.
    # Table 2 proxies (populated by compute_table2_proxies)
    goal_achievement: float = 0.0   # 1-7
    cognitive_effort: float = 0.0   # 1-7 (higher = harder)
    intent_alignment: float = 0.0   # 1-7 (higher = better)
    # LLM judge scores (populated post-hoc or from reward_info)
    trust: float = 0.0              # 1-7
    use_again: float = 0.0          # 0 or 1


def extract_task_metrics(
    simulation_run: dict,
    condition: str = "",
    persona_id: str = "",
    persona_codes: Optional[dict] = None,
) -> TaskMetrics:
    """Extract task metrics from a tau-bench SimulationRun dict."""
    reward_info = simulation_run.get("reward_info", {}) or {}
    reward = reward_info.get("reward", 0.0) if reward_info else 0.0
    messages = simulation_run.get("messages", [])

    # Extract critic scores if present
    info = reward_info.get("info", {}) or {}
    critic = info.get("persona_critic", {}) or {}

    tm = TaskMetrics(
        task_id=simulation_run.get("task_id", ""),
        reward=reward,
        termination_reason=simulation_run.get("termination_reason", ""),
        num_messages=len(messages),
        duration=simulation_run.get("duration", 0.0),
        persona_id=persona_id or simulation_run.get("_persona_id", ""),
        persona_codes=persona_codes or simulation_run.get("_persona_codes", {}),
        condition=condition or simulation_run.get("_condition", ""),
    )

    # Compute transcript-derived proxies
    tm.goal_achievement = compute_goal_achievement(reward)
    tm.cognitive_effort = compute_cognitive_effort(messages)
    tm.intent_alignment = compute_intent_alignment(messages, reward)

    # Pull LLM judge scores from critic if available
    if isinstance(critic, dict):
        tm.trust = critic.get("trust", 0.0)
        tm.use_again = critic.get("use_again", 0.0)

    return tm


def aggregate_task_metrics(metrics: list[TaskMetrics]) -> dict:
    """Aggregate task metrics across simulations."""
    if not metrics:
        return {}
    rewards = [m.reward for m in metrics]
    durations = [m.duration for m in metrics]
    msg_counts = [m.num_messages for m in metrics]
    terminations = Counter(m.termination_reason for m in metrics)

    return {
        "n": len(metrics),
        "task_success_rate": _mean(rewards),
        "mean_reward": _mean(rewards),
        "mean_messages": _mean(msg_counts),
        "mean_duration": _mean(durations),
        "termination_reasons": dict(terminations),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. TRANSCRIPT-DERIVED PROXIES (Table 2 — no LLM judge)
# ═══════════════════════════════════════════════════════════════════════════

def extract_user_messages(messages: list[dict]) -> list[str]:
    """Pull just the user-role messages from a simulation transcript."""
    return [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "user" and msg.get("content")
    ]


def extract_agent_messages(messages: list[dict]) -> list[str]:
    """Pull just the assistant-role messages from a simulation transcript."""
    return [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("content")
    ]


def compute_goal_achievement(reward: float) -> float:
    """Goal achievement: direct from tau-bench reward.

    Binary success → 7 (full success) or 1 (failure).
    Partial rewards map linearly: 0.0→1, 0.5→4, 1.0→7.
    """
    return 1.0 + reward * 6.0


def compute_cognitive_effort(messages: list[dict]) -> float:
    """Cognitive effort proxy: how much work did the user have to do?

    Counts friction events from the transcript:
    - User re-explaining info already provided
    - Agent requesting information
    - User correcting the agent
    - Total message count as baseline effort

    Maps to 1-7 scale (1 = effortless, 7 = exhausting).
    """
    user_msgs = extract_user_messages(messages)
    agent_msgs = extract_agent_messages(messages)

    if not user_msgs:
        return 1.0

    # Count friction signals
    info_requests = 0
    for msg in agent_msgs:
        lower = msg.lower()
        if any(p in lower for p in [
            "could you provide", "can you provide", "could you share",
            "what is your", "what's your", "may i have",
            "can you confirm", "could you confirm", "please provide",
            "i need your", "i'll need", "do you have your",
        ]):
            info_requests += 1

    user_corrections = 0
    for msg in user_msgs:
        lower = msg.lower()
        if any(p in lower for p in [
            "no, i", "no i", "that's not", "thats not",
            "i already", "i said", "i told you", "i meant",
            "like i said", "as i said", "i just said",
            "wrong", "incorrect", "not what i",
        ]):
            user_corrections += 1

    user_repeats = 0
    seen_info = set()
    for msg in user_msgs:
        tokens = set(msg.lower().split())
        overlap = len(tokens & seen_info)
        if overlap > 5 and overlap > len(tokens) * 0.4:
            user_repeats += 1
        seen_info.update(tokens)

    # Composite score
    friction_score = (
        info_requests * 0.5
        + user_corrections * 1.0
        + user_repeats * 0.7
        + len(user_msgs) * 0.15  # baseline: more messages = more effort
    )

    # Map to 1-7: low friction → 1, high friction → 7
    # Calibration: 0 friction events ≈ 1, 10+ events ≈ 7
    effort = 1.0 + min(friction_score, 10.0) * 0.6
    return min(7.0, max(1.0, effort))


def compute_intent_alignment(messages: list[dict], reward: float) -> float:
    """Intent alignment proxy: how quickly did agent get on the right track?

    Counts misalignment events:
    - User redirect/correction messages
    - Agent apologizing for misunderstanding
    - Number of turns before productive action

    Maps to 1-7 scale (7 = perfect alignment, 1 = constant misalignment).
    """
    user_msgs = extract_user_messages(messages)
    agent_msgs = extract_agent_messages(messages)

    if not user_msgs:
        return 7.0 if reward > 0 else 1.0

    # Count redirect events from user
    redirects = 0
    for msg in user_msgs:
        lower = msg.lower()
        if any(p in lower for p in [
            "no, i", "that's not what", "thats not what",
            "i meant", "not what i asked", "wrong",
            "i want to", "can you just", "please just",
            "you're not understanding", "you misunderstood",
        ]):
            redirects += 1

    # Count agent confusion signals
    agent_confusion = 0
    for msg in agent_msgs:
        lower = msg.lower()
        if any(p in lower for p in [
            "i apologize for the confusion", "sorry for the misunderstanding",
            "let me clarify", "i misunderstood", "sorry about that",
        ]):
            agent_confusion += 1

    misalignment_events = redirects + agent_confusion * 0.5

    # Factor in outcome: successful task with no redirects = perfect alignment
    if reward >= 1.0 and misalignment_events == 0:
        return 7.0

    # Map: 0 events → 7, 5+ events → 1
    alignment = 7.0 - min(misalignment_events, 5.0) * 1.2
    # Penalize failure
    if reward < 1.0:
        alignment = min(alignment, 4.0)

    return min(7.0, max(1.0, alignment))


# ═══════════════════════════════════════════════════════════════════════════
# 3. CALIBRATED LLM JUDGE (Trust + Use Again — k=3 multi-call)
# ═══════════════════════════════════════════════════════════════════════════

TRUST_JUDGE_PROMPT = """You are evaluating a customer service interaction from the perspective of a specific customer persona.

=== CALIBRATION EXAMPLES ===

Example 1 (Trust = 2/7): Agent gave incorrect information, dismissed concerns, and didn't verify before acting. The customer felt their issue wasn't taken seriously.

Example 2 (Trust = 4/7): Agent was technically competent and resolved the issue, but was robotic, didn't acknowledge frustration, and felt like talking to a script.

Example 3 (Trust = 6/7): Agent was warm, competent, verified before acting, acknowledged the customer's specific situation, and explained their reasoning clearly.

=== PERSONA BEHAVIORAL SPEC ===
{persona_spec}

=== TRANSCRIPT ===
{transcript}

=== INSTRUCTIONS ===
Rate how much THIS specific persona would trust the agent, on a 1-7 scale.
Consider what THIS persona specifically needs to feel safe:
- A high-agency persona needs to feel in control
- A wary/adversarial persona needs transparency and honesty
- An L2 English speaker needs plain, jargon-free language
- A low-bandwidth persona needs efficiency, not warmth

Output ONLY valid JSON:
{{"trust": [integer 1-7], "evidence": "one sentence from the persona's perspective"}}"""


USE_AGAIN_JUDGE_PROMPT = """You are evaluating a customer service interaction from the perspective of a specific customer persona.

=== PERSONA BEHAVIORAL SPEC ===
{persona_spec}

=== TRANSCRIPT ===
{transcript}

=== INSTRUCTIONS ===
Would THIS specific persona use this service again?
Consider their overall experience: was their goal met? Was the effort reasonable for them? Did the interaction match what they need?

A persona with low stakes and easy resolution → likely yes.
A persona who was frustrated and barely got help → likely no.
A persona whose bandwidth was exhausted by the process → likely no even if successful.

Output ONLY valid JSON:
{{"use_again": [0 or 1], "evidence": "one sentence from the persona's perspective"}}"""


def judge_trust_calibrated(
    messages: list[dict],
    persona_spec: str,
    llm_call: Callable[[str], str],
    k: int = 3,
) -> dict:
    """Multi-call calibrated trust judgment. Returns median of k calls.

    Args:
        messages: Full simulation transcript.
        persona_spec: The persona's assembled prompt or YAML.
        llm_call: Function that takes a prompt string and returns response string.
        k: Number of independent judge calls.

    Returns:
        {"trust": median_score, "all_scores": [s1, s2, ...], "agreement": pct}
    """
    transcript = _format_transcript(messages)
    prompt = TRUST_JUDGE_PROMPT.format(
        persona_spec=persona_spec[:3000],  # Truncate to avoid token limits
        transcript=transcript[:4000],
    )

    scores = []
    for _ in range(k):
        response = llm_call(prompt)
        parsed = _parse_json_response(response)
        score = parsed.get("trust", 0)
        if 1 <= score <= 7:
            scores.append(score)

    if not scores:
        return {"trust": 0, "all_scores": [], "agreement": 0.0}

    median_score = sorted(scores)[len(scores) // 2]
    agreement = _simple_agreement(scores)

    return {
        "trust": median_score,
        "all_scores": scores,
        "agreement": agreement,
    }


def judge_use_again(
    messages: list[dict],
    persona_spec: str,
    llm_call: Callable[[str], str],
    k: int = 3,
) -> dict:
    """Multi-call use-again judgment. Returns majority vote of k calls.

    Returns:
        {"use_again": 0 or 1, "all_votes": [v1, v2, ...], "agreement": pct}
    """
    transcript = _format_transcript(messages)
    prompt = USE_AGAIN_JUDGE_PROMPT.format(
        persona_spec=persona_spec[:3000],
        transcript=transcript[:4000],
    )

    votes = []
    for _ in range(k):
        response = llm_call(prompt)
        parsed = _parse_json_response(response)
        vote = parsed.get("use_again")
        if vote in (0, 1):
            votes.append(vote)

    if not votes:
        return {"use_again": 0, "all_votes": [], "agreement": 0.0}

    majority = 1 if sum(votes) > len(votes) / 2 else 0
    agreement = _simple_agreement(votes)

    return {
        "use_again": majority,
        "all_votes": votes,
        "agreement": agreement,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. ROBUSTNESS METRICS (Paper Section 2.3)
# ═══════════════════════════════════════════════════════════════════════════

def compute_robustness_metrics(per_persona_rewards: dict[str, list[float]]) -> dict:
    """Compute persona-conditioned robustness metrics (Section 2.3).

    Args:
        per_persona_rewards: {persona_id: [reward_trial1, reward_trial2, ...]}

    Returns:
        Dict with mean, variance, tail risk, persona gap.
    """
    if not per_persona_rewards:
        return {}

    # Mean reward per persona
    persona_means = {
        p: _mean(rs) for p, rs in per_persona_rewards.items()
    }
    all_means = list(persona_means.values())

    if not all_means:
        return {}

    sorted_means = sorted(all_means)
    n = len(sorted_means)

    result = {
        "mean_performance": _mean(all_means),
        "cross_persona_variance": _variance(all_means),
        "worst_case": min(all_means),
        "best_case": max(all_means),
        "persona_gap": max(all_means) - min(all_means),
        "n_personas": n,
    }

    # Tail risk: worst-α percentile
    if n >= 4:
        idx_10 = max(0, int(n * 0.10) - 1)
        idx_25 = max(0, int(n * 0.25) - 1)
        result["tail_risk_10"] = sorted_means[idx_10]
        result["tail_risk_25"] = sorted_means[idx_25]
    else:
        result["tail_risk_10"] = sorted_means[0]
        result["tail_risk_25"] = sorted_means[0]

    # Per-persona breakdown
    result["per_persona"] = persona_means

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. TRAIT SENSITIVITY (Paper Section 2.3 — core contribution)
# ═══════════════════════════════════════════════════════════════════════════

def compute_trait_sensitivity(
    per_persona_rewards: dict[str, float],
    persona_codes: dict[str, dict[str, str]],
) -> dict[str, dict[str, dict]]:
    """Compute E[M(A,p) | t_j = v] for each dimension and value.

    Args:
        per_persona_rewards: {persona_id: mean_reward}
        persona_codes: {persona_id: {"agency": "3A", "bandwidth": "B3", ...}}

    Returns:
        {dimension: {code: {"mean": float, "ci_lo": float, "ci_hi": float, "n": int, "rewards": [...]}}}
    """
    # Collect all dimension names from any persona
    all_dims = set()
    for codes in persona_codes.values():
        all_dims.update(codes.keys())

    result = {}

    for dim in sorted(all_dims):
        # Group rewards by this dimension's value
        code_rewards = defaultdict(list)
        for pid, codes in persona_codes.items():
            if dim in codes and pid in per_persona_rewards:
                code = codes[dim]
                code_rewards[code].append(per_persona_rewards[pid])

        dim_result = {}
        for code in sorted(code_rewards.keys()):
            rewards = code_rewards[code]
            mean_r = _mean(rewards)
            ci_lo, ci_hi = bootstrap_ci(rewards) if len(rewards) >= 3 else (mean_r, mean_r)
            dim_result[code] = {
                "mean": mean_r,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": len(rewards),
                "rewards": rewards,
            }
        result[dim] = dim_result

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. BOOTSTRAP CONFIDENCE INTERVALS (Paper Section 2.5)
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for mean.

    Returns (ci_lower, ci_upper) at the (1-alpha) confidence level.
    """
    if len(values) < 2:
        m = _mean(values)
        return (m, m)

    rng = random.Random(seed)
    n = len(values)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(_mean(sample))

    boot_means.sort()
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    return (boot_means[lo_idx], boot_means[hi_idx])


# ═══════════════════════════════════════════════════════════════════════════
# 7. TABLE 2 AGGREGATION (Hybrid: proxies + LLM judge)
# ═══════════════════════════════════════════════════════════════════════════

def compute_table2_metrics(task_metrics_list: list[TaskMetrics]) -> dict:
    """Aggregate Table 2 metrics across all simulations.

    3 objective (transcript-derived): goal_achievement, cognitive_effort, intent_alignment
    2 subjective (LLM judge): trust, use_again

    Returns dict with means and bootstrap CIs.
    """
    if not task_metrics_list:
        return {}

    ga = [m.goal_achievement for m in task_metrics_list]
    ce = [m.cognitive_effort for m in task_metrics_list]
    ia = [m.intent_alignment for m in task_metrics_list]
    tr = [m.trust for m in task_metrics_list if m.trust > 0]
    ua = [m.use_again for m in task_metrics_list if m.trust > 0]  # only if judge ran

    result = {
        "goal_achievement": _mean(ga),
        "cognitive_effort": _mean(ce),
        "intent_alignment": _mean(ia),
    }

    if tr:
        result["trust"] = _mean(tr)
    if ua:
        result["use_again"] = _mean(ua)

    # Bootstrap CIs for each metric
    for name, vals in [("goal_achievement", ga), ("cognitive_effort", ce),
                       ("intent_alignment", ia), ("trust", tr), ("use_again", ua)]:
        if len(vals) >= 3:
            lo, hi = bootstrap_ci(vals)
            result[f"{name}_ci"] = (lo, hi)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. JUDGE RELIABILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_judge_agreement(all_judge_results: list[dict]) -> dict:
    """Compute inter-judge agreement across k=3 calls.

    Args:
        all_judge_results: List of dicts with "all_scores" or "all_votes" keys.

    Returns:
        {"mean_agreement": float, "n_judgments": int}
    """
    agreements = []
    for result in all_judge_results:
        scores = result.get("all_scores") or result.get("all_votes") or []
        if len(scores) >= 2:
            agreements.append(_simple_agreement(scores))

    if not agreements:
        return {"mean_agreement": 0.0, "n_judgments": 0}

    return {
        "mean_agreement": _mean(agreements),
        "n_judgments": len(agreements),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. COMBINED EVALUATION (wires everything together)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionReport:
    """Full evaluation report for one experimental condition (one agent)."""
    condition: str
    task_metrics: dict               # aggregate_task_metrics output
    robustness: dict = field(default_factory=dict)       # compute_robustness_metrics
    trait_sensitivity: dict = field(default_factory=dict) # compute_trait_sensitivity
    table2: dict = field(default_factory=dict)            # compute_table2_metrics
    judge_reliability: dict = field(default_factory=dict) # compute_judge_agreement
    n_simulations: int = 0
    task_metrics_list: list = field(default_factory=list)  # raw list[TaskMetrics] for analyze.py


def evaluate_condition(
    simulations: list[dict],
    condition: str,
    persona_ids: Optional[list[str]] = None,
    persona_codes_list: Optional[list[dict]] = None,
) -> ConditionReport:
    """Evaluate all simulations for one condition.

    Args:
        simulations: List of SimulationRun dicts from tau-bench.
        condition: "layer", "none", etc.
        persona_ids: Optional persona ID per simulation.
        persona_codes_list: Optional list of {dim: code} dicts per simulation.
    """
    persona_ids = persona_ids or [
        sim.get("_persona_id", f"sim_{i}") for i, sim in enumerate(simulations)
    ]
    persona_codes_list = persona_codes_list or [
        sim.get("_persona_codes", {}) for sim in simulations
    ]

    # Extract per-simulation metrics
    task_mets = []
    for sim, pid, pcodes in zip(simulations, persona_ids, persona_codes_list):
        tm = extract_task_metrics(sim, condition=condition, persona_id=pid, persona_codes=pcodes)
        task_mets.append(tm)

    agg_task = aggregate_task_metrics(task_mets)

    # Group rewards by persona for robustness metrics
    per_persona_rewards = defaultdict(list)
    per_persona_codes = {}
    for tm in task_mets:
        pid = tm.persona_id or "unknown"
        per_persona_rewards[pid].append(tm.reward)
        if tm.persona_codes:
            per_persona_codes[pid] = tm.persona_codes

    robustness = compute_robustness_metrics(dict(per_persona_rewards))

    # Trait sensitivity (only if we have persona codes)
    trait_sens = {}
    if per_persona_codes and robustness.get("per_persona"):
        trait_sens = compute_trait_sensitivity(
            robustness["per_persona"],
            per_persona_codes,
        )

    # Table 2 metrics
    table2 = compute_table2_metrics(task_mets)

    return ConditionReport(
        condition=condition,
        task_metrics=agg_task,
        robustness=robustness,
        trait_sensitivity=trait_sens,
        table2=table2,
        n_simulations=len(simulations),
        task_metrics_list=task_mets,
    )


def compare_conditions(reports: list[ConditionReport]) -> dict:
    """Compare evaluation reports across conditions/agents.

    Returns a summary dict suitable for paper tables.
    """
    summary = {}
    for report in reports:
        entry = {
            # Table 1
            "task_success_rate": report.task_metrics.get("task_success_rate", 0.0),
            "mean_messages": report.task_metrics.get("mean_messages", 0),
            "n_simulations": report.n_simulations,
            # Table 2
            **{f"table2_{k}": v for k, v in report.table2.items() if not k.endswith("_ci")},
            # Robustness
            **{f"robustness_{k}": v for k, v in report.robustness.items()
               if k not in ("per_persona",)},
        }
        summary[report.condition] = entry

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL QA METRICS (not in paper, kept for development)
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that real people almost never use in chat
LLM_TELLTALE_PATTERNS = [
    (r"(?i)^hello,?\s+i'?m reaching out", "formal_opening"),
    (r"(?i)i would like to (express|convey|communicate)", "formal_expression"),
    (r"(?i)as a (customer|long-?time user|loyal)", "self_labeling"),
    (r"(?i)\b(utilize|prior to|furthermore|regarding|henceforth)\b", "corporate_vocabulary"),
    (r"(?i)i understand your? (concern|position|perspective),?\s*(however|but)", "agent_speak_leak"),
    (r"(?i)^(dear|greetings|good (morning|afternoon|evening))", "formal_greeting"),
    (r"(?i)thank you for your (patience|understanding|assistance|time)", "formal_thanks"),
    (r"(?i)i appreciate your (help|effort|prompt)", "formal_appreciation"),
    (r"^\d+\.\s", "numbered_list"),
    (r"(?i)(in conclusion|to summarize|in summary)", "summary_language"),
    (r"(?i)i (hereby|formally|respectfully) (request|demand|ask)", "formal_request"),
]

HUMAN_POSITIVE_PATTERNS = [
    (r"(?i)\b(um|uh|hmm|haha|lol|idk|tbh|nvm|bc|rn)\b", "filler_words"),
    (r"(?i)\b(ok|okay|yeah|yep|nope|nah)\b", "casual_affirmatives"),
    (r"[a-z]{2,}\?\?", "repeated_punctuation"),
    (r"(?i)^(hey|hi|so|ok so)", "casual_opening"),
    (r"\.\.\.", "ellipsis"),
    (r"(?i)(i think|maybe|probably|i guess|not sure)", "hedging"),
    (r"[a-z]$", "no_end_punctuation"),
]


@dataclass
class HumannessScore:
    """Humanness assessment for a set of user messages (internal QA only)."""
    llm_pattern_count: int = 0
    llm_patterns_found: list[str] = field(default_factory=list)
    human_pattern_count: int = 0
    human_patterns_found: list[str] = field(default_factory=list)
    total_messages: int = 0
    score: float = 0.0


def score_humanness(user_messages: list[str]) -> HumannessScore:
    """Score how human-like a set of user messages are (internal QA)."""
    if not user_messages:
        return HumannessScore()

    all_text = "\n".join(user_messages)
    llm_found = [name for pat, name in LLM_TELLTALE_PATTERNS if re.search(pat, all_text)]
    human_found = [name for pat, name in HUMAN_POSITIVE_PATTERNS if re.search(pat, all_text)]

    llm_penalty = len(llm_found) * 0.15
    human_bonus = len(human_found) * 0.08
    score = max(0.0, min(1.0, 0.5 + human_bonus - llm_penalty))

    return HumannessScore(
        llm_pattern_count=len(llm_found),
        llm_patterns_found=llm_found,
        human_pattern_count=len(human_found),
        human_patterns_found=human_found,
        total_messages=len(user_messages),
        score=score,
    )


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _mean(values: list) -> float:
    """Safe mean that handles empty lists."""
    return sum(values) / len(values) if values else 0.0


def _variance(values: list) -> float:
    """Population variance."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def _format_transcript(messages: list[dict]) -> str:
    """Format messages into a readable transcript string."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _simple_agreement(values: list) -> float:
    """Simple agreement: fraction of values matching the mode."""
    if not values:
        return 0.0
    mode = max(set(values), key=values.count)
    return values.count(mode) / len(values)
