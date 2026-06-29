"""PVQ-40 administration and Schwartz scoring helpers.

The assignment path should administer the Portrait Values Questionnaire and
score the responses. It should not ask a model to invent a direct 10-number
Schwartz profile.
"""

from __future__ import annotations

SCHWARTZ_KEYS = [
    "POWER",
    "ACHIEVEMENT",
    "HEDONISM",
    "STIMULATION",
    "SELF_DIRECTION",
    "UNIVERSALISM",
    "BENEVOLENCE",
    "TRADITION",
    "CONFORMITY",
    "SECURITY",
]

PVQ_ITEMS = {
    1: "Thinking up new ideas and being creative is important to him. He likes to do things in his own original way.",
    2: "It is important to him to be rich. He wants to have a lot of money and expensive things.",
    3: "He thinks it is important that every person in the world be treated equally. He believes everyone should have equal opportunities in life.",
    4: "It is very important to him to show his abilities. He wants people to admire what he does.",
    5: "It is important to him to live in secure surroundings. He avoids anything that might endanger his safety.",
    6: "He thinks it is important to do lots of different things in life. He always looks for new things to try.",
    7: "He believes that people should do what they are told. He thinks people should follow rules at all times, even when no one is watching.",
    8: "It is important to him to listen to people who are different from him. Even when he disagrees with them, he still wants to understand them.",
    9: "He thinks it is important not to ask for more than what you have. He believes that people should be satisfied with what they have.",
    10: "He seeks every chance he can to have fun. It is important to him to do things that give him pleasure.",
    11: "It is important to him to make his own decisions about what he does. He likes to be free to plan and choose his activities for himself.",
    12: "It is very important to him to help the people around him. He wants to care for their well-being.",
    13: "Being very successful is important to him. He likes to impress other people.",
    14: "It is very important to him that his country be safe. He thinks the state must be on watch against threats from within and without.",
    15: "He likes to take risks. He is always looking for adventures.",
    16: "It is important to him to always behave properly. He wants to avoid doing anything people would say is wrong.",
    17: "It is important to him to be in charge and tell others what to do. He wants people to do what he says.",
    18: "It is important to him to be loyal to his friends. He wants to devote himself to people close to him.",
    19: "He strongly believes that people should care for nature. Looking after the environment is important to him.",
    20: "Religious belief is important to him. He tries hard to do what his religion requires.",
    21: "It is important to him that things be organized and clean. He really does not like things to be a mess.",
    22: "He thinks it is important to be interested in things. He likes to be curious and to try to understand all sorts of things.",
    23: "He believes all the world's people should live in harmony. Promoting peace among all groups in the world is important to him.",
    24: "He thinks it is important to be ambitious. He wants to show how capable he is.",
    25: "He thinks it is best to do things in traditional ways. It is important to him to keep up the customs he has learned.",
    26: "Enjoying life's pleasures is important to him. He likes to spoil himself.",
    27: "It is important to him to respond to the needs of others. He tries to support those he knows.",
    28: "He believes he should always show respect to his parents and older people. It is important to him to be obedient.",
    29: "He wants everyone to be treated justly, even people he does not know. It is important to him to protect the weak in society.",
    30: "He likes surprises. It is important to him to have an exciting life.",
    31: "He tries hard to avoid getting sick. Staying healthy is very important to him.",
    32: "Getting ahead in life is important to him. He strives to do better than others.",
    33: "Forgiving people who have hurt him is important to him. He tries to see what is good in them and not to hold a grudge.",
    34: "It is important to him to be independent. He likes to rely on himself.",
    35: "Having a stable government is important to him. He is concerned that the social order be protected.",
    36: "It is important to him to be polite to other people all the time. He tries never to disturb or irritate others.",
    37: "He really wants to enjoy life. Having a good time is very important to him.",
    38: "It is important to him to be humble and modest. He tries not to draw attention to himself.",
    39: "He always wants to be the one who makes the decisions. He likes to be the leader.",
    40: "It is important to him to adapt to nature and to fit into it. He believes that people should not change nature.",
}

PVQ_MAPPING = {
    "UNIVERSALISM": [3, 8, 19, 23, 29, 40],
    "BENEVOLENCE": [12, 18, 27, 33],
    "TRADITION": [9, 20, 25, 38],
    "CONFORMITY": [7, 16, 28, 36],
    "SECURITY": [5, 14, 21, 31, 35],
    "POWER": [2, 17, 39],
    "ACHIEVEMENT": [4, 13, 24, 32],
    "HEDONISM": [10, 26, 37],
    "STIMULATION": [6, 15, 30],
    "SELF_DIRECTION": [1, 11, 22, 34],
}

PVQ_DATA = {
    "items": PVQ_ITEMS,
    "mapping": PVQ_MAPPING,
}


def pvq_survey_text() -> str:
    return "\n".join(f"{idx}. {text}" for idx, text in PVQ_ITEMS.items())


def validate_pvq_item_scores(item_scores: dict) -> dict[str, int]:
    """Return canonical string-keyed item scores or raise ValueError."""
    canonical: dict[str, int] = {}
    missing = []
    for idx in PVQ_ITEMS:
        raw = item_scores.get(str(idx), item_scores.get(idx))
        if raw is None:
            missing.append(idx)
            continue
        try:
            val = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PVQ item {idx} is not an integer: {raw!r}") from exc
        if val < 1 or val > 6:
            raise ValueError(f"PVQ item {idx} score {val} is outside 1..6")
        canonical[str(idx)] = val
    if missing:
        raise ValueError(f"PVQ item scores missing items: {missing}")
    return canonical


def score_pvq_value_means(item_scores: dict) -> dict[str, float]:
    canonical = validate_pvq_item_scores(item_scores)
    return {
        trait: sum(canonical[str(idx)] for idx in item_ids) / len(item_ids)
        for trait, item_ids in PVQ_MAPPING.items()
    }


def normalize_value_means(value_means: dict) -> dict[str, float]:
    """Convert PVQ trait means from [1,6] to [0,1] in canonical key order."""
    normalized = {}
    for key in SCHWARTZ_KEYS:
        if key not in value_means:
            raise ValueError(f"PVQ value means missing {key}")
        normalized[key] = max(0.0, min(1.0, (float(value_means[key]) - 1.0) / 5.0))
    return normalized


def score_pvq_assignment(item_scores: dict) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    canonical_items = validate_pvq_item_scores(item_scores)
    value_means = score_pvq_value_means(canonical_items)
    target_vector = normalize_value_means(value_means)
    return canonical_items, value_means, target_vector
