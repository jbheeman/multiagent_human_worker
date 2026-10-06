"""Full-vector Schwartz alignment (pure-Python, dependency-free for unit testing).

Replaces the old dominant-trait-only thresholding. Compares a target Schwartz
vector ([0,1]) against PVQ-measured trait scores ([1,6]) across ALL 10 traits.

Why mean-centering is load-bearing: PVQ scores are ipsative/acquiescence-biased
(respondents rate most items high), so raw magnitudes look uniformly large. Only
the *profile shape* is meaningful. Centering each vector by its own mean removes
that level bias; cosine on the centered vectors is equivalent to Pearson
correlation across traits. Spearman is offered as a rank-based alternative.
"""

from __future__ import annotations

import math

SCHWARTZ_KEYS = [
    "POWER", "ACHIEVEMENT", "HEDONISM", "STIMULATION", "SELF_DIRECTION",
    "UNIVERSALISM", "BENEVOLENCE", "TRADITION", "CONFORMITY", "SECURITY",
]


def _common_keys(target, pvq):
    return [k for k in SCHWARTZ_KEYS if k in target and k in pvq]


def _center(xs):
    m = sum(xs) / len(xs)
    return [x - m for x in xs]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _rank(xs):
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def schwartz_alignment(target_vector: dict, pvq_results: dict,
                       metric: str = "cosine") -> float:
    """Return alignment in [0,1]; 0.5 means undefined/neutral (degenerate input)."""
    if not target_vector or not pvq_results:
        return 0.5
    keys = _common_keys(target_vector, pvq_results)
    if len(keys) < 2:
        return 0.5

    target = [float(target_vector[k]) for k in keys]          # already [0,1]
    pvq01 = [(float(pvq_results[k]) - 1.0) / 5.0 for k in keys]  # [1,6] -> [0,1]

    if metric == "spearman":
        sim = _cosine(_center(_rank(target)), _center(_rank(pvq01)))
    else:  # cosine on mean-centered vectors (== Pearson across traits)
        sim = _cosine(_center(target), _center(pvq01))

    return (sim + 1.0) / 2.0
