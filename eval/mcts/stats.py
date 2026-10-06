"""Paired statistics for MCTS rollouts (mcts.md §5).

Comparisons are paired on (task_id, persona_id, block).

With a single model (current phase-1 panel), model-pair / rank-shift
sections are computed but return empty results — arm contrasts and
point estimates are the actionable outputs. Satisfaction metrics are
included when ``satisfaction_final`` is present; otherwise they are omitted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Sequence

from eval.mcts.assignment import read_assignment_csv

PairKey = tuple[str, str, int]  # task_id, persona_id, block

PRIMARY_METRICS = ("task_success", "transfer")
SAT_METRICS = ("satisfaction_final",)
DEFAULT_ARM_ORDER = ("fixed_prompt", "value_only", "behavior_only", "full_gepa")


@dataclass(frozen=True)
class Estimate:
    mean: float
    ci_low: float
    ci_high: float
    n: int
    cluster: str  # which bootstrap variant produced the wider CI
    n_boot: int


@dataclass(frozen=True)
class PairedContrast:
    metric: str
    arm_a: str
    arm_b: str
    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    sign_test_p: float
    n_pos: int
    n_neg: int
    n_zero: int


def load_rollouts(
    path: Path,
    *,
    domain: str | None = None,
    exclude_sim_error: bool = False,
    exclude_dry_run: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if exclude_dry_run and rec.get("dry_run"):
                continue
            if domain is not None and rec.get("domain") != domain:
                continue
            if exclude_sim_error and rec.get("terminal_state") == "sim_error":
                continue
            rows.append(rec)
    return rows


def pair_key(rec: dict[str, Any]) -> PairKey:
    return (str(rec["task_id"]), str(rec["persona_id"]), int(rec.get("block") or 1))


def metric_value(rec: dict[str, Any], metric: str) -> float | None:
    if metric == "task_success":
        return 1.0 if rec.get("task_success") else 0.0
    if metric == "transfer":
        return 1.0 if rec.get("transfer") else 0.0
    if metric == "satisfaction_final":
        v = rec.get("satisfaction_final")
        return float(v) if v is not None else None
    raise KeyError(metric)


def available_metrics(rows: Sequence[dict[str, Any]]) -> list[str]:
    metrics = list(PRIMARY_METRICS)
    if any(r.get("satisfaction_final") is not None for r in rows):
        metrics.extend(SAT_METRICS)
    return metrics


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation percentile; q in [0, 100]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        cand = min(1.0, p_values[idx] * factor)
        running = max(running, cand)
        adjusted[idx] = running
    return adjusted


def sign_test_pvalue(deltas: Sequence[float]) -> tuple[float, int, int, int]:
    """Two-sided exact sign test (binomial), ignoring zeros."""
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = len(deltas) - n_pos - n_neg
    n = n_pos + n_neg
    if n == 0:
        return 1.0, n_pos, n_neg, n_zero
    k = min(n_pos, n_neg)
    # P(X <= k) * 2 under Bin(n, 0.5), capped at 1.
    # Compute via recursive binomial probs for stability at modest n.
    # C(n,i) / 2^n
    log_half_n = n * math.log(0.5)
    cdf = 0.0
    log_c = 0.0  # log C(n,0)
    for i in range(k + 1):
        cdf += math.exp(log_c + log_half_n)
        # C(n,i+1) = C(n,i) * (n-i)/(i+1)
        if i < n:
            log_c += math.log(n - i) - math.log(i + 1)
    p = min(1.0, 2.0 * cdf)
    return p, n_pos, n_neg, n_zero


def _group_values(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    cluster_key: Callable[[dict[str, Any]], str],
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        val = metric_value(rec, metric)
        if val is None:
            continue
        groups[cluster_key(rec)].append(val)
    return dict(groups)


def cluster_bootstrap_mean(
    groups: dict[str, list[float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    """Cluster bootstrap mean. Returns (mean, ci_low, ci_high, n_obs)."""
    if not groups:
        return float("nan"), float("nan"), float("nan"), 0
    keys = list(groups.keys())
    all_vals = [v for vals in groups.values() for v in vals]
    n_obs = len(all_vals)
    point = sum(all_vals) / n_obs
    rng = Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        drawn = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        pooled: list[float] = []
        for k in drawn:
            pooled.extend(groups[k])
        if not pooled:
            continue
        boots.append(sum(pooled) / len(pooled))
    boots.sort()
    lo = _percentile(boots, 100.0 * (alpha / 2.0))
    hi = _percentile(boots, 100.0 * (1.0 - alpha / 2.0))
    return point, lo, hi, n_obs


def wider_cluster_ci(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """Persona-cluster and task-cluster bootstrap; report the wider interval."""
    by_persona = _group_values(rows, metric=metric, cluster_key=lambda r: str(r["persona_id"]))
    by_task = _group_values(rows, metric=metric, cluster_key=lambda r: str(r["task_id"]))
    m_p, lo_p, hi_p, n = cluster_bootstrap_mean(by_persona, n_boot=n_boot, seed=seed)
    m_t, lo_t, hi_t, _ = cluster_bootstrap_mean(by_task, n_boot=n_boot, seed=seed + 1)
    width_p = hi_p - lo_p if all(map(math.isfinite, (lo_p, hi_p))) else -1.0
    width_t = hi_t - lo_t if all(map(math.isfinite, (lo_t, hi_t))) else -1.0
    if width_t > width_p:
        return Estimate(m_t, lo_t, hi_t, n, "task_id", n_boot)
    return Estimate(m_p, lo_p, hi_p, n, "persona_id", n_boot)


def percentile_metric(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "satisfaction_final",
    q: float = 10.0,
) -> float | None:
    vals = [v for r in rows if (v := metric_value(r, metric)) is not None]
    if not vals:
        return None
    vals.sort()
    return _percentile(vals, q)


def point_estimates(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per (model, arm, domain, metric) point estimate + wider cluster CI."""
    metrics = list(metrics or available_metrics(rows))
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        buckets[(str(rec["model"]), str(rec["arm"]), str(rec["domain"]))].append(rec)

    out: list[dict[str, Any]] = []
    for (model, arm, domain), subset in sorted(buckets.items()):
        for metric in metrics:
            if metric in SAT_METRICS and all(metric_value(r, metric) is None for r in subset):
                continue
            est = wider_cluster_ci(subset, metric=metric, n_boot=n_boot, seed=seed)
            row: dict[str, Any] = {
                "model": model,
                "arm": arm,
                "domain": domain,
                "metric": metric,
                **asdict(est),
            }
            if metric == "satisfaction_final":
                row["p10"] = percentile_metric(subset, metric=metric, q=10.0)
                # CVaR@10%: mean of observations at or below empirical P10.
                vals = sorted(
                    v for r in subset if (v := metric_value(r, metric)) is not None
                )
                if vals and row["p10"] is not None:
                    tail = [v for v in vals if v <= row["p10"]]
                    row["cvar10"] = sum(tail) / len(tail) if tail else None
                else:
                    row["cvar10"] = None
            out.append(row)
    return out


def _index_by_pair(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[PairKey, dict[str, Any]]]:
    """(model, arm, domain) -> pair_key -> record (last write wins)."""
    out: dict[tuple[str, str, str], dict[PairKey, dict[str, Any]]] = defaultdict(dict)
    for rec in rows:
        key = (str(rec["model"]), str(rec["arm"]), str(rec["domain"]))
        out[key][pair_key(rec)] = rec
    return out


def paired_deltas(
    left: dict[PairKey, dict[str, Any]],
    right: dict[PairKey, dict[str, Any]],
    *,
    metric: str,
) -> list[float]:
    deltas: list[float] = []
    for key in sorted(set(left) & set(right)):
        a = metric_value(left[key], metric)
        b = metric_value(right[key], metric)
        if a is None or b is None:
            continue
        deltas.append(b - a)
    return deltas


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    mean = sum(deltas) / len(deltas)
    rng = Random(seed)
    boots: list[float] = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return mean, _percentile(boots, 100.0 * (alpha / 2.0)), _percentile(
        boots, 100.0 * (1.0 - alpha / 2.0)
    )


def arm_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
    baseline: str = "fixed_prompt",
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """E2: within-persona paired deltas across arms (same model/domain)."""
    metrics = list(metrics or available_metrics(rows))
    indexed = _index_by_pair(rows)
    models = sorted({r["model"] for r in rows})
    domains = sorted({r["domain"] for r in rows})
    arm_set = {
        arm
        for (_, arm, _) in indexed
    }
    if arms is None:
        ordered = [a for a in DEFAULT_ARM_ORDER if a in arm_set]
        ordered.extend(sorted(arm_set - set(ordered)))
        arms = ordered

    out: list[dict[str, Any]] = []
    for model in models:
        for domain in domains:
            base_map = indexed.get((model, baseline, domain), {})
            if not base_map:
                # Fall back: pairwise among available arms without a fixed baseline.
                present = [a for a in arms if (model, a, domain) in indexed]
                pairs = [
                    (present[i], present[j])
                    for i in range(len(present))
                    for j in range(i + 1, len(present))
                ]
            else:
                pairs = [(baseline, a) for a in arms if a != baseline and (model, a, domain) in indexed]

            for arm_a, arm_b in pairs:
                left = indexed[(model, arm_a, domain)]
                right = indexed[(model, arm_b, domain)]
                for metric in metrics:
                    deltas = paired_deltas(left, right, metric=metric)
                    if not deltas:
                        continue
                    mean, lo, hi = paired_bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
                    p, n_pos, n_neg, n_zero = sign_test_pvalue(deltas)
                    contrast = PairedContrast(
                        metric=metric,
                        arm_a=arm_a,
                        arm_b=arm_b,
                        n_pairs=len(deltas),
                        mean_delta=mean,
                        ci_low=lo,
                        ci_high=hi,
                        sign_test_p=p,
                        n_pos=n_pos,
                        n_neg=n_neg,
                        n_zero=n_zero,
                    )
                    out.append(
                        {
                            "model": model,
                            "domain": domain,
                            **asdict(contrast),
                        }
                    )
    return out


def pairwise_model_differences(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per arm, all model pairs: paired deltas + sign test + Holm within domain."""
    metrics = list(metrics or available_metrics(rows))
    indexed = _index_by_pair(rows)
    domains = sorted({r["domain"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return []

    raw: list[dict[str, Any]] = []
    for domain in domains:
        family: list[dict[str, Any]] = []
        for arm in arms:
            for i, m1 in enumerate(models):
                for m2 in models[i + 1 :]:
                    left = indexed.get((m1, arm, domain), {})
                    right = indexed.get((m2, arm, domain), {})
                    for metric in metrics:
                        deltas = paired_deltas(left, right, metric=metric)
                        if not deltas:
                            continue
                        mean, lo, hi = paired_bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
                        p, n_pos, n_neg, n_zero = sign_test_pvalue(deltas)
                        family.append(
                            {
                                "domain": domain,
                                "arm": arm,
                                "metric": metric,
                                "model_a": m1,
                                "model_b": m2,
                                "n_pairs": len(deltas),
                                "mean_delta": mean,
                                "ci_low": lo,
                                "ci_high": hi,
                                "sign_test_p": p,
                                "n_pos": n_pos,
                                "n_neg": n_neg,
                                "n_zero": n_zero,
                                "family": f"model_pairs::{domain}",
                            }
                        )
        ps = [f["sign_test_p"] for f in family]
        adj = holm_bonferroni(ps)
        for item, p_adj in zip(family, adj):
            item["sign_test_p_holm"] = p_adj
            raw.append(item)
    return raw


def _mean_by_model(
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    domain: str,
    metric: str,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        if rec.get("arm") != arm or rec.get("domain") != domain:
            continue
        val = metric_value(rec, metric)
        if val is None:
            continue
        buckets[str(rec["model"])].append(val)
    return {m: sum(vs) / len(vs) for m, vs in buckets.items() if vs}


def _ranking(means: dict[str, float]) -> list[str]:
    """Higher metric → better rank (rank 1 = best). Ties broken by model name."""
    return sorted(means.keys(), key=lambda m: (-means[m], m))


def kendall_tau(rank_a: Sequence[str], rank_b: Sequence[str]) -> float:
    """Kendall τ-a on a shared item set (pair concordance)."""
    items = [x for x in rank_a if x in set(rank_b)]
    if len(items) < 2:
        return float("nan")
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    conc = disc = 0
    for i, x in enumerate(items):
        for y in items[i + 1 :]:
            s_a = pos_a[x] - pos_a[y]
            s_b = pos_b[x] - pos_b[y]
            if s_a == 0 or s_b == 0:
                continue
            if (s_a > 0) == (s_b > 0):
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    return float("nan") if tot == 0 else (conc - disc) / tot


def rank_shift(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "task_success",
    baseline_arm: str = "fixed_prompt",
    conditioned_arms: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """E1: Kendall τ between fixed_prompt ranking and each persona-conditioned arm."""
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return []
    domains = sorted({r["domain"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    if conditioned_arms is None:
        conditioned_arms = [a for a in arms if a != baseline_arm]

    # Index observations by (model, arm, domain, persona) for bootstrap over personas.
    by_persona: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    personas_by_domain: dict[str, set[str]] = defaultdict(set)
    for rec in rows:
        val = metric_value(rec, metric)
        if val is None:
            continue
        domain = str(rec["domain"])
        personas_by_domain[domain].add(str(rec["persona_id"]))
        by_persona[(str(rec["model"]), str(rec["arm"]), domain, str(rec["persona_id"]))].append(
            val
        )

    out: list[dict[str, Any]] = []
    rng = Random(seed)
    for domain in domains:
        personas = sorted(personas_by_domain[domain])
        if len(personas) < 2:
            continue
        base_means = _mean_by_model(rows, arm=baseline_arm, domain=domain, metric=metric)
        base_rank = _ranking(base_means)
        for arm in conditioned_arms:
            cond_means = _mean_by_model(rows, arm=arm, domain=domain, metric=metric)
            shared = sorted(set(base_means) & set(cond_means))
            if len(shared) < 2:
                continue
            cond_rank = _ranking(cond_means)
            tau = kendall_tau(base_rank, cond_rank)
            # Bootstrap: resample personas, recompute both rankings.
            boots: list[float] = []
            for _ in range(n_boot):
                drawn = [personas[rng.randrange(len(personas))] for _ in range(len(personas))]
                bm: dict[str, list[float]] = defaultdict(list)
                cm: dict[str, list[float]] = defaultdict(list)
                for pid in drawn:
                    for model in shared:
                        bm[model].extend(
                            by_persona.get((model, baseline_arm, domain, pid), [])
                        )
                        cm[model].extend(by_persona.get((model, arm, domain, pid), []))
                if any(not bm[m] or not cm[m] for m in shared):
                    continue
                b_means = {m: sum(bm[m]) / len(bm[m]) for m in shared}
                c_means = {m: sum(cm[m]) / len(cm[m]) for m in shared}
                boots.append(kendall_tau(_ranking(b_means), _ranking(c_means)))
            boots = [b for b in boots if math.isfinite(b)]
            boots.sort()
            lo = _percentile(boots, 2.5) if boots else float("nan")
            hi = _percentile(boots, 97.5) if boots else float("nan")
            rank_changes = {
                m: base_rank.index(m) - cond_rank.index(m)
                for m in shared
                if m in base_rank and m in cond_rank
            }
            out.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "baseline_arm": baseline_arm,
                    "conditioned_arm": arm,
                    "kendall_tau": tau,
                    "ci_low": lo,
                    "ci_high": hi,
                    "baseline_ranking": base_rank,
                    "conditioned_ranking": cond_rank,
                    "rank_change": rank_changes,  # positive => improved (moved up)
                    "n_boot": n_boot,
                }
            )
    return out


def _persona_strata(assignment_csv: Path | None, domain: str) -> dict[str, str]:
    if assignment_csv is None or not assignment_csv.exists():
        return {}
    _, rows = read_assignment_csv(assignment_csv)
    return {r.persona_id: r.stratum for r in rows if r.domain == domain}


def _stratified_subsample(
    personas: Sequence[str],
    strata: dict[str, str],
    n: int,
    rng: Random,
) -> list[str]:
    """Subsample n personas approximating strata proportions."""
    if n >= len(personas):
        return list(personas)
    by_s: dict[str, list[str]] = defaultdict(list)
    for pid in personas:
        by_s[strata.get(pid, "UNKNOWN")].append(pid)
    for s in by_s:
        rng.shuffle(by_s[s])

    # Hamilton apportionment of n seats across strata.
    labels = sorted(by_s)
    total = len(personas)
    ideal = {s: n * len(by_s[s]) / total for s in labels}
    seats = {s: int(math.floor(ideal[s])) for s in labels}
    rem = n - sum(seats.values())
    frac_order = sorted(labels, key=lambda s: (-(ideal[s] - seats[s]), s))
    for s in frac_order:
        if rem <= 0:
            break
        if seats[s] < len(by_s[s]):
            seats[s] += 1
            rem -= 1
    # Top up if some strata were too small.
    chosen: list[str] = []
    for s in labels:
        chosen.extend(by_s[s][: seats[s]])
    if len(chosen) < n:
        leftover = [p for p in personas if p not in set(chosen)]
        rng.shuffle(leftover)
        chosen.extend(leftover[: n - len(chosen)])
    return chosen[:n]


def convergence_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "task_success",
    arm: str = "full_gepa",
    domain: str | None = None,
    ns: Sequence[int] = (10, 25, 50, 100, 150, 200),
    n_reps: int = 500,
    seed: int = 0,
    assignment_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Estimate spread vs n personas; P(ranking == full-panel) when ≥2 models."""
    domains = [domain] if domain else sorted({str(r["domain"]) for r in rows})
    out: list[dict[str, Any]] = []
    rng = Random(seed)

    for dom in domains:
        subset = [
            r
            for r in rows
            if r.get("domain") == dom and r.get("arm") == arm
        ]
        if not subset:
            continue
        strata = _persona_strata(assignment_csv, dom)
        personas = sorted({str(r["persona_id"]) for r in subset})
        models = sorted({str(r["model"]) for r in subset})
        full_means = _mean_by_model(subset, arm=arm, domain=dom, metric=metric)
        full_rank = _ranking(full_means) if len(models) >= 2 else []

        # Per-persona mean (average over tasks) for fast resampling.
        per_persona: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for rec in subset:
            val = metric_value(rec, metric)
            if val is None:
                continue
            per_persona[str(rec["persona_id"])][str(rec["model"])].append(val)

        for n in ns:
            if n > len(personas):
                continue
            estimates: list[float] = []
            rank_match = 0
            for _ in range(n_reps):
                drawn = _stratified_subsample(personas, strata, n, rng)
                # Overall mean across drawn personas/models (panel mean).
                vals: list[float] = []
                means: dict[str, list[float]] = defaultdict(list)
                for pid in drawn:
                    for model, vs in per_persona.get(pid, {}).items():
                        if not vs:
                            continue
                        m = sum(vs) / len(vs)
                        vals.append(m)
                        means[model].append(m)
                if not vals:
                    continue
                estimates.append(sum(vals) / len(vals))
                if len(models) >= 2 and full_rank:
                    sample_means = {
                        m: sum(ms) / len(ms) for m, ms in means.items() if ms
                    }
                    if len(sample_means) == len(full_means) and _ranking(sample_means) == full_rank:
                        rank_match += 1
            if not estimates:
                continue
            estimates.sort()
            out.append(
                {
                    "domain": dom,
                    "arm": arm,
                    "metric": metric,
                    "n_personas": n,
                    "n_reps": n_reps,
                    "mean": sum(estimates) / len(estimates),
                    "ci_low": _percentile(estimates, 2.5),
                    "ci_high": _percentile(estimates, 97.5),
                    "std": (
                        math.sqrt(
                            sum((e - sum(estimates) / len(estimates)) ** 2 for e in estimates)
                            / len(estimates)
                        )
                        if len(estimates) > 1
                        else 0.0
                    ),
                    "p_rank_match": (
                        rank_match / n_reps if len(models) >= 2 else None
                    ),
                    "n_models": len(models),
                }
            )
    return out


def run_all_stats(
    rows: Sequence[dict[str, Any]],
    *,
    n_boot: int = 10_000,
    n_conv_reps: int = 500,
    seed: int = 0,
    assignment_csv: Path | None = None,
    baseline_arm: str = "fixed_prompt",
) -> dict[str, Any]:
    metrics = available_metrics(rows)
    models = sorted({str(r["model"]) for r in rows})
    notes: list[str] = []
    if len(models) < 2:
        notes.append(
            "Single model in panel: pairwise model differences and rank-shift "
            "(Kendall τ) are empty. Arm contrasts and point estimates remain valid."
        )
    if "satisfaction_final" not in metrics:
        notes.append(
            "No satisfaction_final in JSONL yet — satisfaction / P10 / ECDF / sat "
            "columns omitted until critic enrichment."
        )

    estimates = point_estimates(rows, metrics=metrics, n_boot=n_boot, seed=seed)
    contrasts = arm_contrasts(
        rows, metrics=metrics, baseline=baseline_arm, n_boot=n_boot, seed=seed + 10
    )
    model_pairs = pairwise_model_differences(
        rows, metrics=metrics, n_boot=n_boot, seed=seed + 20
    )
    ranks = rank_shift(
        rows,
        metric="task_success",
        baseline_arm=baseline_arm,
        n_boot=n_boot,
        seed=seed + 30,
    )
    # Convergence for main conditioned arm if present, else first non-baseline.
    arms = sorted({str(r["arm"]) for r in rows})
    conv_arm = "full_gepa" if "full_gepa" in arms else next(
        (a for a in arms if a != baseline_arm), arms[0] if arms else baseline_arm
    )
    convergence = convergence_analysis(
        rows,
        metric="task_success",
        arm=conv_arm,
        n_reps=n_conv_reps,
        seed=seed + 40,
        assignment_csv=assignment_csv,
    )

    return {
        "n_rows": len(rows),
        "models": models,
        "arms": sorted({str(r["arm"]) for r in rows}),
        "domains": sorted({str(r["domain"]) for r in rows}),
        "metrics": metrics,
        "notes": notes,
        "point_estimates": estimates,
        "arm_contrasts": contrasts,
        "pairwise_model_differences": model_pairs,
        "rank_shift": ranks,
        "convergence": convergence,
        "holm_family": "model_pairs::<domain> (empty when |models|<2)",
        "seed": seed,
        "n_boot": n_boot,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("reddit/Eval/mcts_phase1/rollouts_core.jsonl"),
    )
    parser.add_argument("--domain", default="travel")
    parser.add_argument("--exclude-sim-error", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-conv-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--assignment",
        type=Path,
        default=Path("eval/mcts/assignments/assignment_block1.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full JSON report (default: <jsonl_dir>/stats_report.json)",
    )
    args = parser.parse_args(argv)

    rows = load_rollouts(
        args.jsonl,
        domain=args.domain,
        exclude_sim_error=args.exclude_sim_error,
    )
    report = run_all_stats(
        rows,
        n_boot=args.n_boot,
        n_conv_reps=args.n_conv_reps,
        seed=args.seed,
        assignment_csv=args.assignment if args.assignment.exists() else None,
    )
    out = args.out or (args.jsonl.parent / "stats_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {out}")
    for note in report["notes"]:
        print(f"NOTE: {note}")
    print(f"models={report['models']} arms={report['arms']} metrics={report['metrics']}")
    print(f"point_estimates={len(report['point_estimates'])} "
          f"arm_contrasts={len(report['arm_contrasts'])} "
          f"model_pairs={len(report['pairwise_model_differences'])} "
          f"rank_shift={len(report['rank_shift'])} "
          f"convergence={len(report['convergence'])}")
    # Compact arm-contrast summary for the one-model case.
    for c in report["arm_contrasts"]:
        if c["metric"] != "task_success":
            continue
        print(
            f"  {c['arm_a']}→{c['arm_b']} success Δ={c['mean_delta']:+.3f} "
            f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] "
            f"n={c['n_pairs']} sign_p={c['sign_test_p']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
