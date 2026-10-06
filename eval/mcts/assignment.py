"""Seeded stratified snake-draft assignment (persona -> task slots)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Iterable

from .schwartz import assign_strata, group_by_stratum, load_schwartz_by_persona_id


def stable_hash_int(*parts: object) -> int:
    """Deterministic 64-bit-ish int from string parts (for RNG / rollout seeds)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True)
class AssignmentRow:
    domain: str
    task_id: str
    slot: int
    persona_id: str
    rollout_seed: int
    stratum: str
    block: int


@dataclass(frozen=True)
class AssignmentMeta:
    master_seed: int
    block: int
    domain: str
    s: int
    strata_mode: str
    p: int
    t: int
    k: int
    created_at: str


def proportional_quota(n_stratum: int, n_total: int, k: int) -> float:
    """Ideal (possibly fractional) count of this stratum on a task of size K."""
    if n_total == 0:
        return 0.0
    return k * (n_stratum / n_total)


def hamilton_quotas(
    stratum_sizes: dict[str, int],
    task_ids: list[str],
    k: int,
) -> dict[str, dict[str, int]]:
    """Per-task integer quotas via floor + remainder seats.

    For stratum size S and T tasks, ideal = S/T (= K*S/P). Each task gets
    floor(S/T) or ceil(S/T), so counts are within ±1 of proportional.
    Remainder seats are placed so every task ends with exactly K personas.
    """
    t = len(task_ids)
    n = sum(stratum_sizes.values())
    if n != t * k:
        raise ValueError(f"P={n} must equal T*K={t * k}")

    labels = sorted(stratum_sizes.keys())
    floors: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for label in labels:
        size = stratum_sizes[label]
        floor_v = size // t
        rem = size - floor_v * t
        floors[label] = floor_v
        remainders[label] = rem
        if rem < 0 or rem >= t:
            raise RuntimeError(f"Invalid remainder for {label}: {rem}")

    base = sum(floors.values())
    if base > k:
        raise RuntimeError(f"Floor load {base} exceeds K={k}")
    need_extra = k - base
    total_rem = sum(remainders.values())
    if total_rem != need_extra * t:
        raise RuntimeError(
            f"Remainder seats {total_rem} != T*need_extra={t * need_extra}"
        )

    quotas: dict[str, dict[str, int]] = {
        tid: {lab: floors[lab] for lab in labels} for tid in task_ids
    }
    extra_count = {tid: 0 for tid in task_ids}

    # Place remainder seats stratum-by-stratum (sorted), preferring tasks with
    # fewest extras so far, then snake index for determinism.
    for label in labels:
        rem = remainders[label]
        if rem == 0:
            continue
        # Eligible: tasks that don't already have the ceil bump for this stratum
        # (each task gets at most +1 per stratum) and still need extras.
        ranked = sorted(
            task_ids,
            key=lambda tid: (
                extra_count[tid],
                task_ids.index(tid),
            ),
        )
        # Prefer tasks still under need_extra.
        under = [tid for tid in ranked if extra_count[tid] < need_extra]
        if len(under) < rem:
            # Should not happen if totals match; fall back to full ranked.
            under = ranked
        chosen = under[:rem]
        if len(chosen) < rem:
            raise RuntimeError(f"Cannot place {rem} remainder seats for {label}")
        for tid in chosen:
            quotas[tid][label] += 1
            extra_count[tid] += 1

    for tid in task_ids:
        if sum(quotas[tid].values()) != k:
            raise RuntimeError(
                f"Task {tid} quota sum {sum(quotas[tid].values())} != K={k} "
                f"(extras={extra_count[tid]}, need={need_extra})"
            )
        if extra_count[tid] != need_extra:
            raise RuntimeError(
                f"Task {tid} extras {extra_count[tid]} != need_extra {need_extra}"
            )
    for label in labels:
        got = sum(quotas[tid][label] for tid in task_ids)
        if got != stratum_sizes[label]:
            raise RuntimeError(
                f"Stratum {label} quota total {got} != size {stratum_sizes[label]}"
            )
    return quotas


def snake_draft(
    strata_groups: dict[str, list[str]],
    task_ids: list[str],
    k: int,
    rng: Random,
) -> list[tuple[str, int, str, str]]:
    """Assign personas to (task_id, slot) via Hamilton quotas + seeded snake fill.

    Returns list of (task_id, slot, persona_id, stratum).
    """
    t = len(task_ids)
    if t == 0:
        raise ValueError("task_ids is empty")
    n = sum(len(v) for v in strata_groups.values())
    if n != t * k:
        raise ValueError(f"P={n} must equal T*K={t * k}")

    pools: dict[str, list[str]] = {}
    for label, ids in strata_groups.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        pools[label] = shuffled

    stratum_sizes = {label: len(ids) for label, ids in pools.items()}
    quotas = hamilton_quotas(stratum_sizes, task_ids, k)

    # Build slot list per task in snake-deal order across tasks, consuming quotas.
    remaining_quota = {tid: dict(q) for tid, q in quotas.items()}
    assignments: list[tuple[str, int, str, str]] = []
    slot_counter: dict[str, int] = {tid: 0 for tid in task_ids}

    labels = sorted(pools.keys())
    # Snake through tasks; at each visit place one persona from a stratum that
    # still has quota on that task (prefer largest remaining quota, then label).
    direction = 1
    idx = 0
    placed = 0
    while placed < n:
        tid = task_ids[idx]
        if sum(remaining_quota[tid].values()) > 0:
            # Choose stratum with remaining quota on this task.
            candidates = [
                lab for lab in labels if remaining_quota[tid].get(lab, 0) > 0 and pools.get(lab)
            ]
            if not candidates:
                raise RuntimeError(f"No placeable stratum for task {tid}")
            candidates.sort(key=lambda lab: (-remaining_quota[tid][lab], lab))
            stratum = candidates[0]
            persona_id = pools[stratum].pop(0)
            remaining_quota[tid][stratum] -= 1
            slot = slot_counter[tid]
            assignments.append((tid, slot, persona_id, stratum))
            slot_counter[tid] += 1
            placed += 1

        next_idx = idx + direction
        if next_idx >= t or next_idx < 0:
            direction *= -1
            next_idx = idx + direction
            if next_idx >= t or next_idx < 0:
                # All remaining on current if stuck (shouldn't happen).
                next_idx = idx
        idx = next_idx

    for tid in task_ids:
        if slot_counter[tid] != k:
            raise RuntimeError(f"Task {tid} got {slot_counter[tid]} personas, expected {k}")
    for label, pool in pools.items():
        if pool:
            raise RuntimeError(f"Unplaced personas in stratum {label}: {len(pool)}")

    return assignments


def generate_block_assignment(
    *,
    schwartz_jsonl: Path,
    domain: str,
    task_ids: list[str],
    master_seed: int,
    block: int,
    k: int,
) -> tuple[AssignmentMeta, list[AssignmentRow]]:
    vectors = load_schwartz_by_persona_id(schwartz_jsonl)
    strata_map, s, mode, _counts = assign_strata(vectors)
    if set(strata_map) != set(vectors):
        raise RuntimeError("strata map / vector key mismatch")

    p = len(vectors)
    t = len(task_ids)
    if p != t * k:
        raise ValueError(f"P={p} must equal T*K={t}*{k}={t * k} for domain={domain}")

    groups = group_by_stratum(strata_map)
    rng = Random(stable_hash_int(master_seed, block, domain, "shuffle"))
    drafted = snake_draft(groups, task_ids, k, rng)

    rows: list[AssignmentRow] = []
    for task_id, slot, persona_id, stratum in drafted:
        seed = stable_hash_int(master_seed, block, domain, task_id, persona_id)
        rows.append(
            AssignmentRow(
                domain=domain,
                task_id=task_id,
                slot=slot,
                persona_id=persona_id,
                rollout_seed=seed,
                stratum=stratum,
                block=block,
            )
        )

    meta = AssignmentMeta(
        master_seed=master_seed,
        block=block,
        domain=domain,
        s=s,
        strata_mode=mode,
        p=p,
        t=t,
        k=k,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return meta, rows


def write_assignment_csv(
    path: Path,
    meta: AssignmentMeta,
    rows: Iterable[AssignmentRow],
    *,
    append_domain: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    header_lines = [
        f"# master_seed={meta.master_seed}",
        f"# block={meta.block}",
        f"# S={meta.s}",
        f"# strata_mode={meta.strata_mode}",
        f"# P={meta.p}",
        f"# T={meta.t}",
        f"# K={meta.k}",
        f"# created_at={meta.created_at}",
        f"# domain={meta.domain}",
    ]
    mode = "a" if append_domain and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("\n".join(header_lines) + "\n")
            handle.write("domain,task_id,slot,persona_id,rollout_seed,stratum\n")
        else:
            handle.write(f"# --- domain={meta.domain} S={meta.s} strata_mode={meta.strata_mode} ---\n")
        for row in sorted(rows, key=lambda r: (r.domain, r.task_id, r.slot)):
            handle.write(
                f"{row.domain},{row.task_id},{row.slot},{row.persona_id},"
                f"{row.rollout_seed},{row.stratum}\n"
            )


def read_assignment_csv(path: Path, block: int | None = None) -> tuple[dict[str, str], list[AssignmentRow]]:
    """Read assignment CSV. Returns (header_meta, rows)."""
    meta: dict[str, str] = {}
    rows: list[AssignmentRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if "=" in body:
                    key, _, val = body.partition("=")
                    meta[key.strip()] = val.strip()
                continue
            if line.startswith("domain,"):
                continue
            parts = line.split(",")
            if len(parts) != 6:
                raise ValueError(f"Bad assignment row: {line!r}")
            domain, task_id, slot_s, persona_id, seed_s, stratum = parts
            b = int(meta.get("block", block if block is not None else 1))
            rows.append(
                AssignmentRow(
                    domain=domain,
                    task_id=task_id,
                    slot=int(slot_s),
                    persona_id=persona_id,
                    rollout_seed=int(seed_s),
                    stratum=stratum,
                    block=b,
                )
            )
    return meta, rows


def validate_assignment(
    rows: list[AssignmentRow],
    *,
    k: int,
    expected_s: int | None = None,
    strata_mode: str | None = None,
) -> list[str]:
    """Return list of invariant violation messages (empty => OK)."""
    errors: list[str] = []
    by_domain: dict[str, list[AssignmentRow]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)

    for domain, drows in by_domain.items():
        personas = [r.persona_id for r in drows]
        if len(personas) != len(set(personas)):
            dup = [p for p, c in Counter(personas).items() if c > 1]
            errors.append(f"{domain}: persona(s) appear more than once: {dup[:5]}")

        by_task: dict[str, list[AssignmentRow]] = defaultdict(list)
        for r in drows:
            by_task[r.task_id].append(r)

        # Stratum pool sizes for proportional check.
        stratum_counts = Counter(r.stratum for r in drows)
        n = len(drows)
        for task_id, trows in by_task.items():
            if len(trows) != k:
                errors.append(f"{domain}/{task_id}: expected K={k} personas, got {len(trows)}")
            ids = [r.persona_id for r in trows]
            if len(ids) != len(set(ids)):
                errors.append(f"{domain}/{task_id}: duplicate personas in task")
            task_strata = Counter(r.stratum for r in trows)
            for label, total in stratum_counts.items():
                ideal = proportional_quota(total, n, k)
                got = task_strata.get(label, 0)
                # Within ±1 of proportional (allow floor/ceil band).
                lo = math.floor(ideal - 1e-9)
                hi = math.ceil(ideal + 1e-9)
                # Spec: differ by ≤1 from uniform/proportional — accept got in [floor(ideal)-0? 
                # "within ±1 of proportional" means |got - ideal| <= 1 when ideal integer-ish,
                # else got in {floor(ideal), ceil(ideal)} which is already |got-ideal|<1,
                # and ±1 expands to floor-1 .. ceil+1. Use |got - ideal| <= 1.0 + eps.
                if abs(got - ideal) > 1.0 + 1e-9:
                    errors.append(
                        f"{domain}/{task_id}: stratum {label} count {got} "
                        f"not within ±1 of proportional {ideal:.3f}"
                    )

        if expected_s is not None:
            n_labels = len(stratum_counts)
            # S is the mode size (10 or 4), not necessarily all labels present.
            if strata_mode and "S4" in strata_mode and expected_s == 4:
                if n_labels > 4:
                    errors.append(f"{domain}: expected <=4 higher-order strata, got {n_labels}")
            elif strata_mode and "S10" in strata_mode and expected_s == 10:
                if n_labels > 10:
                    errors.append(f"{domain}: expected <=10 Schwartz strata, got {n_labels}")

    return errors
