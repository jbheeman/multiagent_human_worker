"""Select representative Reddit users from PVQ-enriched records.

This replaces the legacy clustering script that clustered old direct-inference
vectors. The input must come from enrich_users.py after PVQ-40 administration.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pvq import SCHWARTZ_KEYS


def _load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vector = row.get("target_vector") or {}
            if all(k in vector for k in SCHWARTZ_KEYS):
                rows.append(row)
    return rows


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[float(row["target_vector"][key]) for key in SCHWARTZ_KEYS] for row in rows],
        dtype=float,
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    return (x - mean) / std


def _init_centroids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    first = int(rng.integers(0, len(x)))
    chosen = [first]
    min_dist = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(min_dist.sum())
        if total == 0.0:
            remaining = [i for i in range(len(x)) if i not in chosen]
            chosen.append(remaining[0])
        else:
            probs = min_dist / total
            idx = int(rng.choice(len(x), p=probs))
            while idx in chosen and len(chosen) < len(x):
                idx = int(rng.choice(len(x), p=probs))
            chosen.append(idx)
        min_dist = np.minimum(min_dist, np.sum((x - x[chosen[-1]]) ** 2, axis=1))
    return x[chosen].copy()


def _kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = _init_centroids(x, k, rng)
    labels = np.zeros(len(x), dtype=int)

    for _ in range(max_iter):
        distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = x[new_labels == cluster_id]
            if len(members):
                new_centroids[cluster_id] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(distances.min(axis=1)))
                new_centroids[cluster_id] = x[farthest]
                new_labels[farthest] = cluster_id

        if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids

    return labels, centroids


def select_medoids(rows: list[dict], k: int, seed: int = 42) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(rows) < k:
        raise ValueError(f"cannot select k={k} medoids from only {len(rows)} rows")

    x = _standardize(_matrix(rows))
    labels, centroids = _kmeans(x, k=k, seed=seed)

    selected = []
    for cluster_id in range(k):
        member_idx = np.where(labels == cluster_id)[0]
        if len(member_idx) == 0:
            continue
        distances = np.sum((x[member_idx] - centroids[cluster_id]) ** 2, axis=1)
        medoid_idx = int(member_idx[int(np.argmin(distances))])
        row = dict(rows[medoid_idx])
        row["pvq_cluster"] = cluster_id
        row["pvq_cluster_distance"] = float(distances.min())
        selected.append(row)

    if len(selected) < k:
        selected_ids = {row.get("user_id") for row in selected}
        nearest_cluster_distance = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2).min(axis=1)
        fill_order = np.argsort(nearest_cluster_distance)
        next_cluster = k
        for idx in fill_order:
            if len(selected) >= k:
                break
            row = rows[int(idx)]
            if row.get("user_id") in selected_ids:
                continue
            fill = dict(row)
            fill["pvq_cluster"] = next_cluster
            fill["pvq_cluster_distance"] = float(nearest_cluster_distance[int(idx)])
            selected.append(fill)
            selected_ids.add(fill.get("user_id"))
            next_cluster += 1

    selected.sort(key=lambda r: int(r["pvq_cluster"]))
    return selected


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PVQ-cluster medoid users.")
    parser.add_argument(
        "--input",
        default="thousand_users_pvq_enriched.jsonl",
        help="PVQ-enriched JSONL from enrich_users.py",
    )
    parser.add_argument("--k", type=int, required=True, help="number of medoids to select")
    parser.add_argument("--seed", type=int, default=42, help="KMeans initialization seed")
    parser.add_argument(
        "--out",
        default=None,
        help="output JSONL path (default: selected_users_pvq_k<K>.jsonl)",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    selected = select_medoids(rows, k=args.k, seed=args.seed)
    out = args.out or f"selected_users_pvq_k{args.k}.jsonl"
    write_jsonl(selected, out)
    print(f"wrote {out} ({len(selected)} rows from {len(rows)} eligible users)")


if __name__ == "__main__":
    main()
