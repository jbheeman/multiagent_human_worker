"""Figures & tables for MCTS eval (mcts.md §6).

One function per artifact; each saves PDF + PNG.

With a single model, Fig R1 is skipped (needs ≥2 models). Satisfaction-based
artifacts (D1, sat columns in M1, supp sat heatmap) are skipped until
``satisfaction_final`` is present — task_success variants are emitted instead
where useful (A1, C1, M1 success/transfer, supp success heatmap).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from eval.mcts.assignment import read_assignment_csv
from eval.mcts.stats import (
    DEFAULT_ARM_ORDER,
    available_metrics,
    load_rollouts,
    metric_value,
    run_all_stats,
)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plots: pip install matplotlib"
        ) from exc
    return plt


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        paths.append(path)
    return paths


def _arm_order(arms: Sequence[str]) -> list[str]:
    ordered = [a for a in DEFAULT_ARM_ORDER if a in arms]
    ordered.extend(sorted(set(arms) - set(ordered)))
    return ordered


def table_m1(
    report: dict[str, Any],
    out_dir: Path,
    *,
    domain: str,
    baseline_arm: str = "fixed_prompt",
    conditioned_arm: str = "full_gepa",
) -> Path:
    """Markdown table: models × (fixed_prompt vs full_gepa) metrics with CIs."""
    estimates = [
        e
        for e in report["point_estimates"]
        if e["domain"] == domain and e["arm"] in (baseline_arm, conditioned_arm)
    ]
    models = sorted({e["model"] for e in estimates}) or report.get("models") or []
    metrics = []
    for m in ("task_success", "transfer", "satisfaction_final"):
        if any(e["metric"] == m for e in estimates):
            metrics.append(m)

    def cell(model: str, arm: str, metric: str) -> str:
        hits = [
            e
            for e in estimates
            if e["model"] == model and e["arm"] == arm and e["metric"] == metric
        ]
        if not hits:
            return "—"
        e = hits[0]
        s = f"{e['mean']:.3f} [{e['ci_low']:.3f}, {e['ci_high']:.3f}]"
        if metric == "satisfaction_final" and e.get("p10") is not None:
            s += f" (P10={e['p10']:.3f})"
        return s

    lines = [
        f"# Table M1 — {domain}",
        "",
        f"Models: {', '.join(models) or '(none)'}. "
        f"Column groups: `{baseline_arm}` vs `{conditioned_arm}`.",
        "",
    ]
    if not models:
        lines.append("_No estimates available._")
    else:
        header = ["model"]
        for arm in (baseline_arm, conditioned_arm):
            for metric in metrics:
                header.append(f"{arm}/{metric}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for model in models:
            row = [model]
            for arm in (baseline_arm, conditioned_arm):
                for metric in metrics:
                    row.append(cell(model, arm, metric))
            lines.append("| " + " | ".join(row) + " |")

    if "satisfaction_final" not in metrics:
        lines.extend(
            [
                "",
                "_Satisfaction columns omitted — critic enrichment not present yet._",
            ]
        )
    if len(models) < 2:
        lines.extend(["", f"_Single-model panel ({models[0] if models else 'n/a'})._"])

    path = out_dir / "table_m1.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fig_r1(
    report: dict[str, Any],
    out_dir: Path,
    *,
    domain: str,
) -> list[Path]:
    """Bump chart: model ranks fixed_prompt → conditioned. Needs ≥2 models."""
    plt = _require_matplotlib()
    shifts = [r for r in report.get("rank_shift", []) if r["domain"] == domain]
    if not shifts:
        note = out_dir / "fig_r1_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "Fig R1 skipped: need ≥2 models for rank-shift / bump chart.\n",
            encoding="utf-8",
        )
        return [note]

    n = len(shifts)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    for ax, shift in zip(axes[0], shifts):
        base = shift["baseline_ranking"]
        cond = shift["conditioned_ranking"]
        models = list(dict.fromkeys(base + cond))
        for model in models:
            if model not in base or model not in cond:
                continue
            y0 = base.index(model) + 1
            y1 = cond.index(model) + 1
            ax.plot([0, 1], [y0, y1], "-o", label=model)
            ax.text(-0.05, y0, model, ha="right", va="center", fontsize=8)
            ax.text(1.05, y1, model, ha="left", va="center", fontsize=8)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(len(models) + 0.5, 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([shift["baseline_arm"], shift["conditioned_arm"]])
        ax.set_ylabel("rank (1=best)")
        tau = shift["kendall_tau"]
        lo, hi = shift["ci_low"], shift["ci_high"]
        ax.set_title(f"τ={tau:.2f} [{lo:.2f}, {hi:.2f}]")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Fig R1 — rank shift ({domain})")
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig_r1_{domain}")
    plt.close(fig)
    return paths


def fig_c1(
    report: dict[str, Any],
    out_dir: Path,
    *,
    domain: str,
) -> list[Path]:
    """Convergence: metric estimate vs n personas with CI bands."""
    plt = _require_matplotlib()
    rows = [c for c in report.get("convergence", []) if c["domain"] == domain]
    if not rows:
        note = out_dir / "fig_c1_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text("Fig C1 skipped: no convergence rows.\n", encoding="utf-8")
        return [note]

    # One series per arm in report (currently one arm from run_all_stats).
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, series in sorted(by_arm.items()):
        series = sorted(series, key=lambda r: r["n_personas"])
        xs = [r["n_personas"] for r in series]
        ys = [r["mean"] for r in series]
        lo = [r["ci_low"] for r in series]
        hi = [r["ci_high"] for r in series]
        ax.plot(xs, ys, "-o", label=arm)
        ax.fill_between(xs, lo, hi, alpha=0.2)
    ax.axvline(10, color="gray", ls="--", lw=1, label="n=10")
    ax.axvline(200, color="gray", ls=":", lw=1, label="n=200")
    ax.set_xlabel("n personas")
    ax.set_ylabel(rows[0]["metric"])
    ax.set_title(f"Fig C1 — convergence ({domain})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig_c1_{domain}")
    plt.close(fig)
    return paths


def fig_d1(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    domain: str,
    arm: str = "full_gepa",
) -> list[Path]:
    """Per-model satisfaction ECDF over personas; P10 marked."""
    plt = _require_matplotlib()
    sat_rows = [
        r
        for r in rows
        if r.get("domain") == domain
        and r.get("arm") == arm
        and r.get("satisfaction_final") is not None
    ]
    if not sat_rows:
        note = out_dir / "fig_d1_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "Fig D1 skipped: no satisfaction_final yet (awaiting critic).\n",
            encoding="utf-8",
        )
        return [note]

    # Persona-level mean satisfaction per model.
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in sat_rows:
        by_model[str(r["model"])][str(r["persona_id"])].append(float(r["satisfaction_final"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, per_persona in sorted(by_model.items()):
        vals = sorted(sum(vs) / len(vs) for vs in per_persona.values() if vs)
        if not vals:
            continue
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, ys, where="post", label=model)
        p10 = vals[max(0, int(0.1 * (len(vals) - 1)))]
        ax.axvline(p10, ls="--", alpha=0.5)
    ax.set_xlabel("satisfaction_final (persona mean)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Fig D1 — satisfaction ECDF ({domain}, {arm})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig_d1_{domain}_{arm}")
    plt.close(fig)
    return paths


def fig_a1(
    report: dict[str, Any],
    out_dir: Path,
    *,
    domain: str,
    metric: str = "task_success",
) -> list[Path]:
    """Arm ablation forest: paired per-persona deltas vs baseline."""
    plt = _require_matplotlib()
    contrasts = [
        c
        for c in report.get("arm_contrasts", [])
        if c["domain"] == domain and c["metric"] == metric
    ]
    if not contrasts:
        note = out_dir / "fig_a1_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text("Fig A1 skipped: no arm contrasts.\n", encoding="utf-8")
        return [note]

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in contrasts:
        by_model[c["model"]].append(c)

    n_models = len(by_model)
    fig, axes = plt.subplots(
        1, n_models, figsize=(max(4.5, 4.0 * n_models), 4.0), squeeze=False
    )
    for ax, (model, items) in zip(axes[0], sorted(by_model.items())):
        items = sorted(items, key=lambda c: (c["arm_a"], c["arm_b"]))
        ys = list(range(len(items)))
        for y, c in zip(ys, items):
            ax.errorbar(
                c["mean_delta"],
                y,
                xerr=[[c["mean_delta"] - c["ci_low"]], [c["ci_high"] - c["mean_delta"]]],
                fmt="o",
                capsize=3,
            )
        ax.axvline(0.0, color="black", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{c['arm_a']}→{c['arm_b']}" for c in items], fontsize=8)
        ax.set_xlabel(f"Δ {metric}")
        ax.set_title(model)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"Fig A1 — arm ablation ({domain})")
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig_a1_{domain}_{metric}")
    plt.close(fig)
    return paths


def fig_supp_heatmap(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    domain: str,
    assignment_csv: Path | None,
    metric: str = "task_success",
) -> list[Path]:
    """Persona-stratum × model (or arm, if one model) mean metric heatmap."""
    plt = _require_matplotlib()
    subset = [r for r in rows if r.get("domain") == domain]
    if not subset:
        note = out_dir / "fig_supp_heatmap_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text("Supp heatmap skipped: no rows.\n", encoding="utf-8")
        return [note]

    strata_map: dict[str, str] = {}
    if assignment_csv and assignment_csv.exists():
        _, arows = read_assignment_csv(assignment_csv)
        strata_map = {r.persona_id: r.stratum for r in arows if r.domain == domain}

    models = sorted({str(r["model"]) for r in subset})
    # Spec is stratum × model; with one model, show stratum × arm instead.
    if len(models) >= 2:
        col_key = "model"
        cols = models
        title_bit = "model"
    else:
        col_key = "arm"
        cols = _arm_order(sorted({str(r["arm"]) for r in subset}))
        title_bit = "arm"

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in subset:
        val = metric_value(r, metric)
        if val is None:
            continue
        stratum = strata_map.get(str(r["persona_id"]), "UNKNOWN")
        buckets[(stratum, str(r[col_key]))].append(val)

    strata = sorted({s for s, _ in buckets})
    if not strata or not cols:
        note = out_dir / "fig_supp_heatmap_SKIPPED.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        note.write_text("Supp heatmap skipped: empty matrix.\n", encoding="utf-8")
        return [note]

    matrix = []
    for s in strata:
        matrix.append(
            [
                (sum(buckets[(s, c)]) / len(buckets[(s, c)]))
                if buckets.get((s, c))
                else float("nan")
                for c in cols
            ]
        )

    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2.5, 0.5 * len(strata) + 2.0))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels(strata, fontsize=8)
    for i, s in enumerate(strata):
        for j, c in enumerate(cols):
            v = matrix[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title(f"Supp — stratum × {title_bit} mean {metric} ({domain})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths = _save(fig, out_dir, f"fig_supp_heatmap_{domain}_{metric}")
    plt.close(fig)
    return paths


def generate_all(
    *,
    rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
    out_dir: Path,
    domain: str,
    assignment_csv: Path | None = None,
) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}

    m1 = table_m1(report, out_dir, domain=domain)
    written["table_m1"] = [str(m1)]

    for name, paths in [
        ("fig_r1", fig_r1(report, out_dir, domain=domain)),
        ("fig_c1", fig_c1(report, out_dir, domain=domain)),
        ("fig_d1", fig_d1(rows, out_dir, domain=domain)),
        ("fig_a1", fig_a1(report, out_dir, domain=domain, metric="task_success")),
        (
            "fig_supp_heatmap",
            fig_supp_heatmap(
                rows, out_dir, domain=domain, assignment_csv=assignment_csv
            ),
        ),
    ]:
        written[name] = [str(p) for p in paths]

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(written, indent=2), encoding="utf-8")
    written["manifest"] = [str(manifest)]
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("reddit/Eval/mcts_phase1/rollouts_core.jsonl"),
    )
    parser.add_argument("--domain", default="travel")
    parser.add_argument("--exclude-sim-error", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--n-conv-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--assignment",
        type=Path,
        default=Path("eval/mcts/assignments/assignment_block1.csv"),
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Reuse an existing stats_report.json instead of recomputing",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <jsonl_dir>/plots",
    )
    args = parser.parse_args(argv)

    rows = load_rollouts(
        args.jsonl,
        domain=args.domain,
        exclude_sim_error=args.exclude_sim_error,
    )
    assignment = args.assignment if args.assignment.exists() else None

    if args.stats_json and args.stats_json.exists():
        report = json.loads(args.stats_json.read_text(encoding="utf-8"))
    else:
        report = run_all_stats(
            rows,
            n_boot=args.n_boot,
            n_conv_reps=args.n_conv_reps,
            seed=args.seed,
            assignment_csv=assignment,
        )
        stats_path = args.jsonl.parent / "stats_report.json"
        stats_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {stats_path}")

    out_dir = args.out_dir or (args.jsonl.parent / "plots")
    written = generate_all(
        rows=rows,
        report=report,
        out_dir=out_dir,
        domain=args.domain,
        assignment_csv=assignment,
    )
    print(f"metrics available: {available_metrics(rows)}")
    for name, paths in written.items():
        print(f"{name}:")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
