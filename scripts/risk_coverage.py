"""Risk-coverage analysis and figures for one benchmark run (Milestone 3).

The risk-coverage curve is the honest way to compare an abstaining system
against an always-answering one: it plots the error rate among answered
questions (risk) against the fraction of questions answered (coverage). A
system can trivially win either axis alone - answer everything, or answer one
question - so the curve, and the area under it (AURC), is the comparison.

For each arm the score used to rank confidence is:

  guarded  - the composite confidence score, which is what the pipeline
             actually decides on.
  baseline - the same composite, computed but never enforced. The comparison
             is therefore "same scores, no gates" - what the confidence layer
             is worth as a *selector* vs what the gates enforce at one point.

Also plotted: the Gate A score distributions for answerable vs unanswerable
(the separation calibration exploits), and reliability-style binned accuracy.

    python scripts/risk_coverage.py --tag squad2-dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - headless, never block on a display server
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import BENCH_DIR, REPORTS_DIR  # noqa: E402
from src.evalstats import (  # noqa: E402
    aurc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)
from src.squad_metrics import exact_match_score, is_abstention  # noqa: E402
from src.store import connect, init_db  # noqa: E402

FIG_DIR = REPORTS_DIR / "figures"


def load_run(tag: str) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries WHERE run_tag = ? ORDER BY id", (tag,)
        ).fetchall()
    if not rows:
        raise SystemExit(f"No telemetry rows found for run_tag {tag!r}.")
    return pd.DataFrame([dict(r) for r in rows])


def load_labels(bench: Path) -> pd.DataFrame:
    items = [json.loads(line) for line in bench.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(
        {
            "item_id": [i["item_id"] for i in items],
            "answerable": [bool(i["answerable"]) for i in items],
            "answers": [list(i["answers"]) for i in items],
        }
    )


def arm_table(df: pd.DataFrame, labels: pd.DataFrame, arm: str) -> pd.DataFrame:
    """One arm with correctness labels attached: EM==1 counts as correct."""
    sub = df[df["arm"] == arm].merge(labels, on="item_id", how="left")
    sub["correct"] = [
        exact_match_score(a, g) == 1.0
        for a, g in zip(sub["answer"], sub["answers"], strict=True)
    ]
    # SQuAD 2.0: an abstention on an unanswerable question is correct behaviour.
    sub.loc[sub["answer"].map(is_abstention) & ~sub["answerable"], "correct"] = True
    return sub


def fig_risk_coverage(tables: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, sub in tables.items():
        curve = risk_coverage_curve(
            sub["composite"].tolist(), sub["correct"].tolist()
        )
        xs = [p.coverage for p in curve]
        ys = [p.risk for p in curve]
        ax.plot(xs, ys, marker=".", label=f"{arm} (AURC={aurc(curve):.3f})")
    ax.set_xlabel("coverage (fraction of questions answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title("Risk-coverage: guarded vs baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_gate_a_separation(tables: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    guarded = tables["guarded"]
    for ans, sub in guarded.groupby("answerable"):
        ax.hist(
            sub["rerank_relevancy"].dropna(),
            bins=25,
            alpha=0.6,
            label="answerable" if ans else "unanswerable",
        )
    ax.set_xlabel("Gate A rerank relevancy")
    ax.set_ylabel("items")
    ax.set_title("Gate A score separation on SQuAD 2.0 dev sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_reliability(tables: dict, out: Path) -> None:
    """Accuracy vs composite-confidence bin: does the score rank its own errors?"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    guarded = tables["guarded"]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    mid, acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sub = guarded[(guarded["composite"] >= lo) & (guarded["composite"] < hi)]
        if len(sub) >= 3:
            mid.append((lo + hi) / 2)
            acc.append(sub["correct"].mean())
    ax.bar([f"{m:.1f}" for m in mid], acc)
    ax.set_xlabel("composite confidence bin")
    ax.set_ylabel("fraction correct")
    ax.set_title("Reliability: correctness by confidence bin (guarded arm)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="squad2-dev")
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "risk_coverage.json")
    args = ap.parse_args()

    df = load_run(args.tag)
    labels = load_labels(args.bench)
    tables = {arm: arm_table(df, labels, arm) for arm in ("guarded", "baseline")}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_risk_coverage(tables, FIG_DIR / "risk_coverage.png")
    fig_gate_a_separation(tables, FIG_DIR / "gate_a_separation.png")
    fig_reliability(tables, FIG_DIR / "reliability.png")

    report = {"run_tag": args.tag, "arms": {}}
    for arm, sub in tables.items():
        curve = risk_coverage_curve(sub["composite"].tolist(), sub["correct"].tolist())
        report["arms"][arm] = {
            "n": len(sub),
            "aurc": round(aurc(curve), 4),
            "risk_at_50pct_coverage": round(risk_at_coverage(curve, 0.5), 4),
            "risk_at_80pct_coverage": round(risk_at_coverage(curve, 0.8), 4),
            "coverage_at_10pct_risk": round(coverage_at_risk(curve, 0.10), 4),
            "accuracy_full_coverage": round(sub["correct"].mean(), 4),
        }
        print(
            f"  {arm:<8} n={report['arms'][arm]['n']}  AURC={report['arms'][arm]['aurc']:.4f}  "
            f"risk@50%cov={report['arms'][arm]['risk_at_50pct_coverage']:.3f}  "
            f"risk@80%cov={report['arms'][arm]['risk_at_80pct_coverage']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} and 3 figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    init_db()
    raise SystemExit(main())
