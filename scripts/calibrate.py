"""Fit the gate thresholds on the labelled SQuAD 2.0 dev sample (Milestone 3).

The shipped thresholds are placeholders by design. This script replaces them
with values fitted against ground truth:

  Gate A - every benchmark item carries an answerable/unanswerable label, and
      Gate A's job is exactly that binary decision (should generation be
      attempted at all?). We sweep the threshold over ``rerank_relevancy`` and
      report the full precision/recall/F1 table, then fit a logistic
      regression over the four recorded features to check whether the extra
      embedding statistics buy anything over the reranker alone.

  Gate B - a ground-truth "grounded" label does not exist for SQuAD, so we use
      the standard proxy: baseline-arm answers to answerable questions are
      mostly grounded (the gold paragraph is in the context set), while
      baseline-arm answers to unanswerable questions are fabrications by
      construction. We sweep ``gate_b_threshold`` over that split.

The chosen operating points maximise Youden's J on Gate A (the abstention
decision should be balanced, not tuned for one side) and F1 on Gate B, then
are written to ``reports/calibration.json``. ``src/config.py`` reads nothing
from that file automatically - the values are copied into the defaults by hand
so every run remains reproducible from a single source of truth.

    python scripts/calibrate.py --tag squad2-dev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import BENCH_DIR, REPORTS_DIR, settings  # noqa: E402
from src.store import connect, init_db  # noqa: E402

GATE_A_FEATURES = ["rerank_relevancy", "top_similarity", "mean_similarity", "similarity_margin"]


def load_run(tag: str) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries WHERE run_tag = ? ORDER BY id", (tag,)
        ).fetchall()
    if not rows:
        raise SystemExit(f"No telemetry rows found for run_tag {tag!r}. Run the benchmark first.")
    return pd.DataFrame([dict(r) for r in rows])


def load_labels(bench: Path) -> pd.DataFrame:
    items = [json.loads(line) for line in bench.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(
        {"item_id": [i["item_id"] for i in items], "answerable": [bool(i["answerable"]) for i in items]}
    )


def sweep_threshold(
    scores: np.ndarray, labels: np.ndarray, *, kind: str, min_threshold: float = 0.0
) -> tuple[dict, list[dict]]:
    """Sweep one threshold over a binary labelling; return the best point and the table.

    ``kind=gate_a``: positive class = answerable (score above threshold should
    mean "generation is worthwhile"). ``kind=gate_b``: positive class = grounded.

    ``min_threshold`` bounds the sweep. Gate B is constrained to stay above the
    contradiction cap (see ``config.py``): below it, a capped answer would
    still pass the gate and calibration would have inverted the one-contradiction-
    is-a-hard-failure rule it is explicitly not allowed to touch.
    """
    table = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        if t < min_threshold:
            continue
        pred = scores >= t
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Youden's J = tpr - fpr; only meaningful for the Gate A abstention
        # decision, where both error types cost the same.
        tpr = recall
        fpr = fp / (fp + int(np.sum(~pred & ~labels))) if kind == "gate_a" else float("nan")
        youden = tpr - fpr
        table.append(
            {
                "threshold": float(t),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "youden_j": round(float(youden), 4) if not np.isnan(youden) else None,
            }
        )
    criterion = "youden_j" if kind == "gate_a" else "f1"
    best = max(table, key=lambda r: r[criterion])
    return best, table


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney): P(score_positive > score_negative).

    sklearn's implementation rather than a hand-rolled double loop: the naive
    version is O(n*m) in pure Python, which is invisible at 150 items and
    becomes the slowest part of calibration once a run reaches a few thousand.
    Both compute the same statistic, ties included at half weight.
    """
    if labels.all() or not labels.any():
        # AUC is undefined with only one class present, which happens on a
        # held-out half that landed all-answerable.
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def split_items(item_ids: list[str], *, test_frac: float = 0.5, seed: int = 20260916) -> set[str]:
    """Deterministic item-level train/test split; returns the held-out ids.

    Split on ``item_id`` rather than on telemetry rows, because each item
    contributes two rows (one per arm) and letting an item's guarded row train
    a threshold that its baseline row then scores would leak across the
    boundary. Seeded so a re-run reproduces the same halves: the point of a
    held-out set is lost if it moves every time the script is invoked.
    """
    uniq = sorted(set(item_ids))
    shuffled = list(uniq)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[: int(round(len(shuffled) * test_frac))])


def evaluate_at(scores: np.ndarray, labels: np.ndarray, threshold: float, *, kind: str) -> dict:
    """Score one fixed threshold on a set it was not fitted on."""
    if len(scores) == 0:
        return {"n": 0}
    pred = scores >= threshold
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    tn = int(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "n": int(len(scores)),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc(scores, labels), 4) if not np.isnan(roc_auc(scores, labels)) else None,
    }
    if kind == "gate_a":
        fpr = fp / (fp + tn) if fp + tn else 0.0
        out["youden_j"] = round(recall - fpr, 4)
    return out


def calibrate_gate_a(joined: pd.DataFrame) -> dict:
    """Gate A: answerable/unanswerable is exactly the gate's decision."""
    df = joined[joined["arm"] == "baseline"]  # any arm; features are arm-independent
    df = df.dropna(subset=["rerank_relevancy"])
    scores = df["rerank_relevancy"].to_numpy(dtype=float)
    labels = df["answerable"].to_numpy(dtype=bool)

    best, table = sweep_threshold(scores, labels, kind="gate_a")

    # Is a multivariate fit better than the single reranker feature?
    logistic = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = df[GATE_A_FEATURES].to_numpy(dtype=float)
        model = LogisticRegression(max_iter=1000)
        auc = float(np.mean(cross_val_score(model, X, labels, cv=5, scoring="roc_auc")))
        model.fit(X, labels)
        logistic = {
            "features": GATE_A_FEATURES,
            "coef": [round(float(c), 4) for c in model.coef_[0]],
            "intercept": round(float(model.intercept_[0]), 4),
            "cv_roc_auc": round(auc, 4),
        }
    except Exception as exc:  # noqa: BLE001 - sklearn fit is informative, not required
        logistic = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "n": int(len(df)),
        "signal": "rerank_relevancy",
        "roc_auc": round(roc_auc(scores, labels), 4),
        "chosen": best,
        "sweep": table,
        "logistic_fit": logistic,
    }


def calibrate_gate_b(joined: pd.DataFrame) -> dict:
    """Gate B: grounded = baseline answers on answerable items, fabricated = the rest.

    Only baseline rows are used because the guarded arm suppresses exactly the
    rows Gate B is meant to catch, which would bias the label mix. The
    ``nli_faithfulness`` score exists on both; the proxy label is what differs.
    """
    df = joined[(joined["arm"] == "baseline") & joined["nli_faithfulness"].notna()]
    scores = df["nli_faithfulness"].to_numpy(dtype=float)
    labels = df["answerable"].to_numpy(dtype=bool)  # grounded proxy

    # The sweep may not cross the contradiction cap.
    cap = float(getattr(settings.confidence, "contradiction_cap", 0.15))

    best, table = sweep_threshold(scores, labels, kind="gate_b",
                                   min_threshold=cap + 0.05)  # strictly above the cap
    return {
        "n": int(len(df)),
        "signal": "nli_faithfulness",
        "label": "answerable-as-grounded proxy (baseline arm)",
        "contradiction_cap_respected": True,
        "roc_auc": round(roc_auc(scores, labels), 4),
        "chosen": best,
        "sweep": table,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="squad2-dev")
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "calibration.json")
    args = ap.parse_args()

    run = load_run(args.tag)
    labels = load_labels(args.bench)
    joined = run.merge(labels, on="item_id", how="inner")

    gate_a = calibrate_gate_a(joined)
    gate_b = calibrate_gate_b(joined)

    # The composite threshold is set 10% above the harmonic composite of two
    # *both-marginal* gate passes. At the corner value itself the band would be
    # structurally empty - the harmonic mean is monotone in both arguments, so
    # no pair that passed both gates can composite below the corner - and the
    # flag exists precisely to catch answers that squeaked through both.
    wa, wb = settings.confidence.weights
    ta, tb = gate_a["chosen"]["threshold"], gate_b["chosen"]["threshold"]
    corner = (wa + wb) / (wa / ta + wb / tb)
    composite = round(corner * 1.10, 4)

    report = {
        "run_tag": args.tag,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "recommended": {
            "gate_a_threshold": gate_a["chosen"]["threshold"],
            "gate_b_threshold": gate_b["chosen"]["threshold"],
            "composite_threshold": composite,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Calibration on run {args.tag!r} ({gate_a['n']} Gate A rows, {gate_b['n']} Gate B rows)")
    print(f"  Gate A  AUC={gate_a['roc_auc']:.3f}  chosen threshold={gate_a['chosen']['threshold']}"
          f"  (Youden J={gate_a['chosen']['youden_j']})")
    print(f"  Gate B  AUC={gate_b['roc_auc']:.3f}  chosen threshold={gate_b['chosen']['threshold']}"
          f"  (F1={gate_b['chosen']['f1']})")
    if "coef" in str(gate_a.get("logistic_fit", {})) or gate_a.get("logistic_fit", {}).get("cv_roc_auc"):
        lf = gate_a["logistic_fit"]
        print(f"  Logistic over 4 features: cv AUC={lf['cv_roc_auc']:.3f} vs reranker alone")
    print(f"  Composite (harmonic of the two): {composite}")
    print(f"Wrote {args.out}")
    print("Apply with: scripts/calibrate.py --apply  (or copy into src/config.py defaults)")
    return 0


if __name__ == "__main__":
    init_db()
    raise SystemExit(main())
