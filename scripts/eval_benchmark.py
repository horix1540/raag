"""Evaluate a benchmark run: join telemetry to labels, score both arms.

This script computes the Milestone 2 headline numbers. It reads every telemetry
row written by ``run_benchmark.py`` under a run tag, joins them to the
benchmark items by ``item_id``, scores answers with the official SQuAD 2.0
EM/F1 (abstention counts as an empty prediction), and produces the comparison
the project is built on:

  hallucination rate - of the unanswerable questions, how often did the arm
      produce an answer anyway (a hallucination by definition)?
  abstention cost    - of the answerable questions, how often did the arm
      abstain and forfeit a question it could have answered?
  EM / F1            - the standard SQuAD 2.0 aggregate over all questions.

    python scripts/eval_benchmark.py --tag squad2-dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import BENCH_DIR, REPORTS_DIR  # noqa: E402
from src.squad_metrics import exact_match_score, f1_score, is_abstention  # noqa: E402
from src.store import connect, init_db  # noqa: E402


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


def score_arm(df: pd.DataFrame, labels: pd.DataFrame, arm: str) -> tuple[dict, pd.DataFrame]:
    """One arm's headline metrics plus the scored frame for further analysis."""
    sub = df[df["arm"] == arm]
    if sub.empty:
        return {"n": 0}, sub
    j = sub.merge(labels, on="item_id", how="left", validate="many_to_one")
    if j["answerable"].isna().any():
        orphans = j.loc[j["answerable"].isna(), "item_id"].unique()[:5]
        raise SystemExit(f"Rows with item_ids not in the benchmark file: {list(orphans)}")

    j["abstained"] = j["answer"].map(is_abstention) | j["verdict"].str.startswith("abstain")
    j["em"] = [
        exact_match_score(a, g) for a, g in zip(j["answer"], j["answers"], strict=True)
    ]
    j["f1"] = [f1_score(a, g) for a, g in zip(j["answer"], j["answers"], strict=True)]

    n_ans = int(j["answerable"].sum())
    n_un = int((~j["answerable"]).sum())
    hallucinated = j.loc[~j["answerable"] & ~j["abstained"]]
    over_abstained = j.loc[j["answerable"] & j["abstained"]]

    out = {
        "n": len(j),
        "n_answerable": n_ans,
        "n_unanswerable": n_un,
        "answered": int((~j["abstained"]).sum()),
        "abstained": int(j["abstained"].sum()),
        "abstention_rate": round(j["abstained"].mean(), 4),
        # Unanswerable questions: producing any answer is a hallucination.
        "hallucination_rate": round(len(hallucinated) / n_un, 4) if n_un else None,
        # Answerable questions: abstaining forfeits a question it could have answered.
        "over_abstention_rate": round(len(over_abstained) / n_ans, 4) if n_ans else None,
        "em": round(j["em"].mean(), 4),
        "f1": round(j["f1"].mean(), 4),
        "em_answerable": round(j.loc[j["answerable"], "em"].mean(), 4) if n_ans else None,
        "f1_answerable": round(j.loc[j["answerable"], "f1"].mean(), 4) if n_ans else None,
        "em_unanswerable": (
            round(j.loc[~j["answerable"], "em"].mean(), 4) if n_un else None
        ),
        "f1_unanswerable": (
            round(j.loc[~j["answerable"], "f1"].mean(), 4) if n_un else None
        ),
        "mean_latency_ms": round(j["total_ms"].mean(), 1),
        "verdicts": j["verdict"].value_counts().to_dict(),
    }
    return out, j


def confusion(j: pd.DataFrame) -> dict:
    """Per-condition behaviour: the table the report will actually argue from."""
    table = {}
    for ans, sub in j.groupby("answerable"):
        key = "answerable" if ans else "unanswerable"
        table[key] = {
            "n": len(sub),
            "answered": int((~sub["abstained"]).sum()),
            "abstained": int(sub["abstained"].sum()),
            "mean_f1": round(sub["f1"].mean(), 4),
        }
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="squad2-dev")
    ap.add_argument("--bench", type=Path, default=BENCH_DIR / "squad2_dev_150.jsonl")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "benchmark_m2.json")
    args = ap.parse_args()

    df = load_run(args.tag)
    labels = load_labels(args.bench)

    report = {"run_tag": args.tag, "n_rows": len(df), "arms": {}}
    print(f"Run {args.tag!r}: {len(df)} rows")
    for arm in ("guarded", "baseline"):
        metrics, scored = score_arm(df, labels, arm)
        report["arms"][arm] = metrics
        report["arms"][arm]["confusion"] = confusion(scored)
        a = report["arms"][arm]
        if a.get("n"):
            print(
                f"\n  {arm:<8} n={a['n']}  answered={a['answered']}  abstained={a['abstained']}"
                f"  EM={a['em']:.3f}  F1={a['f1']:.3f}"
            )
            print(
                f"           hallucination_rate={a['hallucination_rate']:.3f} "
                f"(unanswerable)   over_abstention_rate={a['over_abstention_rate']:.3f} (answerable)"
            )
            print(f"           verdicts={a['verdicts']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    init_db()
    raise SystemExit(main())
