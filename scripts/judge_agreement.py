"""NLI-vs-LLM-judge agreement study (Milestone 3).

The confidence layer's claim is that cheap local cross-encoders can replace a
slower LLM judge for the two gate decisions. This script measures that claim
instead of asserting it: on a subset of the benchmark, both scorers ran on the
same items (``RAG_SCORER=both``), so telemetry holds their scores side by
side. We report correlation, Cohen's kappa on the pass/fail decision, the
disagreement cases, and what the judge costs in latency.

    python scripts/judge_agreement.py --tag squad2-judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR, settings  # noqa: E402
from src.evalstats import judge_agreement  # noqa: E402
from src.store import connect, init_db  # noqa: E402


def load_run(tag: str) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries WHERE run_tag = ? ORDER BY id", (tag,)
        ).fetchall()
    if not rows:
        raise SystemExit(f"No telemetry rows found for run_tag {tag!r}.")
    return pd.DataFrame([dict(r) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="squad2-judge")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "judge_agreement.json")
    args = ap.parse_args()

    df = load_run(args.tag)
    both = df[df["llm_faithfulness"].notna() | df["llm_relevancy"].notna()]
    if both.empty:
        raise SystemExit(
            "No rows carry judge scores. Re-run the subset with RAG_SCORER=both "
            "(scripts/run_benchmark.py --tag squad2-judge does this automatically)."
        )

    thresholds = json.loads(both["thresholds_json"].iloc[0]) if both["thresholds_json"].iloc[0] else {}
    t_a = float(thresholds.get("gate_a", settings.confidence.gate_a_threshold))
    t_b = float(thresholds.get("gate_b", settings.confidence.gate_b_threshold))

    # The judge's scores live on their own scales, so the local scorer's
    # calibrated threshold cannot be reused for it. DeepEval's contextual
    # relevancy is a ratio of relevant statements and in this run never exceeds
    # 0.45, so applying Gate A's 0.45 to it fails every item and yields a kappa
    # of zero that says nothing about whether the two scorers agree. Its
    # faithfulness verdict is effectively binary (0.0 or 1.0), so 0.5 is the
    # only cut-off with a meaning. Both are stated here rather than inherited.
    t_llm_a = 0.5
    t_llm_b = 0.5

    gate_a = judge_agreement(
        both["rerank_relevancy"].tolist(),
        both["llm_relevancy"].tolist(),
        t_a,
        llm_threshold=t_llm_a,
    )
    gate_b = judge_agreement(
        both["nli_faithfulness"].tolist(),
        both["llm_faithfulness"].tolist(),
        t_b,
        llm_threshold=t_llm_b,
    )

    # Latency cost of the judge: rows where it ran vs the nli-only sweep.
    with connect() as conn:
        nli_lat = pd.DataFrame(
            [
                dict(r)
                for r in conn.execute(
                    "SELECT gate_a_ms, gate_b_ms FROM queries "
                    "WHERE run_tag != ? AND scorer = 'nli' AND arm = 'guarded'",
                    (args.tag,),
                )
            ]
        )
    nli_rows = (nli_lat["gate_a_ms"].median(), nli_lat["gate_b_ms"].median())
    report = {
        "run_tag": args.tag,
        "n": len(both),
        "thresholds_used": {
            "nli_gate_a": t_a,
            "nli_gate_b": t_b,
            "llm_gate_a": t_llm_a,
            "llm_gate_b": t_llm_b,
        },
        "gate_a_relevancy": gate_a,
        "gate_b_faithfulness": gate_b,
        # Medians, not means. Two rows in this run carry Gate A latencies of 10
        # and 36 hours: the laptop suspended mid-sweep and the wall-clock timer
        # kept counting. A mean over 36 items is destroyed by that; the median
        # is not. The same reason applies to every latency figure in the report.
        "latency_ms": {
            "judge_run_gate_a_median": round(float(both["gate_a_ms"].median()), 1),
            "judge_run_gate_b_median": round(float(both["gate_b_ms"].dropna().median()), 1),
            "nli_only_gate_a_median": round(float(nli_rows[0] or 0), 1),
            "nli_only_gate_b_median": round(float(nli_rows[1] or 0), 1),
        },
    }

    print(
        f"Agreement study on {len(both)} items "
        f"(nli thresholds A={t_a} B={t_b}; llm thresholds A={t_llm_a} B={t_llm_b})"
    )
    for name, g in (("Gate A (relevancy)", gate_a), ("Gate B (faithfulness)", gate_b)):
        d = g.get("decision_agreement", {})
        rm = g.get("decision_agreement_rate_matched", {})
        print(
            f"  {name}: n={g['n']}  spearman={g['spearman']}  "
            f"kappa={d.get('cohens_kappa')} observed={d.get('observed_agreement')}"
        )
        print(
            f"      rate-matched: kappa={rm.get('cohens_kappa')} "
            f"observed={rm.get('observed_agreement')} "
            f"(both scorers pass {rm.get('both_pass', 0) + rm.get('only_x', 0)} items)"
        )
    print(f"  Latency: judge run {report['latency_ms']['judge_run_gate_b_median']}ms vs "
          f"nli-only {report['latency_ms']['nli_only_gate_b_median']}ms (Gate B, medians)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    init_db()
    raise SystemExit(main())
