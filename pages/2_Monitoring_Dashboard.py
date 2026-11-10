"""Monitoring dashboard: retrieval quality and confidence trends over time.

This is the "monitoring" deliverable from the project description. It reads the
SQLite telemetry written by every pipeline run, so it reflects real queries
rather than a canned demo dataset. Interactive questions and benchmark sweeps
share one table, so the page always asks which of them to analyse: pooling a
150-item benchmark with five demo questions would describe neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui  # noqa: E402
from src.store import clear, connect, init_db, load_dataframe  # noqa: E402

st.set_page_config(page_title="Monitoring · Confidence-Aware RAG", page_icon="📊", layout="wide")
ui.inject_css()
ui.hero("📊 Monitoring dashboard",
        "How often the system answers or abstains, why, how confident it is, and where the time goes.")

INTERACTIVE = "Questions asked in the app"
# The Milestone 4 HotpotQA sweep logged its rows without a run tag; its
# per-item indexes are named in ``sources``, which is what tells them apart.
HOTPOT = "HotpotQA sweep (Milestone 4)"


def _untagged() -> pd.DataFrame:
    df = load_dataframe(limit=100_000)
    if "run_tag" not in df.columns:
        return df
    df = df[df["run_tag"].isna()]
    src = df["sources"].astype(str)
    df = df.assign(_hotpot=src.str.contains("hotpot:"), _empty=src.isin(["[]", ""]))
    return df


def _scopes() -> list[str]:
    init_db()
    with connect() as conn:
        tags = [r["run_tag"] for r in conn.execute(
            "SELECT run_tag, MAX(id) AS last FROM queries WHERE run_tag IS NOT NULL "
            "GROUP BY run_tag ORDER BY last DESC")]
    return [INTERACTIVE, HOTPOT, *tags]


with st.sidebar:
    scope = st.selectbox("Data shown", _scopes(),
                         help="Your own questions from the app, or one benchmark run.")
    limit = st.slider("Most recent rows to load", 20, 3000, 500, step=20)

if scope in (INTERACTIVE, HOTPOT):
    df = _untagged()
    if "_hotpot" in df.columns:
        # Rows with no sources at all are aborted harness starts, not real questions.
        df = df[df["_hotpot"]] if scope == HOTPOT else df[~df["_hotpot"] & ~df["_empty"]]
        df = df.drop(columns=["_hotpot", "_empty"])
    df = df.tail(limit)
else:
    df = load_dataframe(limit=limit, run_tag=scope)

if df.empty:
    st.info("Nothing logged for this selection yet. Ask a few questions on the main page, then come back.")
    st.page_link("app.py", label="Go to the app", icon="🛡️")
    st.stop()

abstained = df["verdict"].str.startswith("abstain")
flagged = df["verdict"].eq("flag_low_confidence")

c1, c2, c3, c4 = st.columns(4)
with c1.container(border=True):
    st.metric("Queries", len(df))
with c2.container(border=True):
    st.metric("Answered", int((~abstained).sum()), f"{flagged.sum()} flagged low-confidence", delta_color="off")
with c3.container(border=True):
    st.metric("Abstained", int(abstained.sum()), f"{abstained.mean():.0%} of queries", delta_color="off")
with c4.container(border=True):
    st.metric("Median time per query", f"{df['total_ms'].median() / 1000:.1f} s")

overview, claims_tab, bench, latency, log = st.tabs(
    ["Overview", "Claims", "Benchmark arms", "Latency", "Query log"])

with overview:
    left, right = st.columns(2)
    with left:
        st.markdown("**Verdicts**")
        counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="queries")
        counts["verdict"] = counts["verdict"].map(lambda v: ui.VERDICTS.get(v, (v,))[0])
        st.bar_chart(counts.set_index("verdict"), height=280, horizontal=True)
    with right:
        st.markdown("**Confidence trend**")
        trend_cols = [c for c in ("gate_a_score", "gate_b_score", "composite") if c in df.columns]
        trend = df.sort_values("id")[trend_cols].rename(columns={
            "gate_a_score": "Gate A", "gate_b_score": "Gate B", "composite": "Combined"})
        trend.index = pd.RangeIndex(1, len(trend) + 1, name="query #")
        st.line_chart(trend, height=280, x_label="query #")
    st.caption("A downward drift in Gate A means people are asking about things the documents do not "
               "cover: the signal that the knowledge base needs updating.")

with claims_tab:
    supported = int((df["n_claims"].fillna(0) - df["n_conflicts"].fillna(0) - df["n_baseless"].fillna(0)).sum())
    tax = pd.Series({"Supported": supported,
                     "Not found in sources": int(df["n_baseless"].fillna(0).sum()),
                     "Contradicted": int(df["n_conflicts"].fillna(0).sum())}, name="claims")
    k1, k2, k3 = st.columns(3)
    k1.metric("Supported claims", tax["Supported"])
    k2.metric("Not found in sources", tax["Not found in sources"])
    k3.metric("Contradicted", tax["Contradicted"])
    st.bar_chart(tax, height=260)
    st.caption("RAGTruth taxonomy, summed over every answer that reached Gate B.")

with bench:
    if "arm" in df.columns and df["arm"].nunique() > 1:
        rows = []
        for arm, sub in df.groupby("arm"):
            ab = sub["verdict"].str.startswith("abstain")
            rows.append({"arm": arm, "queries": len(sub), "answered": int((~ab).sum()),
                         "abstained": int(ab.sum()), "mean combined score": round(sub["composite"].mean(), 3),
                         "median seconds": round(sub["total_ms"].median() / 1000, 2)})
        st.dataframe(pd.DataFrame(rows).set_index("arm"), use_container_width=True)
        st.bar_chart(df.groupby(["arm", "verdict"]).size().unstack(fill_value=0), height=280)
        st.caption("The baseline arm measures the gates but never enforces them, so it always answers.")
    else:
        st.info("This selection has a single arm. Benchmark runs that compared the guarded pipeline "
                "with the baseline (for example squad2-v2) show both arms side by side here.")
    st.caption("Accuracy, hallucination and over-abstention rates need the dataset labels; they are "
               "computed by scripts/audit_results.py and reported in reports/PROJECT_REVIEW.md.")

with latency:
    lat_cols = {"retrieval_ms": "Retrieval", "gate_a_ms": "Gate A",
                "generation_ms": "Generation", "gate_b_ms": "Gate B"}
    present = [c for c in lat_cols if c in df.columns]
    med = df[present].median().rename(lat_cols).round(0)
    st.bar_chart(med.rename("median ms"), height=280, horizontal=True)
    st.caption("Median milliseconds per stage. Medians, because a sleeping laptop during a long run "
               "produces a few huge outliers that would wreck an average. Both gates together stay "
               "small next to generation.")

with log:
    show = [c for c in ("id", "ts", "question", "verdict", "composite", "gate_a_score", "gate_b_score",
                        "n_conflicts", "n_baseless", "total_ms", "run_tag", "arm") if c in df.columns]
    st.dataframe(df[show].sort_values("id", ascending=False), use_container_width=True, height=380,
                 hide_index=True)
    st.download_button("Download this selection as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="confidence_rag_telemetry.csv", mime="text/csv")

with st.sidebar.expander("Danger zone"):
    st.caption("Deletes every logged query, including all benchmark runs. Only for a clean demo.")
    sure = st.checkbox("I understand this deletes the benchmark results too")
    if st.button("Clear all telemetry", type="secondary", disabled=not sure):
        clear()
        st.rerun()
