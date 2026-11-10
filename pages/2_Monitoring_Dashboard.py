"""Monitoring & Benchmark Audit Dashboard for Confidence-Aware RAG Studio.

Ink + Signal visual design: dark-first, calm, high-contrast, Linear/Vercel discipline.
Restyled Altair charts with transparent backgrounds, muted gridlines (6% white), and semantic tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402
from src.store import clear, connect, init_db, load_dataframe  # noqa: E402

st.set_page_config(
    page_title="Telemetry & Audit · Confidence-Aware RAG",
    page_icon="📊",
    layout="wide",
)
ui.inject_css()

INTERACTIVE = "Interactive Studio Queries"
HOTPOT = "HotpotQA Multi-Hop Sweep (Milestone 4)"


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
        tags = [
            r["run_tag"]
            for r in conn.execute(
                "SELECT run_tag, MAX(id) AS last FROM queries WHERE run_tag IS NOT NULL "
                "GROUP BY run_tag ORDER BY last DESC"
            )
        ]
    return [INTERACTIVE, HOTPOT, *tags]


# Altair theme configuration helper
def _style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke="transparent", fill="transparent")
        .configure_axis(
            gridColor="rgba(255, 255, 255, 0.06)",
            gridDash=[2, 2],
            domainColor="rgba(255, 255, 255, 0.12)",
            tickColor="rgba(255, 255, 255, 0.12)",
            labelColor="#8B93A5",
            labelFont="JetBrains Mono, monospace",
            labelFontSize=11,
            titleColor="#8B93A5",
            titleFont="Geist, Inter, sans-serif",
            titleFontSize=12,
            titleFontWeight=600,
        )
        .configure_legend(
            labelColor="#E8EAF0",
            labelFont="Geist, Inter, sans-serif",
            labelFontSize=11,
            titleColor="#8B93A5",
            titleFont="Geist, Inter, sans-serif",
            titleFontSize=11,
            titleFontWeight=600,
        )
        .properties(background="transparent")
    )


with st.sidebar:
    st.html("<div class='sidebar-sec-label'>Navigation</div>")
    st.page_link("app.py", label="RAG Verification Studio", icon=":material/shield:")
    st.page_link("pages/1_How_to_use.py", label="Architecture & Viva Guide", icon=":material/menu_book:")
    st.page_link("pages/2_Monitoring_Dashboard.py", label="Telemetry & Audit Dashboard", icon=":material/bar_chart:")

    st.html("<div class='sidebar-sec-label'>Telemetry Filter</div>")
    scope = st.selectbox(
        "Telemetry stream",
        _scopes(),
        label_visibility="collapsed",
        help="Inspect interactive queries from the studio or a specific benchmark run tag.",
    )
    limit = st.slider("Max records to inspect", 20, 3000, 500, step=20)

ui.hero(
    "Telemetry & M1–M5 Benchmark Audit Dashboard",
    "Real-time SQLite observability across retrieval sufficiency, claim groundedness, latency budget, "
    "and audited SQuAD 2.0 / HotpotQA benchmark evaluations.",
    [
        "Real-Time SQLite Stream",
        "RAGTruth Claim Audit",
        "SQuAD 2.0 Benchmarks",
    ],
)

if scope in (INTERACTIVE, HOTPOT):
    df = _untagged()
    if "_hotpot" in df.columns:
        df = df[df["_hotpot"]] if scope == HOTPOT else df[~df["_hotpot"] & ~df["_empty"]]
        df = df.drop(columns=["_hotpot", "_empty"])
    df = df.tail(limit)
else:
    df = load_dataframe(limit=limit, run_tag=scope)

if not df.empty:
    abstained = df["verdict"].str.startswith("abstain")
    flagged = df["verdict"].eq("flag_low_confidence")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi_card("Logged Queries", f"{len(df):,}", f"Stream: {scope[:22]}", ui.ACCENT)
    with c2:
        ans_cnt = int((~abstained).sum())
        ans_pct = (1.0 - abstained.mean()) * 100
        ui.kpi_card(
            "Verified & Answered",
            f"{ans_cnt:,} ({ans_pct:.0f}%)",
            f"{int(flagged.sum())} low-confidence warnings",
            ui.PASS,
        )
    with c3:
        abs_cnt = int(abstained.sum())
        abs_pct = abstained.mean() * 100
        ui.kpi_card(
            "Protected (Abstained)",
            f"{abs_cnt:,} ({abs_pct:.0f}%)",
            "Refused at Gate A / Gate B",
            ui.FAIL if abs_cnt > 0 else ui.SKIP,
        )
    with c4:
        med_s = df["total_ms"].median() / 1000.0 if "total_ms" in df.columns else 0.0
        comp_mean = df["composite"].mean() if "composite" in df.columns else 0.0
        ui.kpi_card(
            "Median Query Latency",
            f"{med_s:.2f} s",
            f"Mean Composite: {comp_mean:.3f}",
            ui.WARN,
        )

st.html("<div style='height:0.55rem'></div>")

overview, audit_tab, claims_tab, bench, latency, log = st.tabs(
    [
        "Live Overview",
        "M1–M5 Benchmark Audit",
        "RAGTruth Claims",
        "Arm Comparison",
        "Latency Budget",
        "Raw Query Log",
    ]
)

with overview:
    if df.empty:
        st.info("No queries logged for this stream yet. Run a question in the Studio to generate telemetry.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Verdict Distribution")
            short_labels = {
                "answer": "Answered (Verified)",
                "flag_low_confidence": "Answered (Low Confidence)",
                "abstain_no_context": "Abstained (Gate A: No Context)",
                "abstain_ungrounded": "Abstained (Gate B: Ungrounded)",
            }
            v_colors = {
                "Answered (Verified)": ui.PASS,
                "Answered (Low Confidence)": ui.WARN,
                "Abstained (Gate A: No Context)": ui.FAIL,
                "Abstained (Gate B: Ungrounded)": "#fca5a5",
            }
            counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="queries")
            counts["verdict_clean"] = counts["verdict"].map(lambda v: short_labels.get(v, v))

            bar = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=4, height=22)
                .encode(
                    y=alt.Y("verdict_clean:N", title=None, sort="-x"),
                    x=alt.X("queries:Q", title="Query Count"),
                    color=alt.Color(
                        "verdict_clean:N",
                        scale=alt.Scale(
                            domain=list(v_colors.keys()),
                            range=list(v_colors.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=["verdict_clean", "queries"],
                )
                .properties(height=260)
            )
            st.altair_chart(_style_chart(bar), use_container_width=True)

        with right:
            st.markdown("##### Gate & Composite Confidence Trajectory")
            trend_cols = [c for c in ("gate_a_score", "gate_b_score", "composite") if c in df.columns]
            if trend_cols:
                trend = (
                    df.sort_values("id")[trend_cols]
                    .reset_index(drop=True)
                    .reset_index()
                    .rename(columns={"index": "query_num"})
                )
                trend["query_num"] += 1
                trend_long = trend.melt(
                    id_vars=["query_num"],
                    value_vars=trend_cols,
                    var_name="metric",
                    value_name="score",
                )
                metric_labels = {
                    "gate_a_score": "Gate A (Relevance)",
                    "gate_b_score": "Gate B (Faithfulness)",
                    "composite": "Harmonic Composite",
                }
                trend_long["metric_label"] = trend_long["metric"].map(metric_labels)
                color_scale = alt.Scale(
                    domain=["Gate A (Relevance)", "Gate B (Faithfulness)", "Harmonic Composite"],
                    range=[ui.ACCENT, ui.PASS, ui.WARN],
                )

                line = (
                    alt.Chart(trend_long)
                    .mark_line(strokeWidth=2, point=True)
                    .encode(
                        x=alt.X("query_num:Q", title="Sequential Query Number"),
                        y=alt.Y("score:Q", title="Confidence Score", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("metric_label:N", scale=color_scale, title=None),
                        tooltip=["query_num", "metric_label", alt.Tooltip("score:Q", format=".3f")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(_style_chart(line), use_container_width=True)
            else:
                st.caption("No score columns available.")

        st.caption(
            "Operational insight: A sustained drop in Gate A signals corpus drift (users asking questions outside the indexed domain)."
        )

with audit_tab:
    audit_path = REPORTS_DIR / "audit_m5.json"
    if audit_path.exists():
        st.markdown("##### 1. SQuAD 2.0 Single-Hop Benchmark (n = 150 · 75 Answerable / 75 Unanswerable)")
        squad_rows = [
            {
                "Configuration / Arm": "Baseline Control (Gates Off)",
                "Hallucination (Unanswerable)": "100.0% (75/75)",
                "Over-Abstention (Answerable)": "0.0% (0/75)",
                "Answered": "75 / 75",
                "Selective Accuracy (Gold in Answered)": "90.7% (68/75)",
                "Corrected F1": 0.251,
            },
            {
                "Configuration / Arm": "Guarded M2 (Uncalibrated)",
                "Hallucination (Unanswerable)": "4.0% (3/75)",
                "Over-Abstention (Answerable)": "84.0% (63/75)",
                "Answered": "12 / 75",
                "Selective Accuracy (Gold in Answered)": "91.7% (11/12)",
                "Corrected F1": 0.511,
            },
            {
                "Configuration / Arm": "Guarded M3 (Gate B v1 · Passage)",
                "Hallucination (Unanswerable)": "10.7% (8/75)",
                "Over-Abstention (Answerable)": "72.0% (54/75)",
                "Answered": "21 / 75",
                "Selective Accuracy (Gold in Answered)": "95.2% (20/21)",
                "Corrected F1": 0.505,
            },
            {
                "Configuration / Arm": "Guarded M5 (Window-Only)",
                "Hallucination (Unanswerable)": "14.7% (11/75)",
                "Over-Abstention (Answerable)": "50.7% (38/75)",
                "Answered": "37 / 75",
                "Selective Accuracy (Gold in Answered)": "97.3% (36/37)",
                "Corrected F1": 0.543,
            },
            {
                "Configuration / Arm": "Guarded M5 Final (Gate B v2 · Passages + 3-Sent Windows)",
                "Hallucination (Unanswerable)": "13.3% (10/75)",
                "Over-Abstention (Answerable)": "48.0% (36/75)",
                "Answered": "39 / 75",
                "Selective Accuracy (Gold in Answered)": "94.9% (37/39)",
                "Corrected F1": 0.553,
            },
        ]
        st.dataframe(pd.DataFrame(squad_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Key Milestone 5 Breakthrough: Switching Gate B from full-paragraph premises (v1) to passages + overlapping "
            "3-sentence windows (v2) cut over-abstention from 72.0% to 48.0% (paired McNemar p < 0.001) while preserving "
            "94.9% selective accuracy (37/39 correct)."
        )

        st.markdown("##### 2. HotpotQA Multi-Hop Granularity & Guarded Ablation (n = 150)")
        hotpot_rows = [
            {
                "Indexing & Arm": "Sentence Indexing · Baseline",
                "Fact Recall": "58.5% (30.7% full)",
                "Answered": "150 / 150 (100%)",
                "Gold Accuracy": "42.7% (64/150)",
                "Error Rate": "57.3% (86/150)",
                "F1": 0.190,
            },
            {
                "Indexing & Arm": "Sentence Indexing · Guarded",
                "Fact Recall": "58.5% (30.7% full)",
                "Answered": "25 / 150 (16.7%)",
                "Gold Accuracy": "64.0% of answered (16/25)",
                "Error Rate": "36.0% (9/25)",
                "F1": 0.056,
            },
            {
                "Indexing & Arm": "800-Char Chunk Indexing · Baseline",
                "Fact Recall": "82.3% (66.0% full)",
                "Answered": "150 / 150 (100%)",
                "Gold Accuracy": "59.3% (89/150)",
                "Error Rate": "40.7% (61/150)",
                "F1": 0.273,
            },
            {
                "Indexing & Arm": "800-Char Chunk Indexing · Guarded",
                "Fact Recall": "82.3% (66.0% full)",
                "Answered": "33 / 150 (22.0%)",
                "Gold Accuracy": "75.8% of answered (25/33)",
                "Error Rate": "24.2% (8/33)",
                "F1": 0.069,
            },
        ]
        st.dataframe(pd.DataFrame(hotpot_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Audit report `reports/audit_m5.json` not found.")

with claims_tab:
    if df.empty:
        st.info("No claim telemetry available for this selection.")
    else:
        supported = int(
            (df["n_claims"].fillna(0) - df["n_conflicts"].fillna(0) - df["n_baseless"].fillna(0)).sum()
        )
        baseless = int(df["n_baseless"].fillna(0).sum())
        conflicts = int(df["n_conflicts"].fillna(0).sum())
        k1, k2, k3 = st.columns(3)
        with k1:
            ui.kpi_card("Supported Claims", f"{supported:,}", "Entailed by retrieved source", ui.PASS)
        with k2:
            ui.kpi_card("Baseless Information", f"{baseless:,}", "Absent from context", ui.WARN)
        with k3:
            ui.kpi_card("Evident Conflict", f"{conflicts:,}", "Directly contradicts source", ui.FAIL)

        tax_df = pd.DataFrame(
            [
                {"Taxonomy": "Supported", "Count": supported, "color": ui.PASS},
                {"Taxonomy": "Baseless Information", "Count": baseless, "color": ui.WARN},
                {"Taxonomy": "Evident Conflict", "Count": conflicts, "color": ui.FAIL},
            ]
        )
        bar_tax = (
            alt.Chart(tax_df)
            .mark_bar(cornerRadiusEnd=4, size=32)
            .encode(
                x=alt.X("Taxonomy:N", title=None, sort=None),
                y=alt.Y("Count:Q", title="Total Evaluated Claims"),
                color=alt.Color("Taxonomy:N", scale=alt.Scale(range=[ui.PASS, ui.WARN, ui.FAIL]), legend=None),
                tooltip=["Taxonomy", "Count"],
            )
            .properties(height=260)
        )
        st.altair_chart(_style_chart(bar_tax), use_container_width=True)

with bench:
    if not df.empty and "arm" in df.columns and df["arm"].nunique() > 1:
        rows = []
        for arm, sub in df.groupby("arm"):
            ab = sub["verdict"].str.startswith("abstain")
            rows.append(
                {
                    "Arm": arm,
                    "Queries": len(sub),
                    "Answered": int((~ab).sum()),
                    "Abstained": int(ab.sum()),
                    "Mean Composite": round(sub["composite"].mean(), 3),
                    "Median Latency (s)": round(sub["total_ms"].median() / 1000, 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        arm_v_counts = df.groupby(["arm", "verdict"]).size().reset_index(name="count")
        bar_arm = (
            alt.Chart(arm_v_counts)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("arm:N", title="Pipeline Arm"),
                y=alt.Y("count:Q", title="Query Count"),
                color=alt.Color("verdict:N", title="Verdict"),
                tooltip=["arm", "verdict", "count"],
            )
            .properties(height=280)
        )
        st.altair_chart(_style_chart(bar_arm), use_container_width=True)
    else:
        st.info(
            "Current selection contains a single arm. Switch to a dual-arm benchmark tag or inspect "
            "the M1–M5 Benchmark Audit tab for full Guarded vs. Baseline comparisons."
        )

with latency:
    if not df.empty:
        lat_cols = {
            "retrieval_ms": "1 · Vector Retrieval",
            "gate_a_ms": "2 · Gate A (Reranker)",
            "generation_ms": "3 · LLM Generation",
            "gate_b_ms": "4 · Gate B (DeBERTa NLI)",
        }
        present = [c for c in lat_cols if c in df.columns]
        if present:
            med_series = df[present].median().rename(lat_cols).reset_index()
            med_series.columns = ["Stage", "Median_ms"]
            lat_colors = [ui.INFO, ui.ACCENT, ui.WARN, ui.PASS]
            bar_lat = (
                alt.Chart(med_series)
                .mark_bar(cornerRadiusEnd=4, height=22)
                .encode(
                    y=alt.Y("Stage:N", title=None, sort=None),
                    x=alt.X("Median_ms:Q", title="Median Wall Time (ms)"),
                    color=alt.Color("Stage:N", scale=alt.Scale(range=lat_colors), legend=None),
                    tooltip=["Stage", alt.Tooltip("Median_ms:Q", format=".1f")],
                )
                .properties(height=240)
            )
            st.altair_chart(_style_chart(bar_lat), use_container_width=True)
            st.caption(
                "Both confidence gates combined (~344 ms warm median) represent a fraction of LLM generation latency."
            )

with log:
    if not df.empty:
        show = [
            c
            for c in (
                "id",
                "ts",
                "question",
                "verdict",
                "composite",
                "gate_a_score",
                "gate_b_score",
                "n_claims",
                "n_conflicts",
                "n_baseless",
                "total_ms",
                "run_tag",
                "arm",
            )
            if c in df.columns
        ]
        st.dataframe(
            df[show].sort_values("id", ascending=False),
            use_container_width=True,
            height=380,
            hide_index=True,
        )
        st.download_button(
            "Export Telemetry Selection as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="confidence_rag_telemetry.csv",
            mime="text/csv",
        )

with st.sidebar.expander("Danger Zone (Reset Database)"):
    st.caption("Permanently clears all query records in `data/runs.sqlite3`.")
    sure = st.checkbox("I confirm deleting all logged telemetry")
    if st.button("Clear Telemetry Database", type="secondary", disabled=not sure):
        clear()
        st.rerun()
