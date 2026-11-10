"""Monitoring & Benchmark Audit Dashboard for the Confidence-Aware RAG Studio.

Reads both live SQLite query telemetry (`data/runs.sqlite3`) and saved M1–M5
evaluation artifacts (`reports/audit_m5.json`, `reports/latency.json`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ui  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402
from src.store import clear, connect, init_db, load_dataframe  # noqa: E402

st.set_page_config(page_title="Telemetry & Audit · Confidence-Aware RAG", page_icon="📊", layout="wide")
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


with st.sidebar:
    st.markdown("### 🛡️ Navigation")
    st.page_link("app.py", label="RAG Verification Studio", icon="🛡️")
    st.page_link("pages/1_How_to_use.py", label="Architecture & Viva Guide", icon="📖")
    st.page_link("pages/2_Monitoring_Dashboard.py", label="Telemetry & Audit Dashboard", icon="📊")
    st.divider()
    st.markdown("### 🔎 Telemetry Filter")
    scope = st.selectbox(
        "Telemetry stream",
        _scopes(),
        help="Inspect interactive queries from the studio or a specific benchmark run tag.",
    )
    limit = st.slider("Max rows to load", 20, 3000, 500, step=20)

ui.hero(
    "Telemetry & M1–M5 Benchmark Audit",
    "Real-time SQLite observability across retrieval sufficiency, NLI claim taxonomy, stage latency, "
    "and audited SQuAD 2.0 / HotpotQA benchmark evaluations.",
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
        ui.kpi_card("Logged Queries", f"{len(df):,}", f"Stream: {scope[:26]}", ui.BRAND)
    with c2:
        ans_cnt = int((~abstained).sum())
        ans_pct = (1.0 - abstained.mean()) * 100
        ui.kpi_card(
            "Verified & Answered",
            f"{ans_cnt:,} ({ans_pct:.0f}%)",
            f"{int(flagged.sum())} flagged low-confidence",
            ui.PASS,
        )
    with c3:
        abs_cnt = int(abstained.sum())
        abs_pct = abstained.mean() * 100
        ui.kpi_card(
            "Abstained (Protected)",
            f"{abs_cnt:,} ({abs_pct:.0f}%)",
            "Blocked by Gate A / Gate B / Self-Refusal",
            ui.FAIL if abs_cnt > 0 else ui.SKIP,
        )
    with c4:
        med_s = df["total_ms"].median() / 1000.0
        comp_mean = df["composite"].mean() if "composite" in df.columns else 0.0
        ui.kpi_card(
            "Median Query Latency",
            f"{med_s:.2f} s",
            f"Mean Composite Score: {comp_mean:.3f}",
            ui.WARN,
        )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

overview, audit_tab, claims_tab, bench, latency, log = st.tabs(
    [
        "📈 Live Overview",
        "🏆 M1–M5 Benchmark Audit",
        "🔬 RAGTruth Claims",
        "⚖️ Arm Comparison",
        "⏱️ Latency Budget",
        "🗂️ Raw Query Log",
    ]
)

with overview:
    if df.empty:
        st.info("No queries logged for this filter yet. Run a sample question in the Studio or switch stream.")
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
            counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="queries")
            counts["verdict"] = counts["verdict"].map(lambda v: short_labels.get(v, v))
            st.bar_chart(counts.set_index("verdict"), height=290, horizontal=True)
        with right:
            st.markdown("##### Gate & Composite Confidence Trajectory")
            trend_cols = [c for c in ("gate_a_score", "gate_b_score", "composite") if c in df.columns]
            trend = df.sort_values("id")[trend_cols].rename(
                columns={"gate_a_score": "Gate A", "gate_b_score": "Gate B", "composite": "Composite"}
            )
            trend.index = pd.RangeIndex(1, len(trend) + 1, name="Query #")
            st.line_chart(trend, height=290, x_label="Query #")
        st.caption(
            "Operational insight: A sustained drop in Gate A indicates corpus drift (users asking about topics "
            "absent from the indexed documents)."
        )

with audit_tab:
    audit_path = REPORTS_DIR / "audit_m5.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        st.markdown("##### 1. SQuAD 2.0 Single-Hop Benchmark (n = 150 · 75 Answerable / 75 Unanswerable)")
        squad_rows = [
            {
                "Configuration / Arm": "Baseline Control (Gates Off)",
                "Hallucination Rate (Unanswerable)": "100.0% (75/75)",
                "Over-Abstention (Answerable)": "0.0% (0/75)",
                "Answered (Answerable)": "75 / 75",
                "Selective Accuracy (Gold in Answered)": "90.7% (68/75)",
                "Corrected F1": 0.251,
            },
            {
                "Configuration / Arm": "Guarded M2 (Uncalibrated)",
                "Hallucination Rate (Unanswerable)": "4.0% (3/75)",
                "Over-Abstention (Answerable)": "84.0% (63/75)",
                "Answered (Answerable)": "12 / 75",
                "Selective Accuracy (Gold in Answered)": "91.7% (11/12)",
                "Corrected F1": 0.511,
            },
            {
                "Configuration / Arm": "Guarded M3 (Calibrated Gate B v1 · Passage)",
                "Hallucination Rate (Unanswerable)": "10.7% (8/75)",
                "Over-Abstention (Answerable)": "72.0% (54/75)",
                "Answered (Answerable)": "21 / 75",
                "Selective Accuracy (Gold in Answered)": "95.2% (20/21)",
                "Corrected F1": 0.505,
            },
            {
                "Configuration / Arm": "Guarded M5 (Gate B Window-Only)",
                "Hallucination Rate (Unanswerable)": "14.7% (11/75)",
                "Over-Abstention (Answerable)": "50.7% (38/75)",
                "Answered (Answerable)": "37 / 75",
                "Selective Accuracy (Gold in Answered)": "97.3% (36/37)",
                "Corrected F1": 0.543,
            },
            {
                "Configuration / Arm": "Guarded M5 Final (Gate B v2 · Passages + 3-Sent Windows)",
                "Hallucination Rate (Unanswerable)": "14.7% (11/75)",
                "Over-Abstention (Answerable)": "48.0% (36/75)",
                "Answered (Answerable)": "39 / 75",
                "Selective Accuracy (Gold in Answered)": "97.4% (38/39)",
                "Corrected F1": 0.553,
            },
        ]
        st.dataframe(pd.DataFrame(squad_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Key M5 Result: Switching Gate B from full-paragraph premises (v1) to passages + overlapping 3-sentence "
            "windows (v2) cuts over-abstention from 70.7% to 48.0% (paired McNemar p < 0.001) while maintaining "
            "97.4% selective accuracy (38/39 correct)."
        )

        st.markdown("##### 2. HotpotQA Multi-Hop Granularity & Guarded Ablation (n = 150)")
        hotpot_rows = [
            {
                "Indexing & Arm": "Sentence Indexing · Baseline",
                "Supporting-Fact Recall": "58.5% (30.7% full)",
                "Answered": "150 / 150 (100%)",
                "Gold Containment Accuracy": "42.7% (64/150)",
                "Error Rate Among Answered": "57.3% (86/150)",
                "Corrected F1": 0.190,
            },
            {
                "Indexing & Arm": "Sentence Indexing · Guarded",
                "Supporting-Fact Recall": "58.5% (30.7% full)",
                "Answered": "25 / 150 (16.7%)",
                "Gold Containment Accuracy": "64.0% of answered (16/25)",
                "Error Rate Among Answered": "36.0% (9/25)",
                "Corrected F1": 0.056,
            },
            {
                "Indexing & Arm": "800-Char Chunk Indexing · Baseline",
                "Supporting-Fact Recall": "82.3% (66.0% full)",
                "Answered": "150 / 150 (100%)",
                "Gold Containment Accuracy": "59.3% (89/150)",
                "Error Rate Among Answered": "40.7% (61/150)",
                "Corrected F1": 0.273,
            },
            {
                "Indexing & Arm": "800-Char Chunk Indexing · Guarded",
                "Supporting-Fact Recall": "82.3% (66.0% full)",
                "Answered": "33 / 150 (22.0%)",
                "Gold Containment Accuracy": "75.8% of answered (25/33)",
                "Error Rate Among Answered": "24.2% (8/33)",
                "Corrected F1": 0.069,
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
            ui.kpi_card("Supported Claims", f"{supported:,}", "Entailed by retrieved passage", ui.PASS)
        with k2:
            ui.kpi_card("Baseless Information", f"{baseless:,}", "Absent from retrieved context", ui.WARN)
        with k3:
            ui.kpi_card("Evident Conflict", f"{conflicts:,}", "Contradicts retrieved source", ui.FAIL)

        tax = pd.Series(
            {"Supported": supported, "Baseless Information": baseless, "Evident Conflict": conflicts},
            name="claims",
        )
        st.bar_chart(tax, height=260)

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
        st.bar_chart(df.groupby(["arm", "verdict"]).size().unstack(fill_value=0), height=280)
    else:
        st.info(
            "Current selection contains a single arm. Switch to a dual-arm benchmark tag or inspect "
            "the **🏆 M1–M5 Benchmark Audit** tab for full Guarded vs. Baseline comparisons."
        )

with latency:
    if not df.empty:
        lat_cols = {
            "retrieval_ms": "1 · Retrieval",
            "gate_a_ms": "2 · Gate A (Reranker)",
            "generation_ms": "3 · LLM Generation",
            "gate_b_ms": "4 · Gate B (DeBERTa NLI)",
        }
        present = [c for c in lat_cols if c in df.columns]
        med = df[present].median().rename(lat_cols).round(1)
        st.bar_chart(med.rename("Median ms"), height=280, horizontal=True)
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
            "📥 Export Telemetry Selection as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="confidence_rag_telemetry.csv",
            mime="text/csv",
        )

with st.sidebar.expander("⚠️ Danger Zone"):
    st.caption("Permanently clears all rows in `data/runs.sqlite3`.")
    sure = st.checkbox("I confirm deleting all logged telemetry")
    if st.button("Clear Telemetry Database", type="secondary", disabled=not sure):
        clear()
        st.rerun()
