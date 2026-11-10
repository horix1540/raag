"""Ink + Signal Design System for Confidence-Aware RAG Studio.

High-contrast, dark-first, calm presentation layer following Linear / Vercel discipline.
Every HTML helper is a small pure function that returns an escaped string for safe rendering.

WCAG AA Contrast Audit Table:
--------------------------------------------------------------------------------------
Surface               Foreground           Hex Code    Luminance   Contrast    Compliance
--------------------------------------------------------------------------------------
Page Bg (#0B0D12)     Text Primary         #E8EAF0     0.814       14.3:1      WCAG AAA (>7:1)
Page Bg (#0B0D12)     Text Muted           #8B93A5     0.289        5.8:1      WCAG AA  (>4.5:1)
Page Bg (#0B0D12)     Accent (Primary)     #7C8CFF     0.303        6.0:1      WCAG AA  (>4.5:1)
Page Bg (#0B0D12)     Pass (Emerald)       #34D399     0.522        9.8:1      WCAG AAA (>7:1)
Page Bg (#0B0D12)     Fail (Rose)          #F87171     0.301        6.0:1      WCAG AA  (>4.5:1)
Page Bg (#0B0D12)     Warn (Amber)         #FBBF24     0.575       10.7:1      WCAG AAA (>7:1)
Page Bg (#0B0D12)     Info (Sky)           #60A5FA     0.379        7.3:1      WCAG AAA (>7:1)
Card (#11141B)        Text Primary         #E8EAF0     0.814       13.7:1      WCAG AAA (>7:1)
Card (#11141B)        Text Muted           #8B93A5     0.289        5.5:1      WCAG AA  (>4.5:1)
Card (#11141B)        Pass (Emerald)       #34D399     0.522        9.4:1      WCAG AAA (>7:1)
Card (#11141B)        Fail (Rose)          #F87171     0.301        5.6:1      WCAG AA  (>4.5:1)
Card (#11141B)        Warn (Amber)         #FBBF24     0.575       10.1:1      WCAG AAA (>7:1)
Card (#11141B)        Info (Sky)           #60A5FA     0.379        7.0:1      WCAG AAA (>7:1)
--------------------------------------------------------------------------------------
All surfaces clear WCAG AA (>= 4.5:1), with primary copy achieving WCAG AAA (>= 7:1).
"""

from __future__ import annotations

import re
from html import escape

import streamlit as st

# Ink + Signal Semantic Tokens (Hex)
PASS = "#34D399"       # Emerald 400
FAIL = "#F87171"       # Rose 400
WARN = "#FBBF24"       # Amber 400
INFO = "#60A5FA"       # Blue 400
ACCENT = "#7C8CFF"     # Indigo/Periwinkle 400
BRAND = "#7C8CFF"      # Alias for accent (backwards compatibility)
SKIP = "#64748B"       # Slate 500
IDLE = "#5B6376"       # Slate 600

# 12% alpha background tints
PASS_TINT = "rgba(52, 211, 153, 0.12)"
FAIL_TINT = "rgba(248, 113, 113, 0.12)"
WARN_TINT = "rgba(251, 191, 36, 0.12)"
INFO_TINT = "rgba(96, 165, 250, 0.12)"
ACCENT_TINT = "rgba(124, 140, 255, 0.12)"
SKIP_TINT = "rgba(100, 116, 139, 0.12)"

# 35% alpha glow variants
PASS_GLOW = "rgba(52, 211, 153, 0.35)"
FAIL_GLOW = "rgba(248, 113, 113, 0.35)"
WARN_GLOW = "rgba(251, 191, 36, 0.35)"
INFO_GLOW = "rgba(96, 165, 250, 0.35)"
ACCENT_GLOW = "rgba(124, 140, 255, 0.35)"

VERDICTS = {
    "answer": (
        "Verified & Answered",
        "Both Gate A (retrieval relevance) and Gate B (claim-level NLI entailment) passed with high composite confidence.",
        PASS,
        PASS_TINT,
        PASS_GLOW,
    ),
    "flag_low_confidence": (
        "Answered · Low Confidence Warning",
        "Both gates cleared individual thresholds, but the weighted harmonic composite fell below the warning line.",
        WARN,
        WARN_TINT,
        WARN_GLOW,
    ),
    "abstain_no_context": (
        "Abstained · Insufficient Retrieval Context (Gate A)",
        "Retrieved passages did not meet the Gate A cross-encoder relevance threshold. Generation skipped (0 tokens spent).",
        FAIL,
        FAIL_TINT,
        FAIL_GLOW,
    ),
    "abstain_ungrounded": (
        "Abstained · Ungrounded or Contradicted Answer (Gate B)",
        "An answer was drafted, but Gate B NLI verification detected unsupported or contradicted claims and withheld it.",
        FAIL,
        FAIL_TINT,
        FAIL_GLOW,
    ),
}

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #0B0D12;
    --surface-1: #11141B;
    --surface-2: #171B24;
    --surface-3: #1E2330;
    --border: rgba(255, 255, 255, 0.07);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text: #E8EAF0;
    --text-muted: #8B93A5;
    --text-faint: #5B6376;
    --accent: #7C8CFF;
    --pass: #34D399;
    --fail: #F87171;
    --warn: #FBBF24;
    --info: #60A5FA;
    --skip: #64748B;
    --pass-tint: rgba(52, 211, 153, 0.12);
    --fail-tint: rgba(248, 113, 113, 0.12);
    --warn-tint: rgba(251, 191, 36, 0.12);
    --info-tint: rgba(96, 165, 250, 0.12);
    --accent-tint: rgba(124, 140, 255, 0.12);
    --skip-tint: rgba(100, 116, 139, 0.12);
    --pass-glow: rgba(52, 211, 153, 0.35);
    --fail-glow: rgba(248, 113, 113, 0.35);
    --warn-glow: rgba(251, 191, 36, 0.35);
    --accent-glow: rgba(124, 140, 255, 0.35);
}

/* Base resets & typography */
html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
}

/* Tabular numerals for mono metrics */
.mono-num, [data-testid="stMetricValue"], [data-testid="stDataFrame"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
}

/* Chrome suppression */
header[data-testid="stHeader"] {
    background: transparent !important;
    visibility: hidden !important;
    height: 0 !important;
}
#MainMenu, footer, [data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
}
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1320px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--surface-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem !important;
}
.sidebar-sec-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-faint);
    margin: 1.1rem 0 0.35rem 0;
}

/* Native widget restyling: Inputs */
.stTextInput > div > div {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-tint) !important;
}
.stSelectbox > div > div {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Native widget restyling: Buttons */
.stButton > button {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: transform 0.12s ease, background-color 0.12s ease, border-color 0.12s ease;
}
.stButton > button:hover {
    background-color: var(--surface-3) !important;
    border-color: var(--border-strong) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: scale(0.98);
}
.stButton > button[kind="primary"] {
    background-color: var(--accent) !important;
    color: #0B0D12 !important;
    font-weight: 600 !important;
    border: 1px solid var(--accent) !important;
    box-shadow: 0 1px 8px var(--accent-tint);
}
.stButton > button[kind="primary"]:hover {
    background-color: #929fff !important;
    box-shadow: 0 2px 14px var(--accent-glow);
}

/* Native widget restyling: Expanders */
[data-testid="stExpander"] {
    background-color: var(--surface-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] details summary {
    color: var(--text-muted) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details summary:hover {
    color: var(--text) !important;
}

/* Native widget restyling: Tabs (Underline pattern) */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 1.5rem !important;
    padding-bottom: 0.15rem !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 0.2rem !important;
    transition: color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Focus outline accessibility */
a:focus-visible, button:focus-visible, input:focus-visible, [tabindex="0"]:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}

/* Top bar wordmark and live status */
.ink-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 0.6rem 0 1rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.ink-wordmark {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
}
.ink-wordmark-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text);
}
.ink-wordmark-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: var(--accent-tint);
    border: 1px solid rgba(124, 140, 255, 0.25);
    color: var(--accent);
}
.ink-status-chips {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.ink-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: var(--text-muted);
    background: var(--surface-1);
    border: 1px solid var(--border);
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
}
.ink-status-chip strong {
    color: var(--text);
    font-weight: 600;
}
.ink-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--pass);
}

/* Command Bar query input container */
.ink-command-bar {
    position: relative;
    margin: 0.25rem 0 0.5rem 0;
}
.ink-key-hint {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    background: var(--surface-3);
    border: 1px solid var(--border-strong);
    color: var(--text-muted);
    border-radius: 4px;
    user-select: none;
}

/* Diagnostic pill row */
.ink-samples-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.35rem 0 1.25rem 0;
}
.ink-samples-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-faint);
    margin-right: 0.25rem;
}

/* Horizontal Timeline: 5 connected nodes */
.ink-timeline {
    position: relative;
    margin: 1.1rem 0 1.5rem 0;
    padding: 1rem 1.25rem;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
}
.ink-timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-bottom: 0.9rem;
}
.ink-timeline-nodes {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    position: relative;
    gap: 0.5rem;
}
.ink-timeline-track {
    position: absolute;
    top: 14px;
    left: 20px;
    right: 20px;
    height: 2px;
    background: var(--surface-3);
    z-index: 1;
}
.ink-node {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    flex: 1 1 0;
    min-width: 0;
}
.ink-node-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    background: var(--surface-1);
    border: 2px solid var(--border);
    color: var(--text-faint);
    margin-bottom: 0.45rem;
    transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1);
}
.ink-node-circle.pass {
    background: var(--pass-tint);
    border-color: var(--pass);
    color: var(--pass);
}
.ink-node-circle.fail {
    background: var(--fail-tint);
    border-color: var(--fail);
    color: var(--fail);
}
.ink-node-circle.warn {
    background: var(--warn-tint);
    border-color: var(--warn);
    color: var(--warn);
}
.ink-node-circle.skip {
    background: var(--skip-tint);
    border-color: var(--skip);
    color: var(--skip);
}
.ink-node-circle.running {
    background: var(--accent-tint);
    border-color: var(--accent);
    color: var(--accent);
    animation: pulseRing 1.4s infinite cubic-bezier(0.22, 1, 0.36, 1);
}
.ink-node-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.15rem;
    white-space: nowrap;
}
.ink-node-latency {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    color: var(--text-muted);
}
.ink-node-result {
    font-size: 0.68rem;
    color: var(--text-faint);
    margin-top: 0.1rem;
    max-width: 130px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

@keyframes pulseRing {
    0% {
        box-shadow: 0 0 0 0 rgba(124, 140, 255, 0.6);
    }
    70% {
        box-shadow: 0 0 0 8px rgba(124, 140, 255, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(124, 140, 255, 0);
    }
}

/* Verdict banner */
.ink-verdict {
    border-radius: 10px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 1rem;
    background: var(--surface-1);
    border: 1px solid var(--border);
    animation: slideInGlow 0.45s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
.ink-verdict-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.02rem;
    font-weight: 700;
}
.ink-verdict-plain {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.3rem 0 0.2rem 0;
    line-height: 1.45;
}
.ink-verdict-expl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: var(--text-faint);
    margin-top: 0.3rem;
}

@keyframes slideInGlow {
    0% {
        opacity: 0;
        transform: translateY(10px);
        box-shadow: 0 0 18px var(--glow-color, rgba(124, 140, 255, 0.35));
    }
    60% {
        opacity: 1;
        transform: translateY(0);
        box-shadow: 0 0 28px var(--glow-color, rgba(124, 140, 255, 0.35));
    }
    100% {
        opacity: 1;
        transform: translateY(0);
        box-shadow: none;
    }
}

/* System Response Card */
.ink-card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 1rem;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.ink-card:hover {
    border-color: var(--border-strong);
    background-color: var(--surface-2);
}
.ink-card-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-bottom: 0.75rem;
}
.ink-card-body {
    font-size: 0.94rem;
    line-height: 1.68;
    color: var(--text);
    white-space: pre-wrap;
}

/* Inline Citation Pill */
.cr-cite {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.06rem 0.44rem;
    margin: 0 0.14rem;
    border-radius: 5px;
    background: var(--accent-tint);
    color: var(--accent);
    border: 1px solid rgba(124, 140, 255, 0.28);
    vertical-align: baseline;
}

/* Highlighted span in passages */
.cr-hl {
    background: var(--pass-tint);
    border-bottom: 2px solid var(--pass);
    padding: 0.06rem 0.22rem;
    border-radius: 4px;
    color: var(--text);
}

/* Verification Inspector (Sticky right rail) */
.ink-inspector {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    position: sticky;
    top: 1rem;
    margin-bottom: 1rem;
}
.ink-inspector-hdr {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-faint);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Gauge item */
.ink-gauge {
    margin-bottom: 1rem;
}
.ink-gauge-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.ink-gauge-lbl {
    font-size: 0.81rem;
    font-weight: 600;
    color: var(--text);
}
.ink-gauge-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.77rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.ink-gauge-track {
    position: relative;
    height: 8px;
    border-radius: 999px;
    background: var(--surface-3);
    margin: 0.35rem 0;
}
.ink-gauge-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.ink-gauge-tick {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    border-radius: 1px;
    background: #ffffff;
    box-shadow: 0 0 0 1.5px rgba(11, 13, 18, 0.9);
}
.ink-gauge-desc {
    font-size: 0.72rem;
    color: var(--text-faint);
    line-height: 1.35;
}

/* Latency Breakdown Bar */
.ink-latency-bar {
    position: relative;
    height: 8px;
    border-radius: 999px;
    background: var(--surface-3);
    display: flex;
    overflow: hidden;
    margin: 0.45rem 0 0.55rem 0;
}
.ink-latency-seg {
    height: 100%;
}
.ink-latency-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
}
.ink-latency-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.ink-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

/* Chip pill */
.cr-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    font-weight: 600;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    border: 1px solid currentColor;
}

/* Claim verification row */
.ink-claim-row {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s ease;
}
.ink-claim-row:hover {
    border-color: var(--border-strong);
}
.ink-claim-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.ink-claim-text {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 0.45rem;
    line-height: 1.45;
}
.ink-claim-segbar {
    display: flex;
    height: 6px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--surface-3);
    margin-bottom: 0.35rem;
}
.ink-claim-stats {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.71rem;
    color: var(--text-muted);
}

/* KPI stat card */
.ink-kpi {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.ink-kpi:hover {
    border-color: var(--border-strong);
    background-color: var(--surface-2);
}
.ink-kpi-lbl {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
}
.ink-kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0.15rem 0 0.1rem 0;
}
.ink-kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-delay: -1ms !important;
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        background-attachment: initial !important;
        scroll-behavior: auto !important;
        transition-duration: 0ms !important;
        transition-delay: 0ms !important;
    }
    .ink-node-circle.running {
        animation: none !important;
    }
    .ink-verdict {
        animation: none !important;
    }
}
</style>
"""


def inject_css() -> None:
    """Inject Ink + Signal design tokens, layout styles, and animations."""
    st.html(_CSS)


# --- Inline SVG Helpers (16px, 1.5 stroke, Lucide-style) ---

def icon_svg(name: str, size: int = 16, stroke_width: float = 1.5, color: str = "currentColor") -> str:
    """Return a clean inline SVG icon (16px default, 1.5 stroke, Lucide style)."""
    icons = {
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
        "check": '<polyline points="20 6 9 17 4 12"/>',
        "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
        "alert-triangle": (
            '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
        ),
        "alert-circle": (
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
        ),
        "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
        "database": (
            '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
            '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
        ),
        "cpu": (
            '<rect x="4" y="4" width="16" height="16" rx="2"/>'
            '<rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/>'
            '<line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/>'
            '<line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/>'
            '<line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/>'
            '<line x1="1" y1="14" x2="4" y2="14"/>'
        ),
        "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
        "sliders": (
            '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
            '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
            '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
            '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
            '<line x1="17" y1="16" x2="23" y2="16"/>'
        ),
        "book-open": (
            '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
            '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
        ),
        "bar-chart": (
            '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="16"/>'
        ),
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    }
    body = icons.get(name, '<circle cx="12" cy="12" r="8"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;flex-shrink:0;">{body}</svg>'
    )


# --- Top Bar ---

def top_bar(
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    scorer: str = "nli",
    n_passages: int = 6,
    collection: str = "cornerstone",
    arm: str = "Guarded (2-Gate)",
) -> None:
    """Render the slim top bar with product wordmark and live telemetry status chips."""
    arm_dot_col = PASS if "Guarded" in arm else (WARN if "Prompt-Only" in arm else FAIL)
    st.html(
        f"<div class='ink-topbar'>"
        f"<div class='ink-wordmark'>"
        f"{icon_svg('shield-check', 20, 1.6, ACCENT)}"
        f"<span class='ink-wordmark-title'>Confidence-Aware RAG</span>"
        f"<span class='ink-wordmark-badge'>STUDIO</span>"
        f"</div>"
        f"<div class='ink-status-chips'>"
        f"<span class='ink-status-chip'>{icon_svg('cpu', 13, 1.5, 'var(--text-faint)')} {escape(provider)} · <strong>{escape(model)}</strong></span>"
        f"<span class='ink-status-chip'>{icon_svg('sliders', 13, 1.5, 'var(--text-faint)')} verifier <strong>{escape(scorer)}</strong></span>"
        f"<span class='ink-status-chip'>{icon_svg('database', 13, 1.5, 'var(--text-faint)')} <strong>{n_passages}</strong> passages · {escape(collection)}</span>"
        f"<span class='ink-status-chip'><span class='ink-status-dot' style='background:{arm_dot_col}'></span> <strong>{escape(arm)}</strong></span>"
        f"</div>"
        f"</div>"
    )


# --- Horizontal Pipeline Timeline ---

def pipeline_timeline_html(
    result=None,
    baseline: bool = False,
    prompt_only: bool = False,
) -> str:
    """Format the 5-stage horizontal timeline as a clean HTML string."""
    stages = [
        ("Retrieve", "MiniLM Top-K"),
        ("Gate A", "Cross-Encoder"),
        ("Generate", "Contextual LLM"),
        ("Gate B", "DeBERTa NLI"),
        ("Decide", "Harmonic Verdict"),
    ]
    unforced = baseline or prompt_only

    if result is None:
        # Idle state
        node_states = [("idle", "1", "awaiting query", "—")] * 5
    else:
        rep = result.confidence
        a, b = rep.gate_a, rep.gate_b
        gen = result.generation
        self_refused = bool(gen and gen.self_reported_insufficient)

        # Stage 1: Retrieve
        s1_lat = f"{result.retrieval_ms:.0f}ms"
        s1_res = f"{len(result.retrieval.chunks)} chunks"

        # Stage 2: Gate A
        s2_lat = f"{a.latency_ms:.0f}ms"
        s2_cls = "pass" if a.passed else "fail"
        s2_res = f"{a.score:.2f} passed" if a.passed else f"{a.score:.2f} failed"
        if unforced:
            s2_cls = "warn"
            s2_res = f"{a.score:.2f} unforced"

        # Stage 3: Generate
        if gen is None:
            s3_cls, s3_lat, s3_res = "skip", "0ms", "skipped (0 tok)"
        elif self_refused:
            s3_cls, s3_lat, s3_res = "warn", f"{gen.latency_ms:.0f}ms", "self-refused"
        else:
            s3_cls, s3_lat, s3_res = "pass", f"{gen.latency_ms:.0f}ms", "generated"

        # Stage 4: Gate B
        if b is None:
            s4_cls, s4_lat, s4_res = "skip", "—", "not run"
        else:
            s4_lat = f"{b.latency_ms:.0f}ms"
            s4_cls = "pass" if b.passed else "fail"
            s4_res = f"{b.score:.2f} passed" if b.passed else f"{b.score:.2f} failed"
            if unforced:
                s4_cls = "warn"
                s4_res = f"{b.score:.2f} unforced"

        # Stage 5: Decide
        s5_lat = f"{result.total_ms / 1000:.2f}s"
        s5_cls = "pass" if not rep.abstained else "fail"
        s5_res = f"{rep.composite:.2f} answered" if not rep.abstained else "abstained"
        if unforced and not rep.abstained:
            s5_res = f"{rep.composite:.2f} (unforced)"

        node_states = [
            ("pass", "✓", s1_res, s1_lat),
            (s2_cls, "✓" if a.passed else "✕", s2_res, s2_lat),
            (s3_cls, "—" if gen is None else "✓", s3_res, s3_lat),
            (s4_cls, "✓" if (b and b.passed) else ("—" if b is None else "✕"), s4_res, s4_lat),
            (s5_cls, "✓" if not rep.abstained else "✕", s5_res, s5_lat),
        ]

    mode_label = (
        "BASELINE (GATES OFF)"
        if baseline
        else ("PROMPT-ONLY (H3 ABLATION)" if prompt_only else "GUARDED (2-GATE PIPELINE)")
    )

    nodes_html = []
    for idx, ((name, _sub), (state_cls, icon_txt, res_txt, lat_txt)) in enumerate(
        zip(stages, node_states, strict=True), 1
    ):
        nodes_html.append(
            f"<div class='ink-node'>"
            f"<div class='ink-node-circle {state_cls}'>{escape(icon_txt)}</div>"
            f"<div class='ink-node-name'>{idx}. {escape(name)}</div>"
            f"<div class='ink-node-latency'>{escape(lat_txt)}</div>"
            f"<div class='ink-node-result' title='{escape(res_txt)}'>{escape(res_txt)}</div>"
            f"</div>"
        )

    return (
        f"<div class='ink-timeline'>"
        f"<div class='ink-timeline-header'>"
        f"<span>{icon_svg('layers', 14, 1.5, 'var(--text-faint)')} Pipeline Execution Timeline</span>"
        f"<span>{escape(mode_label)}</span>"
        f"</div>"
        f"<div class='ink-timeline-nodes'>"
        f"<div class='ink-timeline-track'></div>"
        f"{''.join(nodes_html)}"
        f"</div>"
        f"</div>"
    )


def pipeline_strip(result=None, baseline: bool = False, prompt_only: bool = False) -> None:
    """Render the horizontal pipeline timeline."""
    st.html(pipeline_timeline_html(result, baseline=baseline, prompt_only=prompt_only))


# --- Verdict Banner ---

def verdict_banner_html(verdict: str, explanation: str) -> str:
    """Format the compact verdict banner as an HTML string."""
    title, plain, colour, tint, glow = VERDICTS.get(
        verdict,
        (verdict, "", SKIP, SKIP_TINT, "transparent"),
    )
    icon_name = "shield-check" if verdict == "answer" else ("alert-triangle" if "warn" in verdict else "alert-circle")
    svg = icon_svg(icon_name, 18, 1.6, colour)

    return (
        f"<div class='ink-verdict' style='border-left:4px solid {colour};background:{tint};--glow-color:{glow}'>"
        f"<div class='ink-verdict-title' style='color:{colour}'>"
        f"{svg}<span>{escape(title)}</span>"
        f"</div>"
        f"<div class='ink-verdict-plain'>{escape(plain)}</div>"
        f"<div class='ink-verdict-expl'>{escape(explanation)}</div>"
        f"</div>"
    )


def verdict_banner(verdict: str, explanation: str) -> None:
    """Render the compact verdict banner."""
    st.html(verdict_banner_html(verdict, explanation))


# --- Answer Box with Citation Badges ---

def format_answer_html(text: str) -> str:
    """Format answer text with inline [Ref 1] citation pills."""
    if not text:
        return ""
    cleaned = escape(text.strip())
    # Merge trailing bracket citation tags onto preceding sentence
    cleaned = re.sub(r"\n+\s*(\[(?:\d+(?:\s*,\s*\d+)*)\])\s*$", r" \1", cleaned)
    # Replace citations with sleek inline pill
    cleaned = _CITATION_RE.sub(r"<span class='cr-cite'>Ref \1</span>", cleaned)
    return cleaned


def highlight_passage_html(passage: str, claims: list[str]) -> str:
    """Highlight sentences in a passage that support verified claims."""
    if not claims:
        return escape(passage)
    claim_words = {
        w.lower()
        for c in claims
        for w in re.findall(r"[A-Za-z0-9]{5,}", c)
        if w.lower() not in {"which", "where", "their", "there", "these", "those", "about", "after", "before"}
    }
    if not claim_words:
        return escape(passage)
    sents = re.split(r"(?<=[.!?])\s+", passage.strip())
    out = []
    for s in sents:
        s_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]{5,}", s)}
        if len(s_words & claim_words) >= 2:
            out.append(f"<span class='cr-hl'>{escape(s)}</span>")
        else:
            out.append(escape(s))
    return " ".join(out)


# --- Verification Inspector (Gauges + Latency Stack) ---

def score_bar_html(label: str, value: float | None, threshold: float | None, caption: str = "") -> str:
    """Format a single 0-1 horizontal gauge with threshold tick."""
    if value is None:
        return (
            f"<div class='ink-gauge'>"
            f"<div class='ink-gauge-top'>"
            f"<span class='ink-gauge-lbl'>{escape(label)}</span>"
            f"<span class='ink-gauge-val'>{chip('SKIPPED', SKIP)}</span>"
            f"</div>"
            f"<div class='ink-gauge-track'></div>"
            f"<div class='ink-gauge-desc'>{escape(caption)}</div>"
            f"</div>"
        )
    v = max(0.0, min(1.0, float(value)))
    ok = threshold is None or value >= threshold
    colour = PASS if ok else FAIL
    status_txt = "PASS" if ok else "BELOW"
    mark = (
        f"<div class='ink-gauge-tick' style='left:calc({threshold * 100:.1f}% - 1px)' "
        f"title='Threshold {threshold:.2f}'></div>"
        if threshold is not None
        else ""
    )
    thresh_txt = f" / {threshold:.2f}" if threshold is not None else ""
    return (
        f"<div class='ink-gauge'>"
        f"<div class='ink-gauge-top'>"
        f"<span class='ink-gauge-lbl'>{escape(label)}</span>"
        f"<span class='ink-gauge-val'>"
        f"<span><strong>{value:.3f}</strong>{escape(thresh_txt)}</span>"
        f"{chip(status_txt, colour)}"
        f"</span>"
        f"</div>"
        f"<div class='ink-gauge-track'>"
        f"<div class='ink-gauge-fill' style='width:{v * 100:.1f}%;background:{colour}'></div>"
        f"{mark}"
        f"</div>"
        f"<div class='ink-gauge-desc'>{escape(caption)}</div>"
        f"</div>"
    )


def score_bar(label: str, value: float | None, threshold: float | None, caption: str = "") -> None:
    st.html(score_bar_html(label, value, threshold, caption))


def verification_inspector_html(
    gate_a_score: float,
    gate_a_thresh: float,
    gate_b_score: float | None,
    gate_b_thresh: float,
    composite: float,
    composite_thresh: float,
    timings: dict[str, float],
) -> str:
    """Format the sticky Verification Inspector rail with 3 gauges and latency budget."""
    gb_caption = (
        "DeBERTa NLI mean entailment. Contradiction veto caps at 0.15."
        if gate_b_score is not None
        else "Not run: skipped at Gate A."
    )

    g1 = score_bar_html("Gate A · Retrieval Sufficiency", gate_a_score, gate_a_thresh, "Cross-encoder top-2 mean relevance.")
    g2 = score_bar_html("Gate B · Groundedness", gate_b_score, gate_b_thresh, gb_caption)
    g3 = score_bar_html("Composite Confidence", composite, composite_thresh, "Weighted harmonic mean (0.4 Gate A + 0.6 Gate B).")

    # Latency stacked breakdown
    ret_ms = timings.get("retrieval_ms", 0.0)
    ga_ms = timings.get("gate_a_ms", 0.0)
    gen_ms = timings.get("generate_ms", 0.0)
    gb_ms = timings.get("gate_b_ms", 0.0)
    total_ms = max(timings.get("total_ms", 0.1), 0.1)

    pct_ret = (ret_ms / total_ms) * 100
    pct_ga = (ga_ms / total_ms) * 100
    pct_gen = (gen_ms / total_ms) * 100
    pct_gb = (gb_ms / total_ms) * 100

    latency_html = (
        f"<div style='margin-top:1.15rem;padding-top:0.85rem;border-top:1px solid var(--border)'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem'>"
        f"<span style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-faint)'>Latency Budget</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;font-weight:600;color:var(--text)'>{total_ms / 1000:.2f} s total</span>"
        f"</div>"
        f"<div class='ink-latency-bar'>"
        f"<div class='ink-latency-seg' style='width:{pct_ret:.1f}%;background:var(--info)' title='Retrieve {ret_ms:.0f}ms'></div>"
        f"<div class='ink-latency-seg' style='width:{pct_ga:.1f}%;background:var(--accent)' title='Gate A {ga_ms:.0f}ms'></div>"
        f"<div class='ink-latency-seg' style='width:{pct_gen:.1f}%;background:var(--warn)' title='Generate {gen_ms:.0f}ms'></div>"
        f"<div class='ink-latency-seg' style='width:{pct_gb:.1f}%;background:var(--pass)' title='Gate B {gb_ms:.0f}ms'></div>"
        f"</div>"
        f"<div class='ink-latency-legend'>"
        f"<span class='ink-latency-item'><span class='ink-dot' style='background:var(--info)'></span>Ret: {ret_ms:.0f}ms</span>"
        f"<span class='ink-latency-item'><span class='ink-dot' style='background:var(--accent)'></span>Gate A: {ga_ms:.0f}ms</span>"
        f"<span class='ink-latency-item'><span class='ink-dot' style='background:var(--warn)'></span>Gen: {gen_ms:.0f}ms</span>"
        f"<span class='ink-latency-item'><span class='ink-dot' style='background:var(--pass)'></span>Gate B: {gb_ms:.0f}ms</span>"
        f"</div>"
        f"</div>"
    )

    return (
        f"<div class='ink-inspector'>"
        f"<div class='ink-inspector-hdr'>"
        f"<span>{icon_svg('sliders', 13, 1.5, 'var(--text-faint)')} Verification Inspector</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.68rem'>HARMONIC DECISION</span>"
        f"</div>"
        f"{g1}{g2}{g3}"
        f"{latency_html}"
        f"</div>"
    )


# --- Reusable UI Elements ---

def chip(text: str, colour: str) -> str:
    """Format a small status chip with inline border."""
    return (
        f"<span class='cr-chip' style='color:{colour};background:{colour}18;"
        f"border-color:{colour}40'>{escape(text)}</span>"
    )


def card(title: str, body: str) -> None:
    """A standard dark surface card."""
    st.html(
        f"<div class='ink-card'>"
        f"<div style='font-size:0.92rem;font-weight:700;margin-bottom:0.35rem;color:var(--text)'>{escape(title)}</div>"
        f"<div style='font-size:0.84rem;color:var(--text-muted);line-height:1.48'>{escape(body)}</div>"
        f"</div>"
    )


def kpi_card(label: str, value: str, subtext: str = "", colour: str = BRAND) -> None:
    """A KPI stat card with tabular numbers."""
    st.html(
        f"<div class='ink-kpi' style='border-top:2px solid {colour}'>"
        f"<div class='ink-kpi-lbl'>{escape(label)}</div>"
        f"<div class='ink-kpi-val'>{escape(value)}</div>"
        f"<div class='ink-kpi-sub' style='color:{colour}'>{escape(subtext)}</div>"
        f"</div>"
    )


def sample_card(tag: str, question: str, why: str, badge_colour: str = BRAND) -> None:
    """Legacy card retained for backwards compatibility."""
    st.html(
        f"<div class='ink-card' style='padding:0.85rem 1rem;min-height:96px;margin-bottom:0.4rem'>"
        f"<div style='margin-bottom:0.25rem'>{chip(tag.upper(), badge_colour)}</div>"
        f"<div style='font-size:0.88rem;font-weight:600;color:var(--text);margin-bottom:0.2rem'>{escape(question)}</div>"
        f"<div style='font-size:0.78rem;color:var(--text-muted)'>{escape(why)}</div>"
        f"</div>"
    )


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    """Slim header banner replacing previous gradient hero."""
    badge_list = badges or [
        "ms-marco-MiniLM Gate A",
        "DeBERTa-v3 Gate B v2",
        "Harmonic (0.4 / 0.6)",
    ]
    pills_html = "".join(f"<span class='ink-status-chip'>{escape(b)}</span>" for b in badge_list)
    st.html(
        f"<div style='padding:0.4rem 0 0.8rem 0;border-bottom:1px solid var(--border);margin-bottom:1.1rem'>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem'>"
        f"<div>"
        f"<h2 style='font-size:1.45rem;font-weight:700;letter-spacing:-0.02em;margin:0 0 0.2rem 0;color:var(--text)'>{escape(title)}</h2>"
        f"<p style='font-size:0.86rem;color:var(--text-muted);margin:0'>{escape(subtitle)}</p>"
        f"</div>"
        f"<div style='display:flex;gap:0.4rem;flex-wrap:wrap'>{pills_html}</div>"
        f"</div>"
        f"</div>"
    )


# --- Cursor Spotlight & Edge Glow Component ---

def inject_ambient_cursor_effect(enabled: bool = True) -> None:
    """Inject performant soft spotlight and card edge glow (no external deps, 0 idle CPU)."""
    if not enabled:
        # If toggled off, remove any existing ambient spotlight element
        st.components.v1.html(
            """<script>
            try {
                const el = window.parent.document.getElementById('rag-ambient-spotlight');
                if (el) el.remove();
            } catch(e) {}
            </script>""",
            height=0,
            width=0,
        )
        return

    # Injected once with idempotency guard
    st.components.v1.html(
        """<script>
        (function() {
            try {
                const pDoc = window.parent.document;
                const pWin = window.parent.window;
                if (!pDoc || !pWin) return;

                // Respect user prefers-reduced-motion
                if (pWin.matchMedia && pWin.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    return;
                }
                // Skip on touch devices
                if ('ontouchstart' in pWin || (pWin.navigator && pWin.navigator.maxTouchPoints > 0)) {
                    return;
                }

                if (pDoc.getElementById('rag-ambient-spotlight')) return;

                const spot = pDoc.createElement('div');
                spot.id = 'rag-ambient-spotlight';
                spot.style.position = 'fixed';
                spot.style.width = '480px';
                spot.style.height = '480px';
                spot.style.borderRadius = '50%';
                spot.style.pointerEvents = 'none';
                spot.style.zIndex = '0';
                spot.style.background = 'radial-gradient(circle, rgba(124, 140, 255, 0.055) 0%, transparent 70%)';
                spot.style.transform = 'translate(-50%, -50%)';
                spot.style.transition = 'opacity 0.25s ease';
                spot.style.opacity = '1';
                pDoc.body.appendChild(spot);

                let mouseX = -500, mouseY = -500;
                let curX = -500, curY = -500;
                let isAnimating = false;

                function onMouseMove(e) {
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                    if (!isAnimating) {
                        isAnimating = true;
                        pWin.requestAnimationFrame(render);
                    }
                }

                function render() {
                    const dx = mouseX - curX;
                    const dy = mouseY - curY;
                    curX += dx * 0.12;
                    curY += dy * 0.12;

                    spot.style.left = curX.toFixed(1) + 'px';
                    spot.style.top = curY.toFixed(1) + 'px';

                    // Stop loop when close enough to save 100% idle CPU
                    if (Math.abs(dx) > 0.4 || Math.abs(dy) > 0.4) {
                        pWin.requestAnimationFrame(render);
                    } else {
                        isAnimating = false;
                    }
                }

                pDoc.addEventListener('mousemove', onMouseMove, { passive: true });
            } catch (err) {
                // Silently fallback if iframe sandboxing restricts parent access
            }
        })();
        </script>""",
        height=0,
        width=0,
    )
