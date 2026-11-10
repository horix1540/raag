"""Unit tests for the Ink + Signal UI presentation helpers.

Validates that all formatting helpers in src/ui.py are pure functions that safely
escape user/LLM text and format statuses, scores, and latencies correctly.
"""

from __future__ import annotations

from src import ui


def test_icon_svg_generation():
    svg = ui.icon_svg("shield-check", size=20, stroke_width=1.5, color="#7C8CFF")
    assert "<svg" in svg
    assert 'width="20"' in svg
    assert 'height="20"' in svg
    assert 'stroke="#7C8CFF"' in svg
    assert 'stroke-width="1.5"' in svg
    assert "</svg>" in svg


def test_format_answer_html_citations_and_xss_safety():
    # Test citation replacement: [1] -> Ref 1 pill
    raw = "A RAG pipeline has five stages [1]. Ingestion is the first [2, 3]."
    formatted = ui.format_answer_html(raw)
    assert "<span class='cr-cite'>Ref 1</span>" in formatted
    assert "<span class='cr-cite'>Ref 2, 3</span>" in formatted

    # Test XSS escaping
    malicious = "Malicious draft <script>alert('xss')</script> [1]"
    escaped = ui.format_answer_html(malicious)
    assert "<script>" not in escaped
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in escaped
    assert "<span class='cr-cite'>Ref 1</span>" in escaped


def test_highlight_passage_html():
    passage = "A conventional RAG pipeline has five stages. Lewis et al. introduced RAG."
    claims = ["A conventional RAG pipeline has five stages."]
    highlighted = ui.highlight_passage_html(passage, claims)
    assert "<span class='cr-hl'>" in highlighted
    assert "A conventional RAG pipeline has five stages." in highlighted

    # XSS safety
    malicious_passage = "Safe text. <img src=x onerror=alert(1)>"
    out = ui.highlight_passage_html(malicious_passage, ["Safe text."])
    assert "<img" not in out
    assert "&lt;img" in out


def test_score_bar_html():
    # Pass scenario
    html_pass = ui.score_bar_html("Gate A", 0.50, 0.45, "Relevance caption")
    assert "0.500" in html_pass
    assert "PASS" in html_pass
    assert "50.0%" in html_pass
    assert "Relevance caption" in html_pass

    # Fail scenario
    html_fail = ui.score_bar_html("Gate B", 0.15, 0.25, "Faithfulness caption")
    assert "0.150" in html_fail
    assert "BELOW" in html_fail
    assert "15.0%" in html_fail

    # Skipped scenario
    html_skip = ui.score_bar_html("Gate B", None, 0.25, "Skipped")
    assert "SKIPPED" in html_skip


def test_verdict_banner_html():
    # Answered
    b1 = ui.verdict_banner_html("answer", "Both gates passed high confidence.")
    assert "Verified &amp; Answered" in b1
    assert "Both gates passed high confidence." in b1
    assert ui.PASS in b1

    # Abstained Gate A
    b2 = ui.verdict_banner_html("abstain_no_context", "Gate A relevance 0.00 is below threshold 0.45.")
    assert "Abstained · Insufficient Retrieval Context" in b2
    assert "Gate A relevance 0.00" in b2
    assert ui.FAIL in b2


def test_pipeline_timeline_html_idle_and_active():
    # Idle
    idle_html = ui.pipeline_timeline_html(None)
    assert "Pipeline Execution Timeline" in idle_html
    assert "awaiting query" in idle_html

    # Mocked pipeline result
    class _MockGateA:
        score = 0.50
        passed = True
        latency_ms = 400.0

    class _MockGateB:
        score = 0.997
        passed = True
        latency_ms = 300.0

    class _MockConf:
        gate_a = _MockGateA()
        gate_b = _MockGateB()
        composite = 0.714
        abstained = False

    class _MockGen:
        latency_ms = 1200.0
        self_reported_insufficient = False

    class _MockRet:
        chunks = [1, 2, 3, 4, 5]

    class _MockRes:
        retrieval = _MockRet()
        retrieval_ms = 25.0
        generation = _MockGen()
        confidence = _MockConf()
        total_ms = 1925.0

    active_html = ui.pipeline_timeline_html(_MockRes(), baseline=False, prompt_only=False)
    assert "5 chunks" in active_html
    assert "0.50 passed" in active_html
    assert "1.00 passed" in active_html
    assert "0.71 answered" in active_html
    assert "1.93s" in active_html


def test_verification_inspector_html():
    timings = {
        "retrieval_ms": 20.0,
        "gate_a_ms": 350.0,
        "generate_ms": 1100.0,
        "gate_b_ms": 280.0,
        "total_ms": 1750.0,
    }
    insp = ui.verification_inspector_html(
        gate_a_score=0.50,
        gate_a_thresh=0.45,
        gate_b_score=0.997,
        gate_b_thresh=0.25,
        composite=0.714,
        composite_thresh=0.33,
        timings=timings,
    )
    assert "Verification Inspector" in insp
    assert "Gate A · Retrieval Sufficiency" in insp
    assert "Gate B · Groundedness" in insp
    assert "Composite Confidence" in insp
    assert "Latency Budget" in insp
    assert "1.75 s total" in insp
