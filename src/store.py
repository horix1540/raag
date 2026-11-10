"""SQLite telemetry store backing the monitoring dashboard.

SQLite rather than a CSV or a log file because the dashboard needs to aggregate
over time ("confidence trend for the last 200 queries"), and because a demo
that survives a restart is more convincing than one that forgets everything.
Everything is written synchronously - at demo query volumes the cost is
irrelevant, and it means a crash never loses the record of what happened.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.config import RUNS_DB

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    answer            TEXT,
    verdict           TEXT    NOT NULL,
    composite         REAL,
    gate_a_score      REAL,
    gate_a_passed     INTEGER,
    gate_b_score      REAL,
    gate_b_passed     INTEGER,
    rerank_relevancy  REAL,
    top_similarity    REAL,
    mean_similarity   REAL,
    similarity_margin REAL,
    per_chunk_relevance TEXT,
    nli_faithfulness  REAL,
    llm_relevancy     REAL,
    llm_faithfulness  REAL,
    n_claims          INTEGER,
    n_conflicts       INTEGER,
    n_baseless        INTEGER,
    retrieval_ms      REAL,
    generation_ms     REAL,
    gate_a_ms         REAL,
    gate_b_ms         REAL,
    total_ms          REAL,
    provider          TEXT,
    generator_model   TEXT,
    scorer            TEXT,
    sources           TEXT,
    claims_json       TEXT,
    run_tag           TEXT,
    arm               TEXT,
    self_abstain      INTEGER,
    item_id           TEXT,
    thresholds_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS idx_queries_verdict ON queries(verdict);
CREATE INDEX IF NOT EXISTS idx_queries_run_tag ON queries(run_tag);
"""

# Columns added after the first databases were already on disk. SQLite cannot
# add them via CREATE TABLE IF NOT EXISTS, so they are applied separately;
# without this an existing runs.sqlite3 would silently drop the new fields
# (log_query filters unknown keys) and the benchmark could not tell the guarded
# and baseline arms apart.
MIGRATIONS = {
    "arm": "TEXT",
    "self_abstain": "INTEGER",
    "item_id": "TEXT",
    "per_chunk_relevance": "TEXT",
    "thresholds_json": "TEXT",
}


@contextmanager
def connect(db_path: Path | str = RUNS_DB):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = RUNS_DB) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(queries)")}
        for column, sqltype in MIGRATIONS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE queries ADD COLUMN {column} {sqltype}")
                log.info("Migrated telemetry table: added %s", column)


def log_query(record: dict, db_path: Path | str = RUNS_DB) -> int:
    """Insert one pipeline result. Returns the new row id."""
    init_db(db_path)
    row = dict(record)
    row.setdefault("ts", datetime.now(UTC).isoformat(timespec="seconds"))

    for key in ("sources", "claims_json"):
        if key in row and not isinstance(row[key], (str, type(None))):
            row[key] = json.dumps(row[key], ensure_ascii=False)
    for key in ("gate_a_passed", "gate_b_passed"):
        if row.get(key) is not None:
            row[key] = int(bool(row[key]))

    allowed = _columns(db_path)
    row = {k: v for k, v in row.items() if k in allowed}

    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with connect(db_path) as conn:
        cur = conn.execute(f"INSERT INTO queries ({cols}) VALUES ({marks})", tuple(row.values()))
        return int(cur.lastrowid or 0)


def _columns(db_path: Path | str = RUNS_DB) -> set[str]:
    with connect(db_path) as conn:
        return {r["name"] for r in conn.execute("PRAGMA table_info(queries)")}


def load_dataframe(limit: int = 1000, run_tag: str | None = None, db_path: Path | str = RUNS_DB):
    """Read recent queries into a pandas DataFrame for the dashboard."""
    import pandas as pd

    init_db(db_path)
    sql = "SELECT * FROM queries"
    params: tuple = ()
    if run_tag:
        sql += " WHERE run_tag = ?"
        params = (run_tag,)
    sql += " ORDER BY id DESC LIMIT ?"
    params += (limit,)

    with connect(db_path) as conn:
        df = pd.DataFrame([dict(r) for r in conn.execute(sql, params)])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", format="mixed")
        df = df.sort_values("id")
    return df


def summary_stats(db_path: Path | str = RUNS_DB) -> dict:
    """Headline numbers for the dashboard's metric row."""
    init_db(db_path)
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM queries").fetchone()["n"]
        if not total:
            return {
                "total": 0,
                "answered": 0,
                "abstained": 0,
                "flagged": 0,
                "abstention_rate": 0.0,
                "mean_composite": 0.0,
                "mean_latency_ms": 0.0,
                "total_conflicts": 0,
                "total_baseless": 0,
            }
        counts = {
            r["verdict"]: r["n"]
            for r in conn.execute("SELECT verdict, COUNT(*) AS n FROM queries GROUP BY verdict")
        }
        agg = conn.execute(
            "SELECT AVG(composite) AS c, AVG(total_ms) AS l, "
            "SUM(COALESCE(n_conflicts,0)) AS cf, SUM(COALESCE(n_baseless,0)) AS bl FROM queries"
        ).fetchone()

    abstained = counts.get("abstain_no_context", 0) + counts.get("abstain_ungrounded", 0)
    return {
        "total": total,
        "answered": counts.get("answer", 0),
        "abstained": abstained,
        "flagged": counts.get("flag_low_confidence", 0),
        "abstention_rate": abstained / total,
        "mean_composite": float(agg["c"] or 0.0),
        "mean_latency_ms": float(agg["l"] or 0.0),
        "total_conflicts": int(agg["cf"] or 0),
        "total_baseless": int(agg["bl"] or 0),
    }


def clear(db_path: Path | str = RUNS_DB) -> None:
    """Wipe telemetry. Used before a clean demo run, never automatically."""
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM queries")
    log.warning("Telemetry cleared: %s", db_path)


