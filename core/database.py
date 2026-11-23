import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List


class EvalDB:
    def __init__(self, db_path: str = "eval_history.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    model_a TEXT,
                    model_b TEXT,
                    total_prompts INTEGER,
                    avg_score_a REAL,
                    avg_score_b REAL,
                    p95_latency_a_ms REAL,
                    p95_latency_b_ms REAL,
                    total_cost_a_usd REAL,
                    total_cost_b_usd REAL,
                    wins_a INTEGER,
                    wins_b INTEGER,
                    ties INTEGER,
                    details_json TEXT
                )
                """
            )

    def save_run(self, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO benchmark_runs (
                    created_at, model_a, model_b, total_prompts,
                    avg_score_a, avg_score_b, p95_latency_a_ms, p95_latency_b_ms,
                    total_cost_a_usd, total_cost_b_usd, wins_a, wins_b, ties, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    summary["model_a"],
                    summary["model_b"],
                    summary["total_prompts"],
                    summary["avg_score_a"],
                    summary["avg_score_b"],
                    summary["p95_latency_a_ms"],
                    summary["p95_latency_b_ms"],
                    summary["total_cost_a_usd"],
                    summary["total_cost_b_usd"],
                    summary["wins_a"],
                    summary["wins_b"],
                    summary["ties"],
                    json.dumps(results),
                ),
            )

    def fetch_runs(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, created_at, model_a, model_b, total_prompts, "
                "avg_score_a, avg_score_b, p95_latency_a_ms, p95_latency_b_ms, "
                "wins_a, wins_b, ties FROM benchmark_runs ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
