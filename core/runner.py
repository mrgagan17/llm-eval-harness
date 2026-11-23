from typing import Any, Callable, Dict, List, Optional
import numpy as np
from core.judge import LLMJudge
from core.providers import LLMProvider, ModelConfig


class BenchmarkRunner:
    def __init__(
        self,
        model_a: ModelConfig,
        model_b: ModelConfig,
        judge_config: ModelConfig,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.judge = LLMJudge(judge_config)

    def run(
        self,
        dataset: List[Dict[str, str]],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        total = len(dataset)

        for idx, item in enumerate(dataset, start=1):
            prompt = item["prompt"]
            reference = item["reference_answer"]
            category = item.get("category", "general")

            if progress_cb:
                progress_cb(idx, total, f"Evaluating prompt {idx}/{total} ({category})...")

            res_a = LLMProvider.generate(self.model_a, prompt)
            res_b = LLMProvider.generate(self.model_b, prompt)

            score_a = self.judge.score(prompt, reference, res_a.response_text)
            score_b = self.judge.score(prompt, reference, res_b.response_text)

            if score_a.overall_score > score_b.overall_score:
                winner = self.model_a.name
            elif score_b.overall_score > score_a.overall_score:
                winner = self.model_b.name
            else:
                winner = "Tie"

            rows.append(
                {
                    "id": item.get("id", idx),
                    "category": category,
                    "prompt": prompt,
                    "reference_answer": reference,
                    "response_a": res_a.response_text or f"ERROR: {res_a.error}",
                    "response_b": res_b.response_text or f"ERROR: {res_b.error}",
                    "latency_a_ms": res_a.latency_ms,
                    "latency_b_ms": res_b.latency_ms,
                    "cost_a_usd": res_a.cost_usd,
                    "cost_b_usd": res_b.cost_usd,
                    "score_a": score_a.overall_score,
                    "score_b": score_b.overall_score,
                    "reason_a": score_a.reasoning,
                    "reason_b": score_b.reasoning,
                    "winner": winner,
                }
            )

        summary = self._compute_summary(rows)
        return {"summary": summary, "results": rows}

    def _compute_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        lat_a = [r["latency_a_ms"] for r in rows]
        lat_b = [r["latency_b_ms"] for r in rows]
        sc_a = [r["score_a"] for r in rows]
        sc_b = [r["score_b"] for r in rows]

        wins_a = sum(1 for r in rows if r["winner"] == self.model_a.name)
        wins_b = sum(1 for r in rows if r["winner"] == self.model_b.name)
        ties = sum(1 for r in rows if r["winner"] == "Tie")

        return {
            "total_prompts": len(rows),
            "model_a": self.model_a.name,
            "model_b": self.model_b.name,
            "avg_score_a": round(float(np.mean(sc_a)), 2),
            "avg_score_b": round(float(np.mean(sc_b)), 2),
            "p95_latency_a_ms": round(float(np.percentile(lat_a, 95)), 1),
            "p95_latency_b_ms": round(float(np.percentile(lat_b, 95)), 1),
            "total_cost_a_usd": round(sum(r["cost_a_usd"] for r in rows), 5),
            "total_cost_b_usd": round(sum(r["cost_b_usd"] for r in rows), 5),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
        }
