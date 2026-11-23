import json
import re
from dataclasses import dataclass
from typing import Dict, Any
from core.providers import LLMProvider, ModelConfig


@dataclass
class JudgeScore:
    accuracy: float
    completeness: float
    conciseness: float
    overall_score: float
    reasoning: str
    used_fallback: bool = False


class LLMJudge:
    RUBRIC_PROMPT = """You are an impartial, strict LLM evaluator.
Score the Candidate Answer against the Reference Answer for the given Question.

Rubric (1 to 5 scale, integers only):
- accuracy: 5 = completely factually correct, 1 = wrong or hallucinated.
- completeness: 5 = covers all key points of the reference, 1 = misses the core answer.
- conciseness: 5 = clear and direct without fluff, 1 = overly verbose or rambling.

Return ONLY valid JSON with these exact keys:
{
  "accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "conciseness": <int 1-5>,
  "reasoning": "<one concise sentence explaining the score>"
}
"""

    def __init__(self, judge_config: ModelConfig) -> None:
        self.judge_config = judge_config
        self.judge_config.system_prompt = self.RUBRIC_PROMPT
        self.judge_config.temperature = 0.0

    def score(self, question: str, reference: str, candidate: str) -> JudgeScore:
        if not candidate.strip():
            return JudgeScore(1.0, 1.0, 1.0, 1.0, "Empty response from model.", used_fallback=True)

        eval_prompt = (
            f"Question:\n{question}\n\n"
            f"Reference Answer:\n{reference}\n\n"
            f"Candidate Answer:\n{candidate}"
        )

        res = LLMProvider.generate(self.judge_config, eval_prompt, timeout=45)
        if not res.error and res.response_text:
            parsed = self._extract_json(res.response_text)
            if parsed:
                acc = float(min(5, max(1, parsed.get("accuracy", 3))))
                comp = float(min(5, max(1, parsed.get("completeness", 3))))
                conc = float(min(5, max(1, parsed.get("conciseness", 3))))
                overall = round(acc * 0.5 + comp * 0.35 + conc * 0.15, 2)
                return JudgeScore(
                    accuracy=acc,
                    completeness=comp,
                    conciseness=conc,
                    overall_score=overall,
                    reasoning=str(parsed.get("reasoning", "Scored by LLM judge.")),
                )

        return self._lexical_fallback(reference, candidate)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _lexical_fallback(reference: str, candidate: str) -> JudgeScore:
        ref_tokens = set(re.findall(r"\w+", reference.lower()))
        cand_tokens = set(re.findall(r"\w+", candidate.lower()))
        if not ref_tokens:
            return JudgeScore(3.0, 3.0, 3.0, 3.0, "Fallback score (no reference tokens).", True)

        overlap = len(ref_tokens & cand_tokens) / len(ref_tokens)
        scaled = round(1.0 + overlap * 4.0, 2)
        return JudgeScore(
            accuracy=scaled,
            completeness=scaled,
            conciseness=3.5,
            overall_score=scaled,
            reasoning="Fallback lexical overlap score (judge JSON unavailable).",
            used_fallback=True,
        )
