import time
from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class ModelConfig:
    name: str
    provider: str  # "ollama" or "openai_compat"
    model_id: str
    base_url: str
    api_key: str = ""
    system_prompt: str = "You are a helpful, accurate, and concise assistant."
    temperature: float = 0.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class GenerationResult:
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None


class LLMProvider:
    @staticmethod
    def generate(config: ModelConfig, user_prompt: str, timeout: int = 60) -> GenerationResult:
        start = time.perf_counter()
        try:
            if config.provider == "ollama":
                return LLMProvider._call_ollama(config, user_prompt, start, timeout)
            elif config.provider == "openai_compat":
                return LLMProvider._call_openai_compat(config, user_prompt, start, timeout)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return GenerationResult(
                response_text="",
                latency_ms=round(elapsed_ms, 2),
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=str(exc),
            )

    @staticmethod
    def _call_ollama(
        config: ModelConfig, user_prompt: str, start: float, timeout: int
    ) -> GenerationResult:
        url = f"{config.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": config.model_id,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": config.temperature},
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = data.get("message", {}).get("content", "").strip()
        in_tok = int(data.get("prompt_eval_count", max(1, len(user_prompt.split()))))
        out_tok = int(data.get("eval_count", max(1, len(text.split()))))
        cost = (in_tok / 1000) * config.cost_per_1k_input + (out_tok / 1000) * config.cost_per_1k_output

        return GenerationResult(
            response_text=text,
            latency_ms=round(elapsed_ms, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )

    @staticmethod
    def _call_openai_compat(
        config: ModelConfig, user_prompt: str, start: float, timeout: int
    ) -> GenerationResult:
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.model_id,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", max(1, len(user_prompt.split()))))
        out_tok = int(usage.get("completion_tokens", max(1, len(text.split()))))
        cost = (in_tok / 1000) * config.cost_per_1k_input + (out_tok / 1000) * config.cost_per_1k_output

        return GenerationResult(
            response_text=text,
            latency_ms=round(elapsed_ms, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )
