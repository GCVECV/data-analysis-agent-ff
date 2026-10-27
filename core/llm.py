from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import LLMConfig

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 3)


@dataclass  
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class LLMClient:
    """统一封装：重试、usage 埋点、单次 chat。"""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.usage = LLMUsage()

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        t0 = time.time()
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        self.usage.calls += 1
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens
            self.usage.completion_tokens += resp.usage.completion_tokens
        return resp, time.time() - t0

    def stream(self, messages: list[dict], tools: list[dict] | None = None):
        """流式生成：逐 chunk 产出文本增量，用于报告阶段的流式输出。"""
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        self.usage.calls += 1
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
