from __future__ import annotations
import json
import re
from pathlib import Path


class Memory:
    """两层记忆：
    - 短期记忆：本次会话内的历史（查询、结论），注入规划上下文，避免重复分析。
    - 长期记忆：跨会话持久化（JSONL），基于关键词重叠检索，不引入向量库依赖。
    """

    def __init__(self, store_path: str = "memory_store.jsonl", max_short: int = 6, max_long: int = 100):
        self.store_path = Path(store_path)
        self.max_short = max_short
        self.max_long = max_long
        self.short: list[dict] = []
        self._long: list[dict] = self._load()
            
    def remember_short(self, kind: str, content: str) -> None:
        self.short.append({"kind": kind, "content": content})
        self.short = self.short[-self.max_short:]

    def short_context(self) -> str:
        if not self.short:
            return ""
        lines = [f"- [{m['kind']}] {m['content']}" for m in self.short]
        return "本次会话已完成的历史信息（可复用，避免重复分析）：\n" + "\n".join(lines)

    def remember_long(self, query: str, conclusion: str) -> None:
        self._long.append({"query": query, "conclusion": conclusion})
        self._save()


    def recall(self, query: str, top_k: int = 3) -> str:
        """基于关键词重叠的轻量检索（中文按 2-gram 切分，英文按词）。"""
        if not self._long:
            return ""
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return ""
        scored = []
        for e in self._long:
            e_tokens = set(self._tokenize(e.get("query", "")))  
            if not e_tokens:
                continue
            overlap = len(q_tokens & e_tokens) / max(1, len(q_tokens))
            if overlap > 0:
                scored.append((overlap, e))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return ""
        lines = [f"- 历史问题「{e['query']}」→ 结论：{e['conclusion']}" for _, e in scored[:top_k]]
        return "历史任务参考（来自长期记忆，仅作参考，数据可能不同）：\n" + "\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text or "")


        words: list[str] = []
        for t in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]+", t):
                words.extend(t[i:i + 2] for i in range(max(1, len(t) - 1)))
            else:
                words.append(t.lower())
        return words

    def _load(self) -> list[dict]:
        if not self.store_path.exists():
            return []
        out = []
        try:
            for line in self.store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception:
            return []
        return out[-self.max_long:]

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for e in self._long[-self.max_long:]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
