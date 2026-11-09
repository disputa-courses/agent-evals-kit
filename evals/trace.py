"""Удобный доступ к трейсу для проверок траектории."""

from __future__ import annotations

import json
from dataclasses import dataclass

from kladovka.store import Store


@dataclass
class Step:
    index: int
    role: str  # user, assistant, tool
    text: str
    tool: str | None = None
    arguments: dict | None = None
    result: str | None = None


class Trace:
    def __init__(self, record: dict):
        self.record = record
        self.id = f"{record['task_id']}:{record['trial']}"
        self.task_id = record["task_id"]
        self.reward = record["grade"]["reward"]
        self.steps: list[Step] = []
        results = {m.get("tool_call_id"): m.get("content") for m in record["messages"] if m["role"] == "tool"}
        for i, m in enumerate(record["messages"]):
            if m["role"] in ("user", "assistant") and m.get("content"):
                self.steps.append(Step(i, m["role"], m["content"]))
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.steps.append(Step(i, "tool", "", c["function"]["name"], args, results.get(c["id"])))

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.role == "tool"]

    @property
    def writes(self) -> list[Step]:
        """Вызовы, которые меняют данные магазина (и прошли без ошибки)."""
        w = Store.write_tools()
        return [s for s in self.tool_calls if s.tool in w and not (s.result or "").startswith("Ошибка")]

    def before(self, step: Step) -> list[Step]:
        """Всё, что случилось в разговоре до этого шага."""
        return self.steps[: self.steps.index(step)]

    def said(self) -> str:
        return "\n".join(s.text for s in self.steps if s.role == "assistant")


def conversation(record: dict) -> str:
    """Разговор текстом для судьи и разметки: без итога проверки.

    Судья не должен видеть, засчитала ли задачу проверка исхода, иначе он
    повторит её, а не оценит своё.
    """
    lines = []
    for m in record["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            lines.append(f"[результат {m.get('name')}] {m['content']}")
            continue
        if m.get("content"):
            lines.append(f"{'Покупатель' if m['role'] == 'user' else 'Агент'}: {m['content']}")
        for c in m.get("tool_calls") or []:
            lines.append(f"[агент вызывает {c['function']['name']}] {c['function']['arguments']}")
    return "\n".join(lines)
