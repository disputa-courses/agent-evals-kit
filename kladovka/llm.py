"""Клиент модели по протоколу OpenAI Chat Completions, без зависимостей.

Подходит ИИ-шлюз Disputa и любой совместимый адрес. Настройка — переменные
окружения: LLM_BASE_URL, LLM_API_KEY, AGENT_MODEL, USER_MODEL, JUDGE_MODEL.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://disputa.ru/api/llm/v1"


@dataclass
class Reply:
    message: dict[str, Any]  # как в ответе API: role, content, tool_calls
    usage: dict[str, int] = field(default_factory=dict)


class LLM:
    def __init__(self, model: str, *, temperature: float = 0.0, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.temperature = temperature
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        data = self._post("/chat/completions", body)
        choice = data["choices"][0]["message"]
        msg = {"role": "assistant", "content": choice.get("content")}
        if choice.get("tool_calls"):
            msg["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                for c in choice["tool_calls"]
            ]
        return Reply(msg, data.get("usage") or {})

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        delay = 2.0
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                # 429 и 5xx бывают у любого провайдера: повтор с паузой.
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise RuntimeError(f"модель ответила {e.code}: {e.read()[:300]!r}") from e
            except urllib.error.URLError:
                if attempt == 4:
                    raise
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("недостижимо")


class ScriptedLLM:
    """Модель по сценарию: отдаёт заранее записанные ответы по очереди.

    Нужна, чтобы проверять обвязку без сети и денег: тесты среды,
    проверки и судьи гоняются на ней.
    """

    def __init__(self, replies: list[dict[str, Any]], model: str = "scripted"):
        self.replies = list(replies)
        self.model = model
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        self.seen.append(messages)
        if not self.replies:
            return Reply({"role": "assistant", "content": "###STOP###"})
        r = self.replies.pop(0)
        msg = {"role": "assistant", "content": r.get("content")}
        if r.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": f"call_{len(self.seen)}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)},
                }
                for i, c in enumerate(r["tool_calls"])
            ]
        return Reply(msg, {"prompt_tokens": 0, "completion_tokens": 0})


def from_env(role: str) -> LLM:
    """Модель для роли: agent, user или judge."""
    var = {"agent": "AGENT_MODEL", "user": "USER_MODEL", "judge": "JUDGE_MODEL"}[role]
    model = os.environ.get(var) or os.environ.get("AGENT_MODEL")
    if not model:
        raise SystemExit(f"не задана модель: {var} или AGENT_MODEL")
    # Симулятору пользователя нужна живость реплик, агенту и судье — повторяемость.
    temperature = float(os.environ.get(f"{var}_TEMPERATURE", "1.0" if role == "user" else "0.0"))
    return LLM(model, temperature=temperature)
