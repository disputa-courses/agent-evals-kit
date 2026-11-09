"""Проверка по исходу: каким стал магазин и что агент сказал покупателю.

Эталон получается так же, как в τ²-bench: эталонные действия задачи
выполняются на свежем магазине, и его снимок сравнивается со снимком
после агента. Засчитывается любой путь, который привёл к тому же
состоянию; лишнее действие (отменил второй заказ, вернул не ту позицию)
меняет снимок и даёт ноль.
"""

from __future__ import annotations

import re
from typing import Any

from kladovka.store import Store


class BadTask(Exception):
    """Эталон задачи не выполняется на свежем магазине: задача сломана."""


def apply(actions: list[dict]) -> Store:
    s = Store()
    for a in actions:
        out = s.call(a["name"], a.get("arguments", {}))
        if out.startswith("Ошибка:"):
            raise BadTask(f"{a['name']} {a.get('arguments')}: {out}")
    return s


def expected(task: dict) -> list[Store]:
    variants = [task.get("actions", [])] + task.get("alternatives", [])
    return [apply(v) for v in variants]


def _norm(text: str) -> str:
    # «2 790», «2 790» и «2790» — одно число.
    text = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", text or "")
    return text.lower()


def said(messages: list[dict]) -> str:
    return "\n".join(
        m.get("content") or "" for m in messages if m.get("role") == "assistant"
    )


def grade(task: dict, store: Store, messages: list[dict]) -> dict[str, Any]:
    targets = expected(task)
    state_ok = any(t.hash() == store.hash() for t in targets)
    text = _norm(said(messages))
    missing = [o for o in task.get("outputs", []) if _norm(o) not in text]
    return {
        "reward": int(state_ok and not missing),
        "state_ok": state_ok,
        "outputs_ok": not missing,
        "missing_outputs": missing,
        # Расхождение с основным эталоном: что агент сделал не так в данных.
        "diff": [] if state_ok else targets[0].diff(store),
    }
