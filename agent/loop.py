"""Агент поддержки «Кладовки»: цикл «модель → инструменты → покупатель».

Это то, что вы будете улучшать. Меняйте промпт, описание инструментов,
сам цикл. Не трогайте среду (`kladovka/`), задачи (`tasks/`) и проверку
(`evals/`): по ним мерят, а не правят.
"""

from __future__ import annotations

import time
from pathlib import Path

from kladovka.store import Store
from kladovka.user import STOP

PROMPT = Path(__file__).parent / "prompt.md"
POLICY = Path(__file__).parent.parent / "kladovka" / "policy.md"
MAX_STEPS = 30


def system_prompt() -> str:
    return PROMPT.read_text(encoding="utf-8").strip() + "\n\n" + POLICY.read_text(encoding="utf-8").strip()


def run(store: Store, user, llm, *, max_steps: int = MAX_STEPS) -> dict:
    """Один разговор. Возвращает трейс: все сообщения, расход и как кончилось."""
    messages: list[dict] = [{"role": "system", "content": system_prompt()}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0}
    started = time.time()
    end = "max_steps"

    first = user.reply(None)
    messages.append({"role": "user", "content": first})
    if STOP in first:
        end = "user_stop"
    else:
        for _ in range(max_steps):
            reply = llm.chat(messages, Store.tool_schemas())
            usage["model_calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += int(reply.usage.get(k) or 0)
            messages.append(reply.message)

            calls = reply.message.get("tool_calls") or []
            if calls:
                for c in calls:
                    result = store.call(c["function"]["name"], c["function"]["arguments"])
                    messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"], "content": result})
                continue

            answer = user.reply(reply.message.get("content") or "")
            messages.append({"role": "user", "content": answer})
            if STOP in answer:
                end = "user_stop"
                break

    return {
        "messages": messages,
        "usage": usage,
        "end": end,
        "seconds": round(time.time() - started, 1),
    }
