"""Симулятор покупателя: модель играет человека с целью из задачи.

Устроен как в τ-bench: у покупателя есть инструкция, он раскрывает
данные, только когда их спросят, и заканчивает разговор словом ###STOP###.
"""

from __future__ import annotations

from .llm import LLM, ScriptedLLM

STOP = "###STOP###"

SYSTEM = """Вы играете покупателя, который пишет в чат поддержки интернет-магазина «Кладовка».
Пишите от первого лица, коротко, как живой человек в чате: одна-две фразы за раз.

Ваша ситуация и цель:
{instructions}

Правила игры:
- Не выдавайте всю информацию сразу: называйте почту, телефон, номер заказа и подробности, только когда агент спросит или когда без них никак.
- Не выдумывайте данных, которых нет в ситуации. Если чего-то не знаете, так и скажите.
- Следуйте ситуации: если там сказано, как поступить в случае отказа, так и поступайте.
- Когда цель достигнута или агент объяснил, что сделать нельзя, и вам больше нечего спросить, ответьте одним словом {stop}.
- Не играйте агента и не описывайте свои действия в скобках."""


class UserSim:
    def __init__(self, instructions: str, llm: LLM | ScriptedLLM):
        self.llm = llm
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM.format(instructions=instructions, stop=STOP)}
        ]

    def reply(self, agent_text: str | None) -> str:
        # Для симулятора роли перевёрнуты: агент для него — «user».
        self.history.append({"role": "user", "content": agent_text or "Здравствуйте! Чем могу помочь?"})
        msg = self.llm.chat(self.history).message
        text = (msg.get("content") or "").strip()
        self.history.append({"role": "assistant", "content": text})
        return text


class ScriptedUser:
    """Покупатель по сценарию для тестов: реплики по очереди, потом STOP."""

    def __init__(self, lines: list[str]):
        self.lines = list(lines)

    def reply(self, agent_text: str | None) -> str:
        return self.lines.pop(0) if self.lines else STOP
