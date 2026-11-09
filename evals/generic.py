"""Метрика «из коробки» для сравнения: релевантность ответа по шкале 1–5.

Так устроены Answer Relevancy и G-Eval в популярных библиотеках: модель
читает вопрос и ответ и ставит балл. Запустите на своём прогоне и
сравните с проверкой исхода.

    python -m evals.generic runs/base
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

from .report import load_run

PROMPT = """Оцените, насколько ответ агента поддержки релевантен обращению покупателя и полезен ему.
Шкала: 1 — не по делу, 5 — полностью отвечает на обращение.

Обращение покупателя: {question}

Последний ответ агента: {answer}

Ответьте одной цифрой от 1 до 5."""


def score(llm, record: dict) -> int | None:
    users = [m["content"] for m in record["messages"] if m["role"] == "user" and m.get("content")]
    answers = [m["content"] for m in record["messages"] if m["role"] == "assistant" and m.get("content")]
    if not users or not answers:
        return None
    text = llm.chat([{"role": "user", "content": PROMPT.format(question=users[0], answer=answers[-1])}]).message.get("content") or ""
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None


def main(argv: list[str] | None = None) -> None:
    from kladovka import llm as L

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    a = ap.parse_args(argv)
    results, _ = load_run(a.run)
    llm = L.from_env("judge")
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda r: score(llm, r), results))
    rows = [(r, s) for r, s in zip(results, scores) if s is not None]
    ok = [s for r, s in rows if r["grade"]["reward"]]
    bad = [s for r, s in rows if not r["grade"]["reward"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"средний балл: задача решена {mean(ok):.2f} ({len(ok)}), не решена {mean(bad):.2f} ({len(bad)})")
    high_bad = [r for r, s in rows if s >= 4 and not r["grade"]["reward"]]
    print(f"проваленных попыток с баллом 4–5: {len(high_bad)} из {len(bad)}")
    for r in high_bad[:10]:
        print(f"  {r['task_id']}:{r['trial']}  {'; '.join(r['grade']['diff'][:2]) or 'не сказал: ' + ', '.join(r['grade']['missing_outputs'])}")


if __name__ == "__main__":
    main()
