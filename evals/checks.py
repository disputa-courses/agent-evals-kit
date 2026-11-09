"""Проверки траектории: код, который находит виды сбоев по трейсу.

Каждая проверка — функция от трейса, которая возвращает список находок
(пусто — сбоя нет). Пишите их по своей таксономии сбоев из разбора
трейсов, а не по списку «полезных метрик». Пример ниже — образец формы.

    python -m evals.checks runs/base           таблица по проверкам
    python -m evals.checks runs/base --hits    где сработали
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Callable

from .report import load_run
from .trace import Trace

Check = Callable[[Trace], list[str]]


def repeated_lookup(t: Trace) -> list[str]:
    """Образец: один и тот же вызов чтения с теми же аргументами дважды."""
    seen = Counter((s.tool, repr(sorted((s.arguments or {}).items()))) for s in t.tool_calls if s.tool.startswith("get_"))
    return [f"{tool} повторён {n} раз" for (tool, _), n in seen.items() if n > 1]


# Ваши проверки — сюда. Ключ — имя вида сбоя из вашей таксономии.
CHECKS: dict[str, Check] = {
    "repeated_lookup": repeated_lookup,
}


def run_checks(traces: list[Trace], checks: dict[str, Check] | None = None) -> dict[str, dict[str, list[str]]]:
    """Имя проверки → {id трейса: находки} только там, где сработала."""
    out: dict[str, dict[str, list[str]]] = {}
    for name, fn in (checks or CHECKS).items():
        out[name] = {}
        for t in traces:
            found = fn(t)
            if found:
                out[name][t.id] = found
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--hits", action="store_true")
    a = ap.parse_args(argv)
    results, _ = load_run(a.run)
    traces = [Trace(r) for r in results]
    failed = {t.id for t in traces if not t.reward}
    hits = run_checks(traces)
    print(f"трейсов {len(traces)}, провалено проверкой исхода {len(failed)}")
    print(f"{'проверка':<28} {'сработала':>9} {'на проваленных':>15} {'на успешных':>12}")
    for name, where in hits.items():
        on_failed = len(set(where) & failed)
        print(f"{name:<28} {len(where):>9} {on_failed:>15} {len(where) - on_failed:>12}")
        if a.hits:
            for tid, found in sorted(where.items()):
                print(f"    {tid}: {'; '.join(found)}")


if __name__ == "__main__":
    main()
