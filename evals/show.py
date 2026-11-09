"""Трейс глазами: разговор, вызовы инструментов, итог проверки.

    python -m evals.show runs/base dev-03          все попытки задачи
    python -m evals.show runs/base dev-03 --trial 1
    python -m evals.show runs/base --failed        только проваленные
"""

from __future__ import annotations

import argparse
import json
import textwrap

from .report import load_run


def render(r: dict) -> str:
    g = r["grade"]
    head = f"═══ {r['task_id']} попытка {r['trial']}  итог {'✓' if g['reward'] else '✗'}  ({r['end']})"
    lines = [head]
    for m in r["messages"]:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            lines.append(f"    ← {m.get('name')}: {textwrap.shorten(m['content'], 300)}")
            continue
        if m.get("content"):
            who = {"user": "ПОКУПАТЕЛЬ", "assistant": "АГЕНТ"}.get(role, role)
            lines.append(f"{who}: {m['content']}")
        for c in m.get("tool_calls") or []:
            lines.append(f"    → {c['function']['name']}({c['function']['arguments']})")
    if r.get("error"):
        lines.append(f"СБОЙ: {r['error']}")
    if not g["state_ok"]:
        lines.append("данные расходятся с эталоном:")
        lines += [f"    {d}" for d in g["diff"][:12]]
    if g["missing_outputs"]:
        lines.append(f"не сказал покупателю: {', '.join(g['missing_outputs'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("task", nargs="?")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--json", action="store_true", help="сырые записи")
    a = ap.parse_args(argv)
    results, _ = load_run(a.run)
    for r in results:
        if a.task and r["task_id"] != a.task:
            continue
        if a.trial is not None and r["trial"] != a.trial:
            continue
        if a.failed and r["grade"]["reward"]:
            continue
        print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else render(r))
        print()


if __name__ == "__main__":
    main()
