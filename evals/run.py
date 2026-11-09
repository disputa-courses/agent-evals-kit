"""Прогон агента на задачах: k попыток на задачу, трейсы и счёт.

    python -m evals.run --tasks tasks/dev.jsonl --k 4 --out runs/base
    python -m evals.run --tasks tasks/dev.jsonl --oracle --out runs/oracle

--oracle вместо агента выполняет эталон задачи: так проверяется, что
задача решаема и проверка её засчитывает (как oracle в Harbor).
--stub ставит агента-заглушку: он ничего не делает и отвечает «Готово».
Задача, где нужно действие, а заглушка её проходит, ничего не меряет.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kladovka.store import Store

from . import grade as G
from . import report
from .tasks import ROOT, load


def agent_version() -> str:
    """Отпечаток агента: что именно мерили. Меняется с любой правкой agent/."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "agent").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".md", ".json", ".yaml", ".txt"):
            h.update(p.relative_to(ROOT).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def oracle_trial(task: dict) -> dict:
    store = G.apply(task.get("actions", []))
    text = "Готово. " + " ".join(task.get("outputs", []))
    messages = [{"role": "assistant", "content": text}]
    return {"messages": messages, "usage": {}, "end": "oracle", "seconds": 0, "_store": store}


def stub_trial(task: dict) -> dict:
    messages = [{"role": "assistant", "content": "Готово."}]
    return {"messages": messages, "usage": {}, "end": "stub", "seconds": 0, "_store": Store()}


def agent_trial(task: dict) -> dict:
    from agent import loop
    from kladovka import llm
    from kladovka.user import UserSim

    store = Store()
    user = UserSim(task["user"]["instructions"], llm.from_env("user"))
    out = loop.run(store, user, llm.from_env("agent"))
    out["_store"] = store
    return out


def one(task: dict, trial: int, mode: str) -> dict:
    try:
        out = {"oracle": oracle_trial, "stub": stub_trial}.get(mode, agent_trial)(task)
        store = out.pop("_store")
        out["grade"] = G.grade(task, store, out["messages"])
        out["calls"] = store.calls
    except G.BadTask:
        raise
    except Exception as e:  # сбой модели или цикла — тоже результат, а не падение прогона
        out = {"messages": [], "usage": {}, "end": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3),
               "grade": {"reward": 0, "state_ok": False, "outputs_ok": False, "missing_outputs": [], "diff": []}, "calls": []}
    return {"task_id": task["id"], "family": task.get("family"), "trial": trial, **out}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=1, help="попыток на задачу")
    ap.add_argument("--out", required=True, help="папка прогона, например runs/base")
    ap.add_argument("--only", help="id задач через запятую")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--oracle", action="store_true", help="выполнить эталон вместо агента")
    ap.add_argument("--stub", action="store_true", help="агент-заглушка: ничего не делает")
    ap.add_argument("--note", default="", help="пометка: что поменяли")
    a = ap.parse_args(argv)

    tasks = load(a.tasks)
    if a.only:
        want = set(a.only.split(","))
        tasks = [t for t in tasks if t["id"] in want]
    for t in tasks:  # сломанная задача видна до трат на модель
        G.expected(t)

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    import os
    meta = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": a.tasks,
        "k": a.k,
        "oracle": a.oracle,
        "stub": a.stub,
        "agent_model": None if (a.oracle or a.stub) else os.environ.get("AGENT_MODEL"),
        "user_model": None if (a.oracle or a.stub) else (os.environ.get("USER_MODEL") or os.environ.get("AGENT_MODEL")),
        "agent_version": agent_version(),
        "git": git_sha(),
        "note": a.note,
    }
    jobs = [(t, i) for t in tasks for i in range(a.k)]
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        mode = "oracle" if a.oracle else "stub" if a.stub else "agent"
        results = list(ex.map(lambda j: one(j[0], j[1], mode), jobs))
    with (out / "traces.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report.print_report(results, meta, k=a.k)
    errors = sum(1 for r in results if r["end"] == "error")
    if errors:
        print(f"\nсбоев: {errors}; первый: {next(r['error'] for r in results if r['end'] == 'error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
