"""Разметка трейсов: ваш разбор сбоев в одном файле analysis/labels.csv.

    python -m evals.analysis init runs/base     заготовка: строка на трейс
    python -m evals.analysis split              раздать train/dev/test для судьи
    python -m evals.analysis summary            счёт видов сбоев и сверка с проверкой исхода

Колонки, которые заполняете вы:
  verdict      ok или problem — ваше суждение по трейсу целиком
  first_error  первая ошибка своими словами (open coding)
  mode         вид сбоя, короткий ключ из вашей таксономии (axial coding)
  judge        yes или no — есть ли в трейсе то, что ловит ваш судья
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from .report import load_run
from .tasks import ROOT

LABELS = ROOT / "analysis" / "labels.csv"
FIELDS = ["trace", "task", "outcome", "verdict", "first_error", "mode", "judge", "split", "note"]


def read(path: Path | None = None) -> list[dict]:
    path = path or LABELS
    if not path.exists():
        raise SystemExit(f"нет {path.relative_to(ROOT)}: сначала python -m evals.analysis init <прогон>")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(rows: list[dict], path: Path | None = None) -> None:
    path = path or LABELS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def init(run: str) -> None:
    results, _ = load_run(run)
    rows = read() if LABELS.exists() else []
    have = {r["trace"] for r in rows}
    added = 0
    for r in results:
        tid = f"{Path(run).name}/{r['task_id']}:{r['trial']}"
        if tid in have:
            continue
        rows.append({"trace": tid, "task": r["task_id"], "outcome": "pass" if r["grade"]["reward"] else "fail"})
        added += 1
    write(rows)
    print(f"добавлено строк: {added}, всего {len(rows)} → {LABELS.relative_to(ROOT)}")


def split(seed: int, shares: tuple[float, float]) -> None:
    rows = read()
    labeled = [r for r in rows if r.get("judge") in ("yes", "no")]
    rnd = random.Random(seed)
    # Делим отдельно yes и no, чтобы в каждой части были оба ответа.
    for value in ("yes", "no"):
        part = [r for r in labeled if r["judge"] == value]
        rnd.shuffle(part)
        n_train = round(len(part) * shares[0])
        n_dev = round(len(part) * shares[1])
        for i, r in enumerate(part):
            r["split"] = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
    write(rows)
    c = Counter((r.get("split"), r.get("judge")) for r in labeled)
    print("разбиение (часть, судья): " + ", ".join(f"{s}/{j} {n}" for (s, j), n in sorted(c.items())))


def summary() -> None:
    rows = read()
    done = [r for r in rows if r.get("verdict") in ("ok", "problem")]
    print(f"размечено {len(done)} из {len(rows)}")
    problems = [r for r in done if r["verdict"] == "problem"]
    unnamed = [r["trace"] for r in problems if not r.get("mode")]
    if unnamed:
        print(f"у {len(unnamed)} проблемных трейсов нет вида сбоя: {', '.join(unnamed[:5])}")
    print("\nвиды сбоев:")
    for mode, n in Counter(r["mode"] for r in problems if r.get("mode")).most_common():
        print(f"  {mode:<30} {n}")
    agree = Counter((r["outcome"], r["verdict"]) for r in done)
    print("\nпроверка исхода против вашего суждения:")
    print(f"  pass и ok       {agree[('pass', 'ok')]}")
    print(f"  fail и problem  {agree[('fail', 'problem')]}")
    print(f"  pass, но problem {agree[('pass', 'problem')]}   ← проверка исхода этого не видит: кандидаты в проверки траектории и судью")
    print(f"  fail, но ok      {agree[('fail', 'ok')]}   ← посмотрите задачу и эталон: не ошибка ли в них")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("run")
    p = sub.add_parser("split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=float, default=0.15)
    p.add_argument("--dev", type=float, default=0.40)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        init(a.run)
    elif a.cmd == "split":
        split(a.seed, (a.train, a.dev))
    else:
        summary()


if __name__ == "__main__":
    main()
