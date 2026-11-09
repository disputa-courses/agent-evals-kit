"""Судья на модели: один вид сбоя, ответ «да» или «нет», сверка с вашей разметкой.

    python -m evals.judge run runs/base                    ответы судьи → analysis/judge.csv
    python -m evals.judge calibrate --split dev            TPR и TNR против вашей разметки
    python -m evals.judge estimate runs/new --split test   доля сбоя в прогоне с поправкой на ошибки судьи

Промпт — evals/judge_prompt.md. Примеры из части train подставляются
в {examples} сами. Подбирайте промпт на dev; test смотрите один раз,
в конце: на нём и видно, не подогнали ли судью.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import analysis
from .report import load_run
from .tasks import ROOT
from .trace import conversation

PROMPT = Path(__file__).parent / "judge_prompt.md"
OUT = ROOT / "analysis" / "judge.csv"


def examples(rows: list[dict], records: dict[str, dict]) -> str:
    train = [r for r in rows if r.get("split") == "train" and r["trace"] in records]
    if not train:
        return ""
    parts = ["Примеры с ответами:"]
    for r in train:
        parts.append(f"--- пример ---\n{conversation(records[r['trace']])}\nОтвет: {r['judge']}" + (f" ({r['note']})" if r.get("note") else ""))
    return "\n\n".join(parts)


def ask(llm, prompt: str) -> tuple[str, str]:
    text = llm.chat([{"role": "user", "content": prompt}]).message.get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    answer = str(data.get("answer", "")).strip().lower()
    return (answer if answer in ("yes", "no") else "unknown"), str(data.get("reason", text[:200]))


def records_of(run: str) -> dict[str, dict]:
    results, _ = load_run(run)
    return {f"{Path(run).name}/{r['task_id']}:{r['trial']}": r for r in results}


def run(run_dir: str, llm=None) -> None:
    from kladovka import llm as L

    llm = llm or L.from_env("judge")
    recs = records_of(run_dir)
    rows = analysis.read() if analysis.LABELS.exists() else []
    template = PROMPT.read_text(encoding="utf-8")
    ex = examples(rows, recs)
    train_ids = {r["trace"] for r in rows if r.get("split") == "train"}
    todo = [tid for tid in recs if tid not in train_ids]

    def one(tid: str) -> tuple[str, str, str]:
        a, why = ask(llm, template.replace("{examples}", ex).replace("{conversation}", conversation(recs[tid])))
        return tid, a, why

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(one, todo))
    old = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8", newline="") as f:
            old = {r["trace"]: r for r in csv.DictReader(f)}
    for tid, a, why in answers:
        old[tid] = {"trace": tid, "answer": a, "reason": why}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "answer", "reason"])
        w.writeheader()
        w.writerows(old.values())
    unknown = sum(1 for _, a, _ in answers if a == "unknown")
    print(f"ответов судьи: {len(answers)}, не разобрать: {unknown} → {OUT.relative_to(ROOT)}")


def rates(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (ваша разметка, ответ судьи). «yes» — сбой есть."""
    tp = sum(1 for h, j in pairs if h == "yes" and j == "yes")
    fn = sum(1 for h, j in pairs if h == "yes" and j != "yes")
    tn = sum(1 for h, j in pairs if h == "no" and j == "no")
    fp = sum(1 for h, j in pairs if h == "no" and j != "no")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "tpr": tpr, "tnr": tnr}


def corrected(p_obs: float, tpr: float, tnr: float) -> float:
    """Поправка Рогана — Гладена: доля сбоя с учётом ошибок судьи."""
    denom = tpr + tnr - 1
    if denom <= 0:
        raise ValueError("судья не лучше монетки (TPR + TNR ≤ 1): поправка не имеет смысла")
    return min(1.0, max(0.0, (p_obs + tnr - 1) / denom))


def judged() -> dict[str, str]:
    if not OUT.exists():
        raise SystemExit(f"нет {OUT.relative_to(ROOT)}: сначала python -m evals.judge run <прогон>")
    with OUT.open(encoding="utf-8", newline="") as f:
        return {r["trace"]: r["answer"] for r in csv.DictReader(f)}


def calibrate(split: str) -> dict:
    rows = [r for r in analysis.read() if r.get("split") == split and r.get("judge") in ("yes", "no")]
    j = judged()
    pairs = [(r["judge"], j[r["trace"]]) for r in rows if r["trace"] in j]
    if not pairs:
        raise SystemExit(f"в части {split} нет размеченных трейсов с ответом судьи")
    m = rates(pairs)
    print(f"часть {split}: {len(pairs)} трейсов, сбой по вашей разметке в {m['tp'] + m['fn']}")
    print(f"TPR {m['tpr']:.2f} (нашёл {m['tp']} из {m['tp'] + m['fn']} сбоев)   TNR {m['tnr']:.2f} (не тронул {m['tn']} из {m['tn'] + m['fp']} чистых)")
    return m


def estimate(run_dir: str, split: str) -> None:
    m = calibrate(split)
    recs = records_of(run_dir)
    j = judged()
    answers = [j[t] for t in recs if t in j and j[t] in ("yes", "no")]
    if not answers:
        raise SystemExit("по этому прогону ответов судьи нет: python -m evals.judge run " + run_dir)
    p = sum(a == "yes" for a in answers) / len(answers)
    print(f"судья говорит «сбой» в {p:.0%} трейсов прогона; с поправкой на его ошибки — {corrected(p, m['tpr'], m['tnr']):.0%}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("run")
    p = sub.add_parser("calibrate")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p = sub.add_parser("estimate")
    p.add_argument("run")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.run)
    elif a.cmd == "calibrate":
        calibrate(a.split)
    else:
        estimate(a.run, a.split)


if __name__ == "__main__":
    main()
