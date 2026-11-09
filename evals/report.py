"""Отчёт прогона и сравнение с базой.

    python -m evals.report runs/new --baseline runs/base
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import stats
from .tasks import ROOT


def load_run(path: str | Path) -> tuple[list[dict], dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    results = [json.loads(line) for line in (p / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8")) if (p / "meta.json").exists() else {}
    return results, meta


def _k_of(results: list[dict]) -> int:
    return min(n for n, _ in stats.by_task(results).values())


def print_report(results: list[dict], meta: dict, *, k: int | None = None) -> None:
    k = k or _k_of(results)
    per = stats.by_task(results)
    fam = {r["task_id"]: r.get("family") or "—" for r in results}
    print(f"агент {meta.get('agent_version')}  модель {meta.get('agent_model') or ('заглушка' if meta.get('stub') else 'эталон')}  задач {len(per)}  попыток на задачу {k}")
    print(f"{'задача':<12} {'семья':<18} {'успехов':>8}  {'pass^1':>6}  {'pass^' + str(k):>6}")
    for t in sorted(per):
        n, c = per[t]
        print(f"{t:<12} {fam[t]:<18} {c:>4}/{n:<3}  {stats.pass_hat_k(n, c, 1):>6.2f}  {stats.pass_hat_k(n, c, k):>6.2f}")
    by_fam: dict[str, list[str]] = defaultdict(list)
    for t in per:
        by_fam[fam[t]].append(t)
    print()
    for f, ts in sorted(by_fam.items()):
        print(f"  {f:<18} pass^1 {sum(stats.pass_hat_k(*per[t], 1) for t in ts) / len(ts):.2f}  pass^{k} {sum(stats.pass_hat_k(*per[t], k) for t in ts) / len(ts):.2f}")
    line = "  ".join(f"pass^{i} {stats.mean_pass_hat(results, i):.2f}" for i in range(1, k + 1))
    print(f"\nвсего: {line}")
    tok = [r.get("usage", {}) for r in results]
    calls = sum(u.get("model_calls", 0) for u in tok)
    if calls:
        print(
            f"на попытку: обращений к модели {calls / len(results):.1f}, "
            f"токенов на входе {sum(u.get('prompt_tokens', 0) for u in tok) / len(results):.0f}, "
            f"на выходе {sum(u.get('completion_tokens', 0) for u in tok) / len(results):.0f}"
        )
    ends = defaultdict(int)
    for r in results:
        ends[r["end"]] += 1
    print("чем кончились попытки:", ", ".join(f"{e} {n}" for e, n in sorted(ends.items())))


def compare(base: list[dict], new: list[dict]) -> None:
    k = min(_k_of(base), _k_of(new))
    pb, pn = stats.by_task(base), stats.by_task(new)
    common = sorted(set(pb) & set(pn))
    print(f"\nсравнение с базой по {len(common)} общим задачам, k={k}")
    for t in common:
        a, b = stats.pass_hat_k(*pb[t], k), stats.pass_hat_k(*pn[t], k)
        if abs(a - b) > 1e-9:
            print(f"  {t:<12} pass^{k} {a:.2f} → {b:.2f}")
    point, lo, hi = stats.bootstrap_diff(base, new, k)
    print(f"разница pass^{k}: {point:+.2f}, 95% интервал [{lo:+.2f}; {hi:+.2f}]")
    solid = lambda per: {t: per[t][1] == per[t][0] for t in common}  # noqa: E731
    broke, fixed, p = stats.mcnemar(solid(pb), solid(pn))
    print(f"задачи «решена во всех попытках»: сломалось {broke}, починилось {fixed}, p Макнемара {p:.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--baseline")
    ap.add_argument("--k", type=int)
    a = ap.parse_args(argv)
    results, meta = load_run(a.run)
    print_report(results, meta, k=a.k)
    if a.baseline:
        base, _ = load_run(a.baseline)
        compare(base, results)


if __name__ == "__main__":
    main()
