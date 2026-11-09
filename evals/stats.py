"""Счёт надёжности и сравнение версий.

pass^k — вероятность, что задача решится во всех k попытках подряд
(τ-bench). Из n попыток с c успехами её несмещённая оценка — C(c,k)/C(n,k).
pass@k — хотя бы одна из k; для агента в поддержке она обманчива:
покупатель приходит один раз.
"""

from __future__ import annotations

import random
from math import comb


def pass_hat_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} больше числа попыток n={n}")
    return comb(c, k) / comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} больше числа попыток n={n}")
    return 1.0 - comb(n - c, k) / comb(n, k)


def by_task(results: list[dict]) -> dict[str, tuple[int, int]]:
    """task_id → (попыток, успехов)."""
    out: dict[str, list[int]] = {}
    for r in results:
        n_c = out.setdefault(r["task_id"], [0, 0])
        n_c[0] += 1
        n_c[1] += int(r["grade"]["reward"])
    return {t: (n, c) for t, (n, c) in out.items()}


def mean_pass_hat(results: list[dict], k: int) -> float:
    per = by_task(results)
    return sum(pass_hat_k(n, c, k) for n, c in per.values()) / len(per)


def bootstrap_diff(a: list[dict], b: list[dict], k: int, *, iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Разница pass^k (b − a) и 95%-интервал бутстрепом по задачам.

    Задачи — единица выборки: попытки одной задачи зависимы, и
    пересэмплировать их по одной значит завысить уверенность.
    """
    pa, pb = by_task(a), by_task(b)
    tasks = sorted(set(pa) & set(pb))
    if not tasks:
        raise ValueError("нет общих задач")
    da = {t: pass_hat_k(*pa[t], k) for t in tasks}
    db = {t: pass_hat_k(*pb[t], k) for t in tasks}
    point = sum(db[t] - da[t] for t in tasks) / len(tasks)
    rnd = random.Random(seed)
    diffs = []
    for _ in range(iters):
        sample = [rnd.choice(tasks) for _ in tasks]
        diffs.append(sum(db[t] - da[t] for t in sample) / len(sample))
    diffs.sort()
    return point, diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Точный тест Макнемара по парам «задача: решена ли».

    Возвращает (сломалось, починилось, p): считаются только задачи,
    где версии разошлись.
    """
    tasks = set(a) & set(b)
    broke = sum(1 for t in tasks if a[t] and not b[t])
    fixed = sum(1 for t in tasks if not a[t] and b[t])
    n = broke + fixed
    if n == 0:
        return broke, fixed, 1.0
    tail = sum(comb(n, i) for i in range(0, min(broke, fixed) + 1)) / 2**n
    return broke, fixed, min(1.0, 2 * tail)
