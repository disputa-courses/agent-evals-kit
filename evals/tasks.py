"""Задачи эвала: кто пишет, чего хочет и каким должен стать магазин."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"{p}: повтор id {sorted(dup)}")
    return rows
