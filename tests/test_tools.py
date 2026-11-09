"""Тесты разметки, проверок траектории и судьи — на сценариях, без сети."""

import json

import pytest

from agent import loop
from evals import analysis, judge
from evals import grade as G
from evals.checks import run_checks
from evals.tasks import load
from evals.trace import Trace, conversation
from kladovka.llm import ScriptedLLM
from kladovka.store import Store
from kladovka.user import ScriptedUser


def record(task_id, agent_turns, user_lines, trial=0):
    task = next(t for t in load("tasks/dev.jsonl") if t["id"] == task_id)
    store = Store()
    out = loop.run(store, ScriptedUser(user_lines), ScriptedLLM(agent_turns))
    return {"task_id": task_id, "trial": trial, **out, "grade": G.grade(task, store, out["messages"]), "calls": store.calls}


RETURN = {"tool_calls": [{"name": "return_items", "arguments": {"order_id": "O1001", "item_ids": ["O1001-2", "O1001-3"], "refund_method": "original"}}]}


@pytest.fixture
def run_dir(tmp_path):
    rows = [
        # Сделал верно, но без подтверждения: проверка исхода это пропустит.
        record("dev-01", [RETURN, {"content": "Вернул 980 ₽."}], ["Верните кружки", "###STOP###"], 0),
        record("dev-01", [{"content": "Подтверждаете возврат 980 ₽?"}, RETURN, {"content": "Готово, 980 ₽."}], ["Верните кружки", "Да", "###STOP###"], 1),
    ]
    d = tmp_path / "base"
    d.mkdir()
    (d / "traces.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (d / "meta.json").write_text("{}", encoding="utf-8")
    return d


def write_without_confirmation(t: Trace):
    found = []
    for w in t.writes:
        prior_user = [s for s in t.before(w) if s.role == "user"]
        if not prior_user or not any(x in prior_user[-1].text.lower() for x in ("да", "подтверж")):
            found.append(f"{w.tool} без согласия")
    return found


def test_trace_sees_writes_and_order(run_dir):
    traces = [Trace(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text().splitlines()]
    assert all(t.reward == 1 for t in traces)
    hits = run_checks(traces, {"write_without_confirmation": write_without_confirmation})
    assert set(hits["write_without_confirmation"]) == {"dev-01:0"}


def test_conversation_hides_grade(run_dir):
    rec = json.loads((run_dir / "traces.jsonl").read_text().splitlines()[0])
    text = conversation(rec)
    assert "reward" not in text and "Правила поддержки" not in text and "return_items" in text


def test_labels_split_and_judge_calibration(run_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "LABELS", tmp_path / "labels.csv")
    monkeypatch.setattr(judge, "OUT", tmp_path / "judge.csv")
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(judge, "ROOT", tmp_path)
    analysis.init(str(run_dir))
    rows = analysis.read()
    assert [r["outcome"] for r in rows] == ["pass", "pass"]
    rows[0].update(verdict="problem", mode="no_confirmation", judge="yes", split="dev")
    rows[1].update(verdict="ok", judge="no", split="dev")
    analysis.write(rows)
    llm = ScriptedLLM([{"content": '{"reason": "не спросил", "answer": "yes"}'}, {"content": '{"reason": "спросил", "answer": "no"}'}])
    judge.run(str(run_dir), llm=llm)
    m = judge.calibrate("dev")
    # Судья мог ответить в любом порядке потоков: проверяем только, что счёт сошёлся.
    assert m["tp"] + m["fn"] == 1 and m["tn"] + m["fp"] == 1


def test_rates_and_correction():
    m = judge.rates([("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", "no")])
    assert (m["tpr"], m["tnr"]) == (0.5, 1.0)
    assert judge.corrected(0.25, 0.5, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        judge.corrected(0.5, 0.5, 0.5)


def test_stub_passes_only_tasks_without_actions_and_outputs():
    from evals import run as R

    for t in load("tasks/dev.jsonl"):
        out = R.one(t, 0, "stub")
        expected = int(not t["actions"] and not t["outputs"])
        assert out["grade"]["reward"] == expected, t["id"]


def test_second_judge_writes_its_own_file(run_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "LABELS", tmp_path / "labels.csv")
    monkeypatch.setattr(judge, "OUT", tmp_path / "judge.csv")
    bad = tmp_path / "bad_prompt.md"
    bad.write_text("Хорош ли ответ?\n{conversation}\n", encoding="utf-8")
    llm = ScriptedLLM([{"content": '{"answer": "no"}'}, {"content": '{"answer": "no"}'}])
    judge.run(str(run_dir), llm=llm, prompt=bad, out=tmp_path / "judge-bad.csv")
    assert (tmp_path / "judge-bad.csv").exists() and not (tmp_path / "judge.csv").exists()
    assert set(judge.judged(tmp_path / "judge-bad.csv").values()) == {"no"}
