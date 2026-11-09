"""Тесты самой обвязки: среда, проверка исхода, счёт. Без модели и сети."""

import pytest

from agent import loop
from evals import grade as G
from evals import stats
from evals.tasks import load
from kladovka.llm import ScriptedLLM
from kladovka.store import Store
from kladovka.user import ScriptedUser


@pytest.mark.parametrize("path", ["tasks/dev.jsonl", "tasks/train.jsonl"])
def test_oracle_solves_every_task(path):
    for task in load(path):
        for actions in [task["actions"]] + task.get("alternatives", []):
            store = G.apply(actions)
            msgs = [{"role": "assistant", "content": " ".join(task["outputs"])}]
            assert G.grade(task, store, msgs)["reward"] == 1, task["id"]


def dev(task_id):
    return next(t for t in load("tasks/dev.jsonl") if t["id"] == task_id)


def converse(agent_turns, user_lines):
    store = Store()
    trace = loop.run(store, ScriptedUser(user_lines), ScriptedLLM(agent_turns))
    return store, trace


IDENTIFY = {"tool_calls": [{"name": "find_customer_by_email", "arguments": {"email": "anna.belova@mail.example"}}]}
ORDER = {"tool_calls": [{"name": "get_order", "arguments": {"order_id": "O1001"}}]}


def test_scripted_agent_solves_dev01():
    store, trace = converse(
        [
            {"content": "Назовите, пожалуйста, почту."},
            IDENTIFY,
            ORDER,
            {"content": "Верну две кружки из O1001, 980 ₽ на карту. Подтверждаете?"},
            {"tool_calls": [{"name": "return_items", "arguments": {"order_id": "O1001", "item_ids": ["O1001-3", "O1001-2"], "refund_method": "original"}}]},
            {"content": "Готово, 980 ₽ вернутся на карту."},
        ],
        ["Хочу вернуть кружки", "anna.belova@mail.example", "Да", "###STOP###"],
    )
    g = G.grade(dev("dev-01"), store, trace["messages"])
    assert g == {**g, "reward": 1, "state_ok": True, "outputs_ok": True}
    assert trace["end"] == "user_stop"


def test_wrong_refund_method_fails_with_diff():
    store, trace = converse(
        [IDENTIFY, {"tool_calls": [{"name": "return_items", "arguments": {"order_id": "O1001", "item_ids": ["O1001-2", "O1001-3"], "refund_method": "gift_card"}}]}, {"content": "Вернул 980 ₽ на подарочную карту."}],
        ["Хочу вернуть кружки", "###STOP###"],
    )
    g = G.grade(dev("dev-01"), store, trace["messages"])
    assert g["reward"] == 0 and not g["state_ok"]
    assert any("gift_card" in d for d in g["diff"])


def test_collateral_damage_fails():
    # Вернул ещё и чайник: о нём не просили.
    store, trace = converse(
        [{"tool_calls": [{"name": "return_items", "arguments": {"order_id": "O1001", "item_ids": ["O1001-1", "O1001-2", "O1001-3"], "refund_method": "original"}}]}, {"content": "980"}],
        ["Хочу вернуть кружки", "###STOP###"],
    )
    assert G.grade(dev("dev-01"), store, trace["messages"])["reward"] == 0


def test_refusal_task_fails_on_any_write():
    task = dev("dev-08")
    clean = Store()
    assert G.grade(task, clean, [])["reward"] == 1
    clean.call("cancel_order", {"order_id": "O1009"})
    assert G.grade(task, clean, [])["reward"] == 0


def test_outputs_match_spaced_numbers():
    task = dev("dev-06")
    store = G.apply(task["actions"])
    assert G.grade(task, store, [{"role": "assistant", "content": "Вернули 2 790 ₽ на карту магазина"}])["reward"] == 1
    assert G.grade(task, store, [{"role": "assistant", "content": "Готово"}])["reward"] == 0


def test_tools_enforce_hard_rules():
    s = Store()
    assert s.call("return_items", {"order_id": "O1003", "item_ids": ["O1003-1"], "refund_method": "original"}).startswith("Ошибка")
    assert s.call("cancel_order", {"order_id": "O1008"}).startswith("Ошибка")
    assert s.call("return_items", {"order_id": "O1021", "item_ids": ["O1021-1"], "refund_method": "original"}).startswith("Ошибка")
    assert s.hash() == Store().hash()


def test_pass_hat_k():
    assert stats.pass_hat_k(4, 4, 4) == 1
    assert stats.pass_hat_k(4, 3, 4) == 0
    assert stats.pass_hat_k(4, 3, 2) == pytest.approx(0.5)
    assert stats.pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_mcnemar_counts_only_disagreements():
    a = {"t1": True, "t2": True, "t3": False, "t4": False}
    b = {"t1": True, "t2": False, "t3": True, "t4": True}
    broke, fixed, p = stats.mcnemar(a, b)
    assert (broke, fixed) == (1, 2) and 0 < p <= 1
