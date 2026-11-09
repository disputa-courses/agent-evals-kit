"""Магазин «Кладовка»: состояние, инструменты агента и снимок для проверки.

Состояние лежит обычным словарём, как в τ-bench: его легко сбросить,
сравнить и показать. Проверка исхода сравнивает не слова агента, а то,
во что он превратил это состояние.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

DATA = Path(__file__).parent / "data" / "store.json"
RETURN_DAYS = 14

# Что входит в снимок для проверки. Журнал вызовов и тексты передачи
# человеку сюда не попадают: они в трейсе, а исход — это данные магазина.
HASHED = ("customers", "products", "orders", "refunds", "transfers")

_TOOLS: dict[str, dict[str, Any]] = {}


def tool(description: str, params: dict[str, dict[str, Any]], *, writes: bool = False):
    """Регистрирует метод магазина как инструмент агента.

    `params` — поля JSON Schema по именам аргументов. `writes` помечает
    инструменты, которые меняют данные: по ним работают проверки
    траектории («было ли подтверждение перед записью»).
    """

    def wrap(fn: Callable) -> Callable:
        required = [
            name
            for name, p in inspect.signature(fn).parameters.items()
            if name != "self" and p.default is inspect.Parameter.empty
        ]
        _TOOLS[fn.__name__] = {
            "fn": fn,
            "writes": writes,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            },
        }
        return fn

    return wrap


class ToolError(Exception):
    """Отказ инструмента: агент получает текст ошибки, данные не меняются."""


def _digits(phone: str) -> str:
    d = re.sub(r"\D", "", phone or "")
    if len(d) == 11 and d[0] == "8":
        d = "7" + d[1:]
    return d


class Store:
    def __init__(self, data: dict | None = None):
        raw = data if data is not None else json.loads(DATA.read_text(encoding="utf-8"))
        self.data = copy.deepcopy(raw)
        self.today = date.fromisoformat(self.data["today"])
        self.calls: list[dict[str, Any]] = []

    # --- снимок и сравнение -------------------------------------------------

    def snapshot(self) -> dict:
        return {k: self.data[k] for k in HASHED}

    def hash(self) -> str:
        blob = json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def diff(self, other: "Store") -> list[str]:
        """Что отличается в данных: путь и два значения."""
        out: list[str] = []

        def walk(a: Any, b: Any, path: str) -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{path}[{i}]")
            elif a != b:
                out.append(f"{path}: {json.dumps(a, ensure_ascii=False)} → {json.dumps(b, ensure_ascii=False)}")

        walk(self.snapshot(), other.snapshot(), "")
        return out

    # --- вызов инструментов -------------------------------------------------

    @staticmethod
    def tool_schemas() -> list[dict]:
        return [t["schema"] for t in _TOOLS.values()]

    @staticmethod
    def write_tools() -> set[str]:
        return {name for name, t in _TOOLS.items() if t["writes"]}

    def call(self, name: str, arguments: dict | str | None) -> str:
        """Выполняет вызов агента и возвращает текст для модели."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return self._log(name, arguments, error="аргументы не JSON")
        arguments = arguments or {}
        spec = _TOOLS.get(name)
        if spec is None:
            return self._log(name, arguments, error=f"нет инструмента {name}")
        try:
            result = spec["fn"](self, **arguments)
        except ToolError as e:
            return self._log(name, arguments, error=str(e))
        except TypeError as e:
            return self._log(name, arguments, error=f"неверные аргументы: {e}")
        return self._log(name, arguments, result=result)

    def _log(self, name: str, arguments: Any, *, result: Any = None, error: str | None = None) -> str:
        self.calls.append({"name": name, "arguments": arguments, "error": error})
        if error is not None:
            return f"Ошибка: {error}"
        return json.dumps(result, ensure_ascii=False)

    # --- внутренние помощники -----------------------------------------------

    def _customer(self, customer_id: str) -> dict:
        c = self.data["customers"].get(customer_id)
        if c is None:
            raise ToolError(f"покупатель {customer_id} не найден")
        return c

    def _order(self, order_id: str) -> dict:
        o = self.data["orders"].get(order_id)
        if o is None:
            raise ToolError(f"заказ {order_id} не найден")
        return o

    def _items(self, order: dict, item_ids: list[str]) -> list[dict]:
        if not item_ids:
            raise ToolError("не указаны позиции")
        if len(set(item_ids)) != len(item_ids):
            raise ToolError("позиция указана дважды")
        by_id = {i["id"]: i for i in order["items"]}
        missing = [i for i in item_ids if i not in by_id]
        if missing:
            raise ToolError(f"в заказе нет позиций {', '.join(missing)}")
        return [by_id[i] for i in item_ids]

    def _refund(self, order_id: str, order: dict, item_ids: list[str], amount: int, method: str) -> None:
        if method == "gift_card":
            self.data["customers"][order["customer_id"]]["gift_card"] += amount
        # Порядок позиций в просьбе не важен: в данных он один, иначе
        # одинаковые возвраты давали бы разные снимки.
        self.data["refunds"].append(
            {"order_id": order_id, "item_ids": sorted(item_ids), "amount": amount, "method": method}
        )

    # --- инструменты: чтение ------------------------------------------------

    @tool(
        "Найти покупателя по электронной почте. Возвращает customer_id.",
        {"email": {"type": "string", "description": "почта покупателя"}},
    )
    def find_customer_by_email(self, email: str) -> dict:
        for cid, c in self.data["customers"].items():
            if c["email"].lower() == email.strip().lower():
                return {"customer_id": cid}
        raise ToolError("покупатель с такой почтой не найден")

    @tool(
        "Найти покупателя по имени с фамилией и телефону. Возвращает customer_id.",
        {
            "name": {"type": "string", "description": "имя и фамилия"},
            "phone": {"type": "string", "description": "телефон в любом виде"},
        },
    )
    def find_customer_by_name_phone(self, name: str, phone: str) -> dict:
        want = " ".join(name.lower().split())
        for cid, c in self.data["customers"].items():
            if " ".join(c["name"].lower().split()) == want and _digits(c["phone"]) == _digits(phone):
                return {"customer_id": cid}
        raise ToolError("покупатель с таким именем и телефоном не найден")

    @tool(
        "Данные покупателя: имя, адрес, баланс подарочной карты и номера заказов.",
        {"customer_id": {"type": "string"}},
    )
    def get_customer(self, customer_id: str) -> dict:
        c = self._customer(customer_id)
        orders = sorted(oid for oid, o in self.data["orders"].items() if o["customer_id"] == customer_id)
        return {"customer_id": customer_id, **c, "orders": orders}

    @tool(
        "Заказ целиком: статус, даты, способ оплаты, адрес, позиции с ценами.",
        {"order_id": {"type": "string", "description": "номер вида O1001"}},
    )
    def get_order(self, order_id: str) -> dict:
        o = self._order(order_id)
        items = [
            {**i, "name": self.data["products"][i["product_id"]]["name"]} for i in o["items"]
        ]
        return {"order_id": order_id, **{k: v for k, v in o.items() if k != "items"}, "items": items}

    @tool(
        "Товар: название, категория, цена, остаток на складе.",
        {"product_id": {"type": "string"}},
    )
    def get_product(self, product_id: str) -> dict:
        p = self.data["products"].get(product_id)
        if p is None:
            raise ToolError(f"товар {product_id} не найден")
        return {"product_id": product_id, **p}

    # --- инструменты: запись ------------------------------------------------

    @tool(
        "Отменить заказ целиком. Только для заказов в статусе processing. "
        "Деньги возвращаются тем же способом, каким платили.",
        {"order_id": {"type": "string"}},
        writes=True,
    )
    def cancel_order(self, order_id: str) -> dict:
        o = self._order(order_id)
        if o["status"] != "processing":
            raise ToolError(f"заказ в статусе {o['status']}, отменить можно только processing")
        o["status"] = "cancelled"
        for item in o["items"]:
            self.data["products"][item["product_id"]]["stock"] += 1
        ids = [i["id"] for i in o["items"]]
        amount = sum(i["price"] for i in o["items"])
        self._refund(order_id, o, ids, amount, o["payment"])
        return {"order_id": order_id, "status": "cancelled", "refund": amount, "method": o["payment"]}

    @tool(
        "Сменить адрес доставки. Только для заказов в статусе processing.",
        {"order_id": {"type": "string"}, "address": {"type": "string", "description": "новый адрес целиком"}},
        writes=True,
    )
    def change_address(self, order_id: str, address: str) -> dict:
        o = self._order(order_id)
        if o["status"] != "processing":
            raise ToolError(f"заказ в статусе {o['status']}, адрес меняется только у processing")
        o["address"] = address.strip()
        return {"order_id": order_id, "address": o["address"]}

    @tool(
        "Оформить возврат позиций доставленного заказа. Возврат возможен "
        f"в течение {RETURN_DAYS} дней после доставки. refund_method: "
        "original — тем же способом, каким платили; gift_card — на подарочную карту.",
        {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}, "description": "позиции вида O1001-2"},
            "refund_method": {"type": "string", "enum": ["original", "gift_card"]},
        },
        writes=True,
    )
    def return_items(self, order_id: str, item_ids: list[str], refund_method: str) -> dict:
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ToolError(f"заказ в статусе {o['status']}, вернуть можно только доставленный")
        days = (self.today - date.fromisoformat(o["delivered"])).days
        if days > RETURN_DAYS:
            raise ToolError(f"с доставки прошло {days} дн., возврат — в течение {RETURN_DAYS}")
        if refund_method not in ("original", "gift_card"):
            raise ToolError("refund_method — original или gift_card")
        items = self._items(o, item_ids)
        done = [i["id"] for i in items if i["status"] != "ok"]
        if done:
            raise ToolError(f"позиции уже возвращены или обменены: {', '.join(done)}")
        for i in items:
            i["status"] = "returned"
        amount = sum(i["price"] for i in items)
        method = o["payment"] if refund_method == "original" else "gift_card"
        self._refund(order_id, o, list(item_ids), amount, method)
        return {"order_id": order_id, "returned": sorted(item_ids), "refund": amount, "method": method}

    @tool(
        "Передать разговор сотруднику. Только если просьбу нельзя решить "
        "инструментами или покупатель прямо просит человека.",
        {
            "customer_id": {"type": "string", "description": "кого передаём; пусто, если не опознан"},
            "summary": {"type": "string", "description": "суть обращения для сотрудника"},
        },
        writes=True,
    )
    def transfer_to_human(self, customer_id: str, summary: str) -> dict:
        # В снимок идёт только факт передачи: текст сводки проверяют
        # по трейсу, иначе любая перефразировка ломала бы сравнение.
        self.data["transfers"].append({"customer_id": customer_id or None})
        return {"transferred": True}
