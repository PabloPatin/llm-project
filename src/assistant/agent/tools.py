"""Инструменты агента: GraphRAG и ERP (FR-6)."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART_ID_RE = re.compile(r"^P-\d{4}[A-Z]?$", re.IGNORECASE)


def _is_part_id(value: object) -> bool:
    return isinstance(value, str) and bool(_PART_ID_RE.fullmatch(value.strip()))


def _answer_to_dict(answer: Answer | dict, *, include_facts: bool) -> dict:
    """Приводит Answer к простому словарю, который ожидает LLM function calling."""
    if isinstance(answer, dict):
        result = {
            "answer": str(answer.get("answer", answer.get("text", ""))),
            "sources": list(answer.get("sources") or []),
            "refused": bool(answer.get("refused", False)),
        }
        if include_facts:
            result["facts"] = list(answer.get("facts") or [])
        return result

    result = {
        "answer": answer.text,
        "sources": list(answer.sources),
        "refused": bool(answer.refused),
    }
    if include_facts:
        result["facts"] = list(answer.facts)
    return result


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Найти конкретные факты о модели, детали или коде ошибки в графе знаний.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос для поиска."}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Получить обзорный ответ по сводкам сообществ графа знаний.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Обзорный вопрос."}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на одном из складов ERP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить актуальную цену детали из ERP.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"}},
                "required": ["part_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ детали в ERP. Вызывать только после явного подтверждения пользователя.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string", "description": "Код модели оборудования, например AR-210."},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tools(
    kb: KnowledgeBase, llm: LLMClient, erp: ErpClient
) -> dict[str, Callable[..., dict]]:
    """Создаёт словарь «имя инструмента -> функция».

    Ошибки аргументов, ErpNotFound и ErpRejected возвращаются как {"error": ...}.
    ErpUnavailable намеренно не перехватывается: повторы выполняет agent.py.
    """

    def search_knowledge(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Параметр question должен быть непустой строкой."}
        answer = local_search(question, kb, llm)
        return _answer_to_dict(answer, include_facts=True)

    def summarize_topic(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Параметр question должен быть непустой строкой."}
        answer = global_search(question, kb, llm)
        return _answer_to_dict(answer, include_facts=False)

    def check_stock(part_id: str = "", warehouse: str = "") -> dict:
        if not _is_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."}
        try:
            return erp.get_stock(part_id.strip().upper(), warehouse)
        except ErpNotFound as exc:
            return {"error": f"Деталь {part_id} не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отказала в запросе остатков: {exc}"}

    def get_price(part_id: str = "") -> dict:
        if not _is_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}
        try:
            return erp.get_price(part_id.strip().upper())
        except ErpNotFound as exc:
            return {"error": f"Деталь {part_id} не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отказала в получении цены: {exc}"}

    def create_service_order(
        part_id: str = "", warehouse: str = "", qty: int = 0, model: str = ""
    ) -> dict:
        if not _is_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "Количество qty должно быть целым числом больше нуля."}
        if not isinstance(model, str) or not model.strip():
            return {"error": "Не указана модель оборудования."}
        try:
            return erp.create_order(part_id.strip().upper(), warehouse, qty, model.strip().upper())
        except ErpNotFound as exc:
            return {"error": f"Деталь {part_id} не найдена в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила заказ: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
