"""Цикл AI-агента с проверками, повторами и подтверждением заказа (FR-7–FR-10)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_PART_RE = re.compile(r"^P-\d{4}[A-Z]?$", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"(?<![\w-])(\d[\d\s\u00a0]*(?:[.,]\d{1,2})?)\s*(?:₽|руб(?:\.?|лей|ля|лями)?|р\.)",
    re.IGNORECASE,
)
_PRICE_LABEL_RE = re.compile(
    r"(?:цена|стоимость)\s*[:—-]?\s*(\d[\d\s\u00a0]*(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)
_STOCK_RE = re.compile(
    r"(?<![\w-])(\d+)\s*(?:шт\.?|штук[аи]?|единиц[аы]?)|"
    r"(?:остаток|в наличии|доступно)\s*[:—-]?\s*(\d+)",
    re.IGNORECASE,
)
_RUSSIAN_NUMBERS = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}
_QTY_WORD_RE = re.compile(
    r"\b(ноль|один|одна|одно|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+"
    r"(?:шт\.?|штук[аи]?|единиц[аы]?)",
    re.IGNORECASE,
)


def _json_key(tool_name: str, arguments: dict) -> str:
    return json.dumps([tool_name, arguments], ensure_ascii=False, sort_keys=True, default=str)


def _all_text(value: Any) -> str:
    """Преобразует вложенные результаты инструментов в текст для поиска фактов и источников."""
    if isinstance(value, dict):
        return " ".join(f"{key} {_all_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_all_text(item) for item in value)
    return str(value)


def _extract_source_names(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        sources = value.get("sources")
        if isinstance(sources, (list, tuple)):
            found.extend(str(source) for source in sources if source)
        for key, item in value.items():
            if key != "sources":
                found.extend(_extract_source_names(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_extract_source_names(item))
    return found


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _mentioned_quantities(text: str) -> set[int]:
    """Числа из пользовательского вопроса, не захватывая цифры внутри кодов AR-210/P-1042."""
    values = {
        int(match.group(0))
        for match in re.finditer(r"(?<![\w-])\d+(?![\w-])", text)
    }
    for match in _QTY_WORD_RE.finditer(text.lower()):
        values.add(_RUSSIAN_NUMBERS[match.group(1).lower()])
    return values


def _is_quantity_in_question(question: str, qty: Any) -> bool:
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        return False
    return qty in _mentioned_quantities(question)


def _known_codes(question: str, observations: list[dict]) -> set[str]:
    text = question + " " + " ".join(_all_text(item) for item in observations)
    return set(find_codes(text))


def _money_values(text: str) -> set[int]:
    values: set[int] = set()
    matches = list(_CURRENCY_RE.finditer(text)) + list(_PRICE_LABEL_RE.finditer(text))
    for match in matches:
        digits = re.sub(r"\D", "", match.group(1))
        if digits:
            values.add(int(digits))
    return values


def _stock_values(text: str) -> set[int]:
    values: set[int] = set()
    for match in _STOCK_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        if raw is not None:
            values.add(int(raw))
    return values


def _structured_values(observations: list[dict], key: str) -> set[int]:
    values: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            raw = value.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool):
                values.add(raw)
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for observation in observations:
        visit(observation)
    return values


def _answer_is_grounded(answer: str, question: str, observations: list[dict]) -> bool:
    """Проверяет коды, денежные суммы и остатки по вопросу и ответам инструментов."""
    known_codes = _known_codes(question, observations)
    if any(code not in known_codes for code in find_codes(answer)):
        return False

    known_prices = _structured_values(observations, "price_rub") | _money_values(question)
    if any(price not in known_prices for price in _money_values(answer)):
        return False

    known_stock = _structured_values(observations, "qty") | _mentioned_quantities(question)
    if any(amount not in known_stock for amount in _stock_values(answer)):
        return False

    return True


def _final_refusal() -> str:
    return "В базе знаний не найдено подтверждённых данных для достоверного ответа."


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Выполняет bounded tool-calling loop и возвращает AgentResult.

    `max_steps` ограничивает число попыток вызова инструментов. Повтор ERP при
    ErpUnavailable выполняется внутри одной попытки инструмента (максимум 3 попытки всего).
    """
    if not isinstance(question, str) or not question.strip():
        return AgentResult(answer="Пожалуйста, задайте вопрос.", refused=True)
    if max_steps < 0:
        raise ValueError("max_steps не может быть отрицательным")

    system_prompt = (
        "Ты ассистент сервисной службы. Используй инструменты для получения фактов. "
        "Не выдумывай номера деталей, цены, остатки и источники. Сначала найди деталь в базе знаний, "
        "если пользователь не указал её номер. Цены и остатки получай только из ERP. "
        "Создание заказа запрашивай только если пользователь явно просит оформить заказ; "
        "самостоятельно не придумывай количество. Если данных нет, честно сообщи об этом."
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]
    steps: list[ToolStep] = []
    observations: list[dict] = []
    sources: list[str] = []
    cache: dict[str, dict] = {}
    successful_knowledge = False
    refused_knowledge = False
    tool_calls_made = 0

    def finish(answer: str, *, refused: bool = False) -> AgentResult:
        # Если заказ всё-таки создан, обязательно показываем возвращённый ERP номер.
        order_ids = [
            str(step.result["order_id"])
            for step in steps
            if step.tool == "create_service_order"
            and isinstance(step.result, dict)
            and step.result.get("order_id")
        ]
        for order_id in order_ids:
            if order_id not in answer:
                answer = f"{answer.rstrip()} Заказ {order_id} создан.".strip()
        return AgentResult(
            answer=answer,
            steps=steps,
            sources=_unique(sources),
            refused=refused,
        )

    def ask_model(*, allow_tools: bool = True):
        return llm.chat(messages, tools=TOOL_SCHEMAS if allow_tools else None)

    # Первый запрос и последующие итерации цикла.
    while True:
        response = ask_model(allow_tools=tool_calls_made < max_steps)

        if not response.tool_calls:
            answer = (response.content or "").strip()
            if not answer:
                answer = "Не удалось сформировать ответ по имеющимся данным."
            if not _answer_is_grounded(answer, question, observations):
                messages.append(response.as_message())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Проверь предыдущий ответ. Исправь его так, чтобы каждый номер кода, "
                            "каждая цена и каждый остаток подтверждались исходным вопросом или "
                            "результатами инструментов. Удали неподтверждённые сведения. "
                            "Не вызывай инструменты; верни только исправленный ответ."
                        ),
                    }
                )
                corrected = llm.chat(messages, tools=None)
                corrected_answer = (corrected.content or "").strip()
                if corrected.tool_calls or not corrected_answer or not _answer_is_grounded(
                    corrected_answer, question, observations
                ):
                    return finish(_final_refusal(), refused=True)
                answer = corrected_answer

            refused = refused_knowledge and not successful_knowledge
            if refused and "не найдено" not in answer.lower():
                answer = "В базе знаний не найдено подтверждённых сведений. " + answer
            return finish(answer, refused=refused)

        # Добавляем сообщение ассистента с запросами инструментов в историю диалога.
        messages.append(response.as_message())

        for call in response.tool_calls:
            if tool_calls_made >= max_steps:
                # Дальше не исполняем инструментальные вызовы; запрашиваем у модели итог ниже.
                messages.append(tool_message(call, {"error": "Достигнут лимит вызовов инструментов."}))
                continue

            tool_calls_made += 1
            tool_name = call.name
            raw_args = call.arguments if isinstance(call.arguments, dict) else {"_raw": str(call.arguments)}
            args = dict(raw_args)

            if "_raw" in args or not all(isinstance(key, str) for key in args):
                error = "Не удалось разобрать аргументы инструмента. Передай корректные аргументы в JSON."
                steps.append(ToolStep(tool=tool_name, args=args, error=error))
                messages.append(tool_message(call, {"error": error}))
                continue

            if tool_name not in tools:
                error = f"Неизвестный инструмент: {tool_name}. Используй только доступные инструменты."
                steps.append(ToolStep(tool=tool_name, args=args, error=error))
                messages.append(tool_message(call, {"error": error}))
                continue

            # Все номера деталей должны быть указаны пользователем или получены из прошлых данных.
            part_id = args.get("part_id")
            if part_id is not None:
                if not isinstance(part_id, str) or not _PART_RE.fullmatch(part_id.strip()):
                    error = "Некорректный номер детали. Сначала найди корректный номер через search_knowledge."
                    steps.append(ToolStep(tool=tool_name, args=args, error=error))
                    messages.append(tool_message(call, {"error": error}))
                    continue
                if part_id.upper() not in _known_codes(question, observations):
                    error = (
                        f"Номер детали {part_id} не встречается в вопросе или предыдущих результатах. "
                        "Сначала вызови search_knowledge и используй только найденный номер."
                    )
                    steps.append(ToolStep(tool=tool_name, args=args, error=error))
                    messages.append(tool_message(call, {"error": error}))
                    continue

            # Количество заказа нельзя выводить из контекста или придумывать.
            if tool_name == "create_service_order":
                qty = args.get("qty")
                if not _is_quantity_in_question(question, qty):
                    error = "Пользователь не указал количество. Спроси, сколько штук нужно заказать."
                    steps.append(ToolStep(tool=tool_name, args=args, error=error))
                    messages.append(tool_message(call, {"error": error}))
                    continue

                if not confirm:
                    pending = {
                        "tool": "create_service_order",
                        "args": {
                            "part_id": args.get("part_id"),
                            "warehouse": args.get("warehouse"),
                            "qty": qty,
                            "model": args.get("model"),
                        },
                    }
                    steps.append(
                        ToolStep(
                            tool=tool_name,
                            args=args,
                            result={"needs_confirmation": True},
                        )
                    )
                    return AgentResult(
                        answer="Подтвердите создание заказа с указанными параметрами.",
                        steps=steps,
                        sources=_unique(sources),
                        refused=False,
                        needs_confirmation=True,
                        pending_action=pending,
                    )

            key = _json_key(tool_name, args)
            if key in cache:
                result = cache[key]
            else:
                result = None
                last_error: Exception | None = None
                attempts = 3 if tool_name in {"check_stock", "get_price", "create_service_order"} else 1
                for _ in range(attempts):
                    try:
                        result = tools[tool_name](**args)
                        last_error = None
                        break
                    except ErpUnavailable as exc:
                        last_error = exc
                    except Exception as exc:  # инструментальная ошибка возвращается модели как наблюдение
                        last_error = exc
                        break

                if last_error is not None:
                    result = {"error": str(last_error)}
                if not isinstance(result, dict):
                    result = {"error": "Инструмент вернул результат неверного формата."}
                cache[key] = result

            # Ошибки ERP и отказы поиска не являются подтверждающими фактами.
            if "error" not in result and not (
                tool_name in {"search_knowledge", "summarize_topic"} and result.get("refused")
            ):
                observations.append(result)
            sources.extend(_extract_source_names(result))
            if "error" in result:
                error = str(result["error"])
                steps.append(ToolStep(tool=tool_name, args=args, result=result, error=error))
            else:
                steps.append(ToolStep(tool=tool_name, args=args, result=result))
                if tool_name in {"search_knowledge", "summarize_topic"}:
                    if result.get("refused"):
                        refused_knowledge = True
                    else:
                        successful_knowledge = True

            messages.append(tool_message(call, result))

        # Если исчерпан лимит, делаем один завершающий запрос без инструментов.
        if tool_calls_made >= max_steps:
            response_after_limit = llm.chat(
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Сформируй краткий ответ только по уже полученным данным. "
                            "Новые инструменты не вызывай."
                        ),
                    }
                ],
                tools=None,
            )
            answer = (response_after_limit.content or "").strip()
            if not answer:
                answer = "Работа прервана: достигнут лимит вызовов инструментов."
            elif "прерван" not in answer.lower() and "лимит" not in answer.lower():
                answer = answer.rstrip() + " Работа прервана: достигнут лимит вызовов инструментов."
            if not _answer_is_grounded(answer, question, observations):
                return finish(_final_refusal(), refused=True)
            return finish(answer, refused=refused_knowledge and not successful_knowledge)
