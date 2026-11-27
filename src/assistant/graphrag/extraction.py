"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
import re
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant.graphrag.extraction_prompts import relation_prompt
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, load_vocabulary
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, RELATION_SIGNATURES, Chunk, Entity, ExtractionResult, Relation

vocabulary_path = Path(__file__).resolve().parents[3] / "data" / "vocabulary.json"


_ERROR_CODE_ROW = re.compile(r"\s*(?:[-+*]|\|)\s*[eе][- ]?\d{2}(?!\w)", re.IGNORECASE)


def _incident_text(text: str) -> str:
    """Убирает строки справочника кодов из входа для извлечения FAILED."""
    in_error_reference = False
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            in_error_reference = line.removeprefix("## ").strip().casefold() == "коды ошибок"
        if not in_error_reference or not _ERROR_CODE_ROW.match(line):
            lines.append(line)
    return "".join(lines)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (json.JSONDecodeError, ValueError, RuntimeError, ConnectionError, TimeoutError)
    ),
    retry_error_callback=lambda e: None,
    before_sleep=lambda retry_state: print(
        f"LLM call failed (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
    ),
)
def _ask(chunk_text: str, llm: LLMClient, kind: str) -> dict | None:
    resp = llm.chat(
        [{"role": "user", "content": relation_prompt(kind) + chunk_text}],
        json_schema=EXTRACTION_SCHEMA,
    )
    raw = getattr(resp, "content", resp)
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if (
            isinstance(data, dict)
            and isinstance(data.get("entities"), list)
            and isinstance(data.get("relations"), list)
    ):
        return data

    raise ValueError(f"Invalid schema returned: {data}")


def _resolve(entity_type: str, entity_raw_name: str, vocabulary: dict[str, list[str]], text_codes) -> str | None:
    # нет заготовленного списка правильных кодов,
    # метод проверяет именно валидность вытащенного ЛЛМкой кода, но не то встречался ли он в тексте
    if not isinstance(entity_type, str) or not isinstance(entity_raw_name, str):
        return None
    entity_name = canonical_name(entity_type, entity_raw_name, vocabulary)
    if entity_name is None:
        return None
    if entity_type in CODE_TYPES and entity_name not in text_codes:
        return None
    return entity_name


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением целевой связи и содержит пример. Запросы выполняются последовательно по типам.
    Неподтверждённые записи отбрасываются и учитываются в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Для каждого запроса предусмотрена одна повторная попытка. Остальные ответы сохраняются.
    """

    entities = []
    responses = []
    for kind in RELATION_SIGNATURES:
        text = _incident_text(chunk.text) if kind == "FAILED" else chunk.text
        data = _ask(text, llm, kind)
        if data is not None:
            entities.extend(data["entities"])
            responses.extend((kind, relation) for relation in data["relations"])

    text_codes = set(find_codes(chunk.text))
    kept: dict[str, Entity] = {}
    dropped = 0

    vocabulary = load_vocabulary(vocabulary_path)

    # entities
    for entity in entities:
        if not isinstance(entity, dict):
            dropped += 1
            continue
        entity_type, entity_name = entity.get("type"), entity.get("name")
        entity_name = _resolve(entity_type, entity_name, vocabulary, text_codes)
        if entity_name is None:
            dropped += 1
            continue

        kept.setdefault(f"{entity_type}:{entity_name}", Entity(entity_type, entity_name))

    # relations
    relations: list[Relation] = []
    seen_relations: set[Relation] = set()
    for kind, relation in responses:
        if not isinstance(relation, dict):
            dropped += 1
            continue
        subject_entity_type, subject_entity_name, object_entity_type, object_entity_name, relation_type = (
            relation.get("subject_type"), relation.get("subject"),
            relation.get("object_type"), relation.get("object"), relation.get("type"),
        )
        if relation_type != kind or (subject_entity_type, object_entity_type) != RELATION_SIGNATURES[kind]:
            dropped += 1
            continue
        subject_entity_name = _resolve(subject_entity_type, subject_entity_name, vocabulary, text_codes)
        object_entity_name = _resolve(object_entity_type, object_entity_name, vocabulary, text_codes)
        if subject_entity_name is None or object_entity_name is None:
            dropped += 1
            continue
        subject_node, object_node = (Entity(subject_entity_type, subject_entity_name),
                                     Entity(object_entity_type, object_entity_name))
        kept.setdefault(subject_node.node_id, subject_node)
        kept.setdefault(object_node.node_id, object_node)
        edge = Relation(subject_node, relation_type, object_node)
        if subject_node == object_node:
            dropped += 1
            continue
        if edge not in seen_relations:
            seen_relations.add(edge)
            relations.append(edge)

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(kept.values()), relations, dropped=dropped)
