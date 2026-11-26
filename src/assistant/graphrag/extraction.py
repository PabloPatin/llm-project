"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, load_vocabulary
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

_PROMPT = (
    "Извлеки сущности и связи строго по JSON-схеме. Направление: subject -> object.\n"
    'Пример: {"entities":[{"type":"Model","name":"AR-210"},{"type":"Component","name":"пусковое реле"},'
    '{"type":"Part","name":"P-1042"}],"relations":[{"subject_type":"Model","subject":"AR-210",'
    '"type":"HAS_PART","object_type":"Component","object":"пусковое реле"}]}\n\nТекст:\n'
)

vocabulary_path = Path(__file__).resolve().parents[3] / "data" / "vocabulary.json"


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (json.JSONDecodeError, ValueError, RuntimeError, ConnectionError, TimeoutError)
    ),
    retry_error_callback=lambda e: None,
)
def _ask(chunk_text: str, llm: LLMClient) -> dict:
    resp = llm.chat(
        [{"role": "user", "content": _PROMPT + chunk_text}],
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
    entity_name = canonical_name(entity_type, entity_raw_name, vocabulary)
    if entity_name is None:
        return None
    if entity_type in CODE_TYPES and entity_name not in text_codes:
        return None
    return entity_name


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    data = _ask(chunk.text, llm)
    if data is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, [], [], dropped=0)

    text_codes = set(find_codes(chunk.text))
    kept: dict[str, Entity] = {}
    dropped = 0

    vocabulary = load_vocabulary(vocabulary_path)

    # entities
    for entity in data["entities"]:
        entity_type, entity_name = entity.get("type"), entity.get("name")
        entity_name = _resolve(entity_type, entity_name, vocabulary, text_codes)
        if entity_name is None:
            dropped += 1
            continue

        kept.setdefault(f"{entity_type}:{entity_name}", Entity(entity_type, entity_name))

    # relations
    relations: list[Relation] = []
    for relation in data["relations"]:
        if not isinstance(relation, dict):
            dropped += 1
            continue
        subject_entity_type, subject_entity_name, object_entity_type, object_entity_name, relation_type = (
            relation.get("subject_type"), relation.get("subject"),
            relation.get("object_type"), relation.get("object"), relation.get("type"),
        )
        subject_entity_name = _resolve(subject_entity_type, subject_entity_name, vocabulary, text_codes)
        object_entity_name = _resolve(object_entity_type, object_entity_name, vocabulary, text_codes)
        if subject_entity_name is None or object_entity_name is None:
            dropped += 1
            continue
        subject_node, object_node = (Entity(subject_entity_type, subject_entity_name),
                                     Entity(object_entity_type, object_entity_name))
        kept.setdefault(subject_node.node_id, subject_node)
        kept.setdefault(object_node.node_id, object_node)
        relations.append(Relation(subject_node, relation_type, object_node))

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(kept.values()), relations, dropped=dropped)
