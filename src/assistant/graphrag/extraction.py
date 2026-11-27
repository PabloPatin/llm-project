"""Последовательное извлечение связей и сущностей одного чанка."""

import json
from pathlib import Path

from assistant.graphrag.extraction_prompts import FAILED_CONTEXTS, PAIR_FIELDS, relation_prompt
from assistant.infra.grounding import (
    CODE_TYPES,
    canonical_name,
    code_type,
    find_codes,
    find_names,
    load_vocabulary,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import RELATION_SIGNATURES, Chunk, Entity, ExtractionResult, Relation

vocabulary_path = Path(__file__).resolve().parents[3] / "data" / "vocabulary.json"


def _pair_schema(kind: str) -> dict:
    fields = PAIR_FIELDS[kind]
    properties = {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {name: {"type": "string"} for name in fields},
                "required": list(fields),
                "additionalProperties": False,
            },
        },
    }
    required = ["pairs"]
    if kind == "FAILED":
        properties["context"] = {"type": "string", "enum": list(FAILED_CONTEXTS)}
        required.insert(0, "context")
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _ask_pairs(kind: str, text: str, llm: LLMClient) -> list | None:
    prompt = relation_prompt(kind) + text
    schema = _pair_schema(kind)
    retry_prompt = "Предыдущий ответ не удалось разобрать. Верни только JSON по указанной схеме.\n\n"
    # У второй попытки отдельный ключ кеша: некорректный JSON мог сохраниться после первого вызова.
    for content in (prompt, retry_prompt + prompt):
        try:
            data = json.loads(llm.chat([{"role": "user", "content": content}], json_schema=schema).content)
            if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
                continue
            if kind == "FAILED":
                context = data.get("context")
                if context not in FAILED_CONTEXTS:
                    continue
                if context != "incident":
                    return []
            return data["pairs"]
        except (ValueError, RuntimeError, ConnectionError, TimeoutError):
            continue
    return None


def _resolve(entity_type: str, raw: object, vocabulary: dict[str, list[str]], codes: set[str]) -> Entity | None:
    if not isinstance(raw, str):
        return None
    name = canonical_name(entity_type, raw, vocabulary)
    if name is None or (entity_type in CODE_TYPES and name not in codes):
        return None
    return Entity(entity_type, name)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Собирает пары из отдельных запросов, нормализует сущности и сохраняет источник чанка."""
    vocabulary = load_vocabulary(vocabulary_path)
    codes = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    dropped = 0
    answered = False

    for kind, (subject_type, object_type) in RELATION_SIGNATURES.items():
        pairs = _ask_pairs(kind, chunk.text, llm)
        if pairs is None:
            continue
        answered = True
        subject_field, object_field = PAIR_FIELDS[kind]
        for pair in pairs:
            if not isinstance(pair, dict):
                dropped += 1
                continue
            subject = _resolve(subject_type, pair.get(subject_field), vocabulary, codes)
            object_ = _resolve(object_type, pair.get(object_field), vocabulary, codes)
            if subject is None or object_ is None or subject == object_:
                dropped += 1
                continue
            entities[subject.node_id] = subject
            entities[object_.node_id] = object_
            relations[(subject.node_id, kind, object_.node_id)] = Relation(subject, kind, object_)

    if answered:
        # Явные упоминания образуют узлы даже тогда, когда для них не найдено связей.
        for code in sorted(codes):
            entity = Entity(code_type(code), code)
            entities.setdefault(entity.node_id, entity)
        for kind, name in find_names(chunk.text, vocabulary):
            entity = Entity(kind, name)
            entities.setdefault(entity.node_id, entity)

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), list(relations.values()), dropped)
