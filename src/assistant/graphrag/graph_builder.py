"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0

    for res in results:
        for entity in res.entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=set())
            graph.nodes[entity.node_id]["sources"].add(res.chunk_id)

    edges = {}
    for res in results:
        for relation in res.relations:
            if relation.matches_schema() and relation.subject.node_id in graph and relation.object.node_id in graph:
                edges.setdefault((relation.subject.node_id, relation.type, relation.object.node_id), set()).add(
                    res.chunk_id)

    for (subject_node_id, relation, object_node_id) in list(edges):
        key = (subject_node_id, relation, object_node_id)
        opposite = (object_node_id, relation, subject_node_id)
        if key not in edges or relation != "REPLACED_BY" or opposite not in edges:
            continue
        graph.graph["conflicts"] += 1
        weight, weight_opposite = len(edges[(subject_node_id, relation, object_node_id)]), len(edges[opposite])
        if weight >= weight_opposite:
            edges.pop(opposite, None)
        if weight <= weight_opposite:
            edges.pop((subject_node_id, relation, object_node_id), None)

    for (subject_node_id, relation, object_node_id), chunks in edges.items():
        graph.add_edge(subject_node_id, object_node_id, type=relation, weight=len(chunks), sources=sorted(chunks))

    return graph
