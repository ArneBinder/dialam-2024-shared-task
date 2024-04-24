import json
import logging
from typing import List, Tuple

import load_map
import networkx as nx
from networkx.classes.digraph import DiGraph

logger = logging.getLogger(__name__)


def get_graph(node_path: str) -> DiGraph:
    """Load the graph stored in JSON format and parse it as DiGraph."""

    try:
        with open(node_path) as json_data:
            graph = load_map.parse_json(json.load(json_data))
    except (IOError):
        logger.error(f"File was not found: {node_path}")
    return graph


def remove_redundant_nodes(graph: DiGraph) -> DiGraph:
    """Remove TA, L, YA nodes from the graph."""

    nodes_to_remove = [
        x
        for x, y in graph.nodes(data=True)
        if y["type"] == "TA" or y["type"] == "L" or y["type"] == "YA"
    ]

    graph.remove_nodes_from(nodes_to_remove)

    return graph


def remove_iso_analyst_nodes(graph: DiGraph) -> DiGraph:
    """Remove isolated L-nodes from the graph."""
    analyst_nodes = []
    isolated_nodes = list(nx.isolates(graph))
    for node in isolated_nodes:
        if graph.nodes[node]["type"] == "L":
            analyst_nodes.append(node)
    graph.remove_nodes_from(analyst_nodes)
    return graph


def get_type_node_list(graph: DiGraph, node_types: List[str]) -> List[Tuple[int, str]]:
    """Filter out and return nodes of a given type."""
    nodes = [
        (x, y["text"])
        for x, y in graph.nodes(data=True)
        if "type" in y and y["type"] in node_types
    ]
    return nodes


def get_s_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
    """Filter out and return S-type nodes (MA, RA, CA)."""
    return get_type_node_list(graph, ["MA", "RA", "CA", "PA"])


def get_l_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
    """Filter out and return L-type nodes (locutions)."""
    return get_type_node_list(graph, ["L"])


def get_i_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
    """Filter out and return I-type nodes (propositions)."""
    return get_type_node_list(graph, ["I"])


def get_rels(rel_type: str, graph: DiGraph) -> List[int]:
    """Collect all nodes in the graph that correspond to the given relation type."""
    rel_nodes = [x for x, y in graph.nodes(data=True) if "type" in y and y["type"] == rel_type]
    return rel_nodes
