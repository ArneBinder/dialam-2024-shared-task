import json
import logging

import networkx as nx
from load_map import CorpusLoader
from networkx.classes.digraph import DiGraph

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Set, Tuple


class Centrality:
    @staticmethod
    def get_graph(node_path: str) -> DiGraph:
        """Load the graph stored in JSON format and parse it as DiGraph."""
        corpus_loader = CorpusLoader()
        try:
            with open(node_path) as json_data:
                graph = corpus_loader.parse_json(json.load(json_data))
        except (IOError):
            logger.error(f"File was not found: {node_path}")
        return graph

    @staticmethod
    def remove_redundant_nodes(graph: DiGraph) -> DiGraph:
        """Remove TA, L, YA nodes from the graph."""
        node_types = nx.get_node_attributes(graph, "type")

        nodes_to_remove = [
            x
            for x, y in graph.nodes(data=True)
            if y["type"] == "TA" or y["type"] == "L" or y["type"] == "YA"
        ]

        graph.remove_nodes_from(nodes_to_remove)

        return graph

    @staticmethod
    def remove_iso_analyst_nodes(graph: DiGraph) -> DiGraph:
        """Remove isolated L-nodes from the graph."""
        analyst_nodes = []
        isolated_nodes = list(nx.isolates(graph))
        for node in isolated_nodes:
            if graph.nodes[node]["type"] == "L":
                analyst_nodes.append(node)
        graph.remove_nodes_from(analyst_nodes)
        return graph

    @staticmethod
    def get_s_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
        """Filter out and return S-type nodes (MA, RA, CA)."""
        s_nodes = [
            x
            for x, y in graph.nodes(data=True)
            if y["type"] == "MA" or y["type"] == "RA" or y["type"] == "CA" or y["type"] == "PA"
        ]
        return s_nodes

    @staticmethod
    def get_l_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
        """Filter out and return L-type nodes (locutions)."""
        l_nodes = [(x, y["text"]) for x, y in graph.nodes(data=True) if y["type"] == "L"]
        return l_nodes

    @staticmethod
    def get_i_node_list(graph: DiGraph) -> List[Tuple[int, str]]:
        """Filter out and return I-type nodes (propositions)."""
        i_nodes = [(x, y["text"]) for x, y in graph.nodes(data=True) if y["type"] == "I"]
        return i_nodes

    @staticmethod
    def get_rels(rel_type: str, graph: DiGraph) -> List[int]:
        """Collect all nodes in the graph that correspond to the given relation type."""
        rel_nodes = [x for x, y in graph.nodes(data=True) if y["type"] == rel_type]
        return rel_nodes
