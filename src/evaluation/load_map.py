import logging
from datetime import datetime
from typing import Any, Dict, List, Union

import networkx as nx
from networkx.classes.digraph import DiGraph

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp: str, ignore_timestamp_casting: bool) -> Union[datetime, str]:
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        if not ignore_timestamp_casting:
            logger.error(f"Failed datetime(timestamp) casting: {timestamp}")
    return timestamp


def parse_scheme_id(scheme_id: str) -> Union[int, str]:
    try:
        return int(scheme_id)
    except (ValueError, TypeError):
        logger.error(f"Failed int(schemeID) casting: {scheme_id}")
    return scheme_id


def parse_node_id(node_id: str) -> Union[int, str]:
    try:
        return int(node_id)
    except (ValueError, TypeError):
        logger.error(f"Failed int(nodeID) casting: {node_id}")
    return node_id


def parse_edge_id(edge_id: str) -> Union[int, str]:
    try:
        return int(edge_id)
    except (ValueError, TypeError):
        logger.error(f"Failed int(edgeID) casting: {edge_id}")
    return edge_id


def parse_json(
    node_set: Dict[str, List[Dict[str, Any]]], ignore_timestamp_casting: bool
) -> DiGraph:
    """Parse JSON file with annotations for nodes, edges and locutions and create a DiGraph."""
    G = nx.DiGraph()
    locution_dict = {}
    # Process nodes.
    for node in node_set["nodes"]:
        if "scheme" in node:
            G.add_node(
                parse_node_id(node["nodeID"]),
                text=node.get("text", None),
                type=node.get("type", None),
                timestamp=parse_timestamp(node.get("timestamp", None), ignore_timestamp_casting),
                scheme=node.get("scheme", None),
                scheme_id=parse_scheme_id(node.get("schemeID", None)),
            )
        else:
            G.add_node(
                parse_node_id(node["nodeID"]),
                text=node.get("text", None),
                type=node.get("type", None),
                timestamp=parse_timestamp(node.get("timestamp", None), ignore_timestamp_casting),
            )
    # Process edges.
    for edge in node_set["edges"]:
        from_id = parse_edge_id(edge["fromID"])
        to_id = parse_edge_id(edge["toID"])
        G.add_edge(from_id, to_id)
    # Process locutions. Currently, not being used anywhere.
    for locution in node_set["locutions"]:
        node_id = parse_node_id(locution["nodeID"])
        locution_dict[node_id] = locution
    return G
