import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_nodeset_ids_from_directory(nodeset_dir: str) -> List[str]:
    """Get the IDs of all nodesets in a directory."""

    return [
        f.split("nodeset")[1].split(".json")[0]
        for f in os.listdir(nodeset_dir)
        if f.endswith(".json")
    ]


def read_nodeset(nodeset_dir: str, nodeset_id: str) -> Dict[str, Any]:
    """Read a nodeset with a given ID from a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename) as f:
        return json.load(f)


def write_nodeset(nodeset_dir: str, nodeset_id: str, data: Dict[str, Any]) -> None:
    """Write a nodeset with a given ID to a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def process_all_nodesets(
    nodeset_dir: str, func: Callable[..., T], show_progress: bool = True, **kwargs
) -> Iterator[Tuple[str, Union[T, Exception]]]:
    """Process all nodesets in a directory.

    Args:
        nodeset_dir: The directory containing the nodesets.
        func: The function to apply to each nodeset.
        show_progress: Whether to show a progress bar.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        A tuple containing the nodeset ID and the result of applying the function.
        If an exception occurs, the result will be the exception.
    """

    nodeset_ids = get_nodeset_ids_from_directory(nodeset_dir=nodeset_dir)
    for nodeset_id in tqdm.tqdm(
        nodeset_ids, desc="Processing nodesets", disable=not show_progress
    ):
        try:
            result = func(
                nodeset_dir=nodeset_dir,
                nodeset_id=nodeset_id,
                **kwargs,
            )
            yield nodeset_id, result
        except Exception as e:
            yield nodeset_id, e


def get_node_ids(node_id2node: Dict[str, Any], allowed_node_types: List[str]) -> List[str]:
    """Get the IDs of nodes with a given type."""

    return [
        node_id for node_id, node in node_id2node.items() if node["type"] in allowed_node_types
    ]


def create_edges_from_relations(
    relations: List[Tuple[str, str, str]],
    edges: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Create edge objects from relations.

    Args:
        relations: A list of binary relations: tuples containing the source node ID, target node ID, and relation node ID.
        edges: A list of edge objects where each object contains the keys "fromID" and "toID".

    Returns:
        A list of edge objects where each object contains the keys "fromID" and "toID".
    """
    biggest_edge_id = max([int(edge["fromID"]) for edge in edges])
    new_edges = []
    for src_id, trg_id, rel_id in relations:
        biggest_edge_id += 1
        new_edges.append({"fromID": src_id, "toID": rel_id, "edgeID": str(biggest_edge_id)})
        biggest_edge_id += 1
        new_edges.append({"fromID": rel_id, "toID": trg_id, "edgeID": str(biggest_edge_id)})
    return new_edges


def create_relation_nodes_from_alignment(
    node_id2node: Dict[str, Any],
    node_alignments: List[Tuple[str, str]],
    node_type: str,
    node_text: str,
    swap_direction: bool = False,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:

    """Create relation nodes from alignments between two nodes.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        node_alignments: A list of tuples containing the alignment between two nodes.
        node_type: The type of the nodes.
        node_text: The text of the nodes.
        swap_direction: A boolean indicating whether to swap the direction of the alignment
            before creating the relation node.

    Returns:
        A tuple containing:
         - a list of binary YA relations: tuples containing the source node ID, target node ID, and YA node ID, and
         - a dictionary containing the newly created YA nodes as a mapping from IDs to node content.
    """
    biggest_node_id = max([int(node_id) for node_id in node_id2node.keys()])
    new_node_id2node = dict()
    relations = []
    for src_id, trg_id in node_alignments:
        if swap_direction:
            src_id, trg_id = trg_id, src_id
        biggest_node_id += 1
        node_id = str(biggest_node_id)
        new_node_id2node[node_id] = {
            "id": node_id,
            "type": node_type,
            "text": node_text,
        }
        relations.append((src_id, trg_id, node_id))

    return relations, new_node_id2node


def get_binary_relations(
    node_id2node: Dict[str, Any],
    edges: List[Dict[str, str]],
    allowed_node_types: Optional[List[str]] = None,
    allowed_source_types: Optional[List[str]] = None,
    allowed_target_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Create binary relations from nodes, i.e. tuples containing the source node ID, target node
    ID, and relation node ID.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        edges: A list of edge objects where each object contains the keys "fromID" and "toID".
        allowed_node_types: A list of node types to consider.
        allowed_source_types: A list of source node types to consider.
        allowed_target_types: A list of target node types to consider.

    Returns:
        A list of binary relations: tuples containing the source node ID, target node ID, and relation node ID.
    """

    # helper dictionaries to map source and target nodes to their corresponding target and source nodes
    src2targets: Dict[str, List[str]] = defaultdict(list)
    trg2sources: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        src_id = edge["fromID"]
        trg_id = edge["toID"]
        src2targets[src_id].append(trg_id)
        trg2sources[trg_id].append(src_id)

    relations = []
    for node_id, node in node_id2node.items():
        # filter nodes based on allowed types
        if allowed_node_types is not None and node["type"] not in allowed_node_types:
            continue
        # iterate over all source nodes ...
        for src_id in trg2sources[node_id]:
            src_node = node_id2node[src_id]
            if allowed_source_types is None or src_node["type"] in allowed_source_types:
                # ... and all target nodes
                for trg_id in src2targets[src_id]:
                    trg_node = node_id2node[trg_id]
                    if allowed_target_types is None or trg_node["type"] in allowed_target_types:
                        relations.append((src_id, trg_id, node_id))
    return relations
