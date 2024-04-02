import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import tqdm

FuncResult = TypeVar("FuncResult")
# add "scheme" and "schemeID"? both seem to be optional. add "timestamp"?
Node = TypedDict("Node", {"nodeID": str, "type": str, "text": str})
# add "formEdgeID"? it is mostly (always?) None
Edge = TypedDict("Edge", {"fromID": str, "toID": str, "edgeID": str})
# add "start", "end", and "source"? "end" and "source" are mostly (always?) None
Locution = TypedDict("Locution", {"nodeID": str, "personID": str, "timestamp": str})
Nodeset = TypedDict(
    "Nodeset",
    {"nodes": List[Node], "edges": List[Edge], "locutions": List[Locution]},
)
Relation = TypedDict("Relation", {"sources": List[str], "targets": List[str], "relation": str})


logger = logging.getLogger(__name__)


def get_nodeset_ids_from_directory(nodeset_dir: str) -> List[str]:
    """Get the IDs of all nodesets in a directory."""

    return [
        f.split("nodeset")[1].split(".json")[0]
        for f in os.listdir(nodeset_dir)
        if f.endswith(".json")
    ]


def read_nodeset(nodeset_dir: str, nodeset_id: str) -> Nodeset:
    """Read a nodeset with a given ID from a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename) as f:
        return json.load(f)


def write_nodeset(nodeset_dir: str, nodeset_id: str, data: Nodeset) -> None:
    """Write a nodeset with a given ID to a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def process_all_nodesets(
    nodeset_dir: str,
    func: Callable[..., FuncResult],
    show_progress: bool = True,
    nodeset_blacklist: Optional[List[str]] = None,
    **kwargs,
) -> Iterator[Tuple[str, Union[FuncResult, Exception]]]:
    """Process all nodesets in a directory.

    Args:
        nodeset_dir: The directory containing the nodesets.
        func: The function to apply to each nodeset.
        show_progress: Whether to show a progress bar.
        nodeset_blacklist: Whether to ignore some nodeset IDs.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        A tuple containing the nodeset ID and the result of applying the function.
        If an exception occurs, the result will be the exception.
    """

    nodeset_ids = get_nodeset_ids_from_directory(nodeset_dir=nodeset_dir)
    failed_nodesets = []
    for nodeset_id in tqdm.tqdm(
        nodeset_ids, desc="Processing nodesets", disable=not show_progress
    ):
        if nodeset_blacklist and nodeset_id in nodeset_blacklist:
            logger.info(f"Skipping nodeset {nodeset_id} from the blacklist.")
            continue
        try:
            nodeset = read_nodeset(nodeset_dir=nodeset_dir, nodeset_id=nodeset_id)
            result = func(nodeset=nodeset, nodeset_id=nodeset_id, **kwargs)
            yield nodeset_id, result
        except Exception as e:
            failed_nodesets.append((nodeset_id, e))
            yield nodeset_id, e

    logger.info(
        f"Successfully processed {len(nodeset_ids) - len(failed_nodesets)} nodesets. "
        f"Failed to process the following nodesets: {failed_nodesets}"
    )


def get_node_ids(node_id2node: Dict[str, Any], allowed_node_types: List[str]) -> List[str]:
    """Get the IDs of nodes with a given type."""

    return [
        node_id for node_id, node in node_id2node.items() if node["type"] in allowed_node_types
    ]


def create_edges_from_relations(
    relations: List[Relation],
    edges: List[Edge],
) -> List[Edge]:
    """Create edge objects from binary_relations.

    Args:
        relations: A list of relations.
        edges: A list of edge objects where each object contains the keys "fromID" and "toID".

    Returns:
        A list of edge objects where each object contains the keys "fromID" and "toID".
    """
    biggest_edge_id = max([int(edge["fromID"]) for edge in edges])
    new_edges: List[Edge] = []
    for rel in relations:
        for src_id in rel["sources"]:
            biggest_edge_id += 1
            new_edges.append(
                {"fromID": src_id, "toID": rel["relation"], "edgeID": str(biggest_edge_id)}
            )
        for trg_id in rel["targets"]:
            biggest_edge_id += 1
            new_edges.append(
                {"fromID": rel["relation"], "toID": trg_id, "edgeID": str(biggest_edge_id)}
            )
    return new_edges


def create_relation_nodes_from_alignment(
    node_id2node: Dict[str, Node],
    node_alignments: List[Tuple[str, str]],
    node_type: str,
    node_text: str,
    swap_direction: bool = False,
) -> Tuple[List[Relation], Dict[str, Node]]:

    """Create relation nodes from alignments between two nodes.

    Args:
        node_id2node: A dictionary mapping node IDs to Node objects.
        node_alignments: A list of tuples containing the alignment between two nodes.
        node_type: The type of the nodes.
        node_text: The text of the nodes.
        swap_direction: A boolean indicating whether to swap the direction of the alignment
            before creating the relation node.

    Returns:
        A tuple containing:
         - a list of binary YA relations: tuples containing the source node ID, target node ID, and YA node ID, and
         - a dictionary containing the newly created YA nodes as a mapping from IDs to the respective Node objects.
    """
    biggest_node_id = max([int(node_id) for node_id in node_id2node.keys()])
    new_node_id2node: Dict[str, Node] = dict()
    relations: List[Relation] = []
    for src_id, trg_id in node_alignments:
        if swap_direction:
            src_id, trg_id = trg_id, src_id
        biggest_node_id += 1
        node_id = str(biggest_node_id)
        new_node_id2node[node_id] = {
            "nodeID": node_id,
            "type": node_type,
            "text": node_text,
        }
        relations.append({"sources": [src_id], "targets": [trg_id], "relation": node_id})

    return relations, new_node_id2node


# TODO: remove when everything is converted to the non-binary version
def get_binary_relations(
    node_id2node: Dict[str, Node],
    edges: List[Edge],
    allowed_node_types: Optional[List[str]] = None,
    allowed_source_types: Optional[List[str]] = None,
    allowed_target_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Create binary relations from nodes, i.e. tuples containing the source node ID, target node
    ID, and relation node ID.

    Args:
        node_id2node: A dictionary mapping node IDs to Node objects.
        edges: A list of Edge objects.
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

    binary_relations = []
    for node_id, node in node_id2node.items():
        # filter nodes based on allowed types
        if allowed_node_types is not None and node["type"] not in allowed_node_types:
            continue
        # iterate over all source nodes ...
        for src_id in trg2sources[node_id]:
            src_node = node_id2node[src_id]
            if allowed_source_types is None or src_node["type"] in allowed_source_types:
                # ... and all target nodes
                for trg_id in src2targets[node_id]:
                    trg_node = node_id2node[trg_id]
                    if allowed_target_types is None or trg_node["type"] in allowed_target_types:
                        binary_relations.append((src_id, trg_id, node_id))
    return binary_relations


def get_two_hop_connections(
    src_node_ids: Collection[str], trg_node_ids: Collection[str], edges: Iterable[Edge]
) -> Set[Tuple[str, str, str]]:
    """Get all two-hop edges (i.e. binary relations) between the given node ids.

    Args:
        src_node_ids: A collection of source node ids.
        trg_node_ids: A collection of target node ids.
        edges: A collection of edges.

    Returns:
        A set of tuples containing the source node id, target node id, and relation node id.
    """

    src2targets = defaultdict(list)
    for edge in edges:
        src2targets[edge["fromID"]].append(edge["toID"])

    result = set()
    for src_node_id in src_node_ids:
        for rel_node_id in src2targets[src_node_id]:
            for trg_node_id in src2targets[rel_node_id]:
                if trg_node_id in trg_node_ids:
                    result.add((src_node_id, trg_node_id, rel_node_id))

    return result


def get_relations(
    nodeset: Nodeset, relation_type: str, enforce_cardinality: bool = False
) -> Iterator[Relation]:
    """Get all relations of a given type from a nodeset.

    Args:
        nodeset: A nodeset.
        relation_type: The type of the relations to extract.
        enforce_cardinality: Whether to enforce the cardinality constraints of the relation type.
            All relations need to have exactly one source and one target node, except for the "S"
            relation type which can have multiple source nodes.

    Returns:
        A list of binary relations: tuples containing the source node ID, target node ID, and relation node ID.
    """
    if relation_type == "TA":
        allowed_node_types = ["TA"]
        allowed_source_types = ["L"]
        allowed_target_types = ["L"]
        allowed_max_sources = 1
        allowed_max_targets = 1
    elif relation_type == "S":
        allowed_node_types = ["RA", "CA", "MA"]
        allowed_source_types = ["I"]
        allowed_target_types = ["I"]
        allowed_max_sources = None
        allowed_max_targets = 1
    elif relation_type == "YA":
        allowed_node_types = ["YA"]
        allowed_source_types = ["L", "TA"]
        # Note: YA-relations L -> YA -> L encode (in-)direct speech
        allowed_target_types = ["I", "L", "RA", "CA", "MA"]
        allowed_max_sources = 1
        allowed_max_targets = 1
    else:
        raise ValueError(f"Unknown relation type: {relation_type}")

    # helper constructs
    node_id2node = {node["nodeID"]: node for node in nodeset["nodes"]}
    src2targets = defaultdict(list)
    trg2sources = defaultdict(list)
    for edge in nodeset["edges"]:
        src_id = edge["fromID"]
        trg_id = edge["toID"]
        src2targets[src_id].append(trg_id)
        trg2sources[trg_id].append(src_id)

    # get all relation nodes
    relation_node_ids = get_node_ids(node_id2node, allowed_node_types)
    for relation_node_id in relation_node_ids:
        # get all source and target nodes
        sources = [
            src_id
            for src_id in trg2sources[relation_node_id]
            if node_id2node[src_id]["type"] in allowed_source_types
        ]
        targets = [
            trg_id
            for trg_id in src2targets[relation_node_id]
            if node_id2node[trg_id]["type"] in allowed_target_types
        ]
        if enforce_cardinality:
            if len(sources) == 0 or len(targets) == 0:
                continue
            if allowed_max_sources is not None and len(sources) > allowed_max_sources:
                continue
            if allowed_max_targets is not None and len(targets) > allowed_max_targets:
                continue
        yield {
            "sources": sources,
            "targets": targets,
            "relation": relation_node_id,
        }


def remove_relation_nodes_and_edges(nodeset: Nodeset, relations: List[Relation]) -> Nodeset:
    """Remove relation nodes and the respective edges from a nodeset.

    Args:
        nodeset: A nodeset.
        relations: A list of binary relations: tuples containing the source node ID, target node ID,
            and relation node ID.
    """
    # create a copy of the nodeset to avoid modifying the original
    result = nodeset.copy()
    # helper sets
    relation_node_ids = {rel["relation"] for rel in relations}
    relation_edges = set()
    for rel in relations:
        for src_id in rel["sources"]:
            relation_edges.add((src_id, rel["relation"]))
        for trg_id in rel["targets"]:
            relation_edges.add((rel["relation"], trg_id))

    # filter out all nodes that are not relation nodes
    result["nodes"] = [
        node for node in nodeset["nodes"] if node["nodeID"] not in relation_node_ids
    ]
    # filter out all edges that are not connected to relation nodes
    result["edges"] = [
        edge for edge in nodeset["edges"] if (edge["fromID"], edge["toID"]) not in relation_edges
    ]

    return result


def remove_isolated_nodes(node_ids: List[str], edges: List[Edge]) -> List[str]:
    """Remove isolated nodes from a list of node IDs.

    Args:
        node_ids: A list of node IDs.
        edges: A list of edge objects where each object contains the keys "fromID" and "toID".

    Returns:
        A list of node IDs that are not isolated.
    """
    # create a set of all node IDs that are connected to at least one edge
    connected_node_ids = {edge["fromID"] for edge in edges} | {edge["toID"] for edge in edges}
    # filter out all node IDs that are not connected to any edge
    return [node_id for node_id in node_ids if node_id in connected_node_ids]


def sort_nodes_by_hierarchy(node_ids: Collection[str], edges: Collection[Edge]) -> List[str]:
    """Sort nodes in reversed depth-first order. The nodes are sorted in such a way that parents
    are always before children.

    Args:
        node_ids (Collection[str]): List of ids of nodes to sort.
        edges (Collection[Edge]): List of edges.

    Returns:
        List[str]: List of sorted node ids.
    """

    # first, get all two-hop edges between the given node ids (i.e. bridging relation nodes)
    valid_binary_relations = get_two_hop_connections(
        src_node_ids=node_ids, trg_node_ids=node_ids, edges=edges
    )

    src2targets = defaultdict(list)
    trg2sources = defaultdict(list)
    for src, trg, _ in valid_binary_relations:
        src2targets[src].append(trg)
        trg2sources[trg].append(src)

    # do a reversed depth-first search starting from the leaves
    result_reverted = []
    # all nodes that are no source of a relation are leaves
    leaves = set(node_ids) - set(src2targets)
    visited = set()
    stack = list(leaves)
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        result_reverted.append(node_id)

        # add all parents to the stack where all children have been visited
        parents = trg2sources.get(node_id, [])
        for parent in parents:
            if parent is not None and all(child in visited for child in src2targets[parent]):
                stack.append(parent)

    result = list(reversed(result_reverted))
    return result


def get_relation_statistics(
    nodeset: Nodeset, nodeset_id: str
) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
    """Get statistics about the relations in a nodeset.

    Args:
        nodeset: A nodeset.
        nodeset_id: The ID of the nodeset.

    Returns:
        A dictionary containing the following keys:
        - "missed_edges": A list of missed edges.
        - "covered_edges": The number of covered edges.
        - "empty_sources": A list of relations with empty sources.
        - "empty_targets": A list of relations with empty targets.
        - "more_than_one_target": A list of relations with more than one target.
        - "type_combinations": A dictionary containing the number of occurrences for each type combination.
        - "edges_covered_multi_times": A list of edges that are covered by multiple relations.
        - "covered_relations": A dictionary containing the number of covered relations for each type.
    """
    node_id2node = {node["nodeID"]: node for node in nodeset["nodes"]}

    all_relations = {
        rel_type: list(get_relations(nodeset, relation_type=rel_type))
        for rel_type in ["TA", "S", "YA"]
    }
    covered_edges: Dict[Tuple[str, str], int] = Counter()
    empty_sources = set()
    empty_targets = set()
    more_than_one_target = set()
    type_combinations = list()
    for relation_type, relations in all_relations.items():
        for relation in relations:
            for src_id in relation["sources"]:
                covered_edges.update({(src_id, relation["relation"]): 1})
            for trg_id in relation["targets"]:
                covered_edges.update({(relation["relation"], trg_id): 1})
            if not relation["sources"]:
                empty_sources.add(relation["relation"])
            if not relation["targets"]:
                empty_targets.add(relation["relation"])
            if len(relation["targets"]) > 1:
                more_than_one_target.add(relation["relation"])
            source_types = sorted(node_id2node[src_id]["type"] for src_id in relation["sources"])
            target_types = sorted(node_id2node[trg_id]["type"] for trg_id in relation["targets"])
            type_combinations.append(f"{relation_type}: {source_types} -> {target_types}")

    missed_edges = set((edge["fromID"], edge["toID"]) for edge in nodeset["edges"]) - set(
        covered_edges
    )
    missed_edges_with_types = [
        f"{src_id}:{trg_id} {node_id2node[src_id]['type']}:{node_id2node[trg_id]['type']}"
        for src_id, trg_id in missed_edges
    ]
    empty_sources_with_types = [
        f"{node_id} {node_id2node[node_id]['type']}" for node_id in empty_sources
    ]
    empty_targets_with_types = [
        f"{node_id} {node_id2node[node_id]['type']}" for node_id in empty_targets
    ]
    more_than_one_target_with_types = [
        f"{node_id} {node_id2node[node_id]['type']}" for node_id in more_than_one_target
    ]

    edges_covered_multi_times = (
        f"{src}:{trg}" for (src, trg), count in covered_edges.items() if count > 1
    )

    def prepend(items: Iterable[str], prefix: str) -> List[str]:
        """Prepend a prefix to all items in a list."""
        return [f"{prefix} {item}" for item in sorted(items)]

    return {
        "missed_edges": prepend(missed_edges_with_types, prefix=nodeset_id),
        "covered_edges": len(covered_edges),
        "empty_sources": prepend(empty_sources_with_types, prefix=nodeset_id),
        "empty_targets": prepend(empty_targets_with_types, prefix=nodeset_id),
        "more_than_one_target": prepend(more_than_one_target_with_types, prefix=nodeset_id),
        "type_combinations": dict(Counter(type_combinations)),
        "edges_covered_multi_times": prepend(edges_covered_multi_times, prefix=nodeset_id),
        "covered_relations": {
            rel_type: len(relations) for rel_type, relations in all_relations.items()
        },
    }


def main(
    nodeset_id: Optional[str] = None, input_dir: str = "data", show_progress: bool = True, **kwargs
) -> None:
    result: Dict[str, Any]
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = get_relation_statistics(nodeset=nodeset, nodeset_id=nodeset_id, **kwargs)
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        result = dict()
        for nodeset_id, result_or_error in process_all_nodesets(
            func=get_relation_statistics,
            nodeset_dir=input_dir,
            show_progress=show_progress,
            **kwargs,
        ):
            if isinstance(result_or_error, Exception):
                logger.error(f"nodeset={nodeset_id}: Failed to process: {result_or_error}")
            else:
                for stat_name, stat_value in result_or_error.items():
                    if stat_name not in result:
                        result[stat_name] = stat_value
                    else:
                        if isinstance(stat_value, int) and isinstance(result[stat_name], int):
                            result[stat_name] += stat_value
                        elif isinstance(stat_value, list) and isinstance(result[stat_name], list):
                            result[stat_name].extend(stat_value)
                        elif isinstance(stat_value, dict) and isinstance(result[stat_name], dict):
                            for key, value in stat_value.items():
                                if key not in result[stat_name]:
                                    result[stat_name][key] = value
                                else:
                                    result[stat_name][key] += value
                        else:
                            raise ValueError(f"Unexpected result type: {type(stat_value)}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process nodesets.")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="The directory containing the nodesets."
    )
    parser.add_argument("--nodeset_id", type=str, help="The ID of the nodeset to process.")
    parser.add_argument(
        "--silent", action="store_false", dest="show_progress", help="Disable progress bar."
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
