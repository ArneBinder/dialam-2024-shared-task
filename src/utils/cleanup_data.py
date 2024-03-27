import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils.nodeset_utils import (
    Nodeset,
    get_binary_relations,
    get_node_ids,
    process_all_nodesets,
    read_nodeset,
    remove_isolated_nodes,
    write_nodeset,
)

logger = logging.getLogger(__name__)

HELP_TEXT = """
Clean-up the data as follows:

0. Remove invalid relation edges. We only allow the following transitions:
   I > S > I
   L > TA > L
   L > YA > I
   TA > YA > S
1. Remove isolated nodes disconnected from the graph.
2. Add RA-rev type to S-nodes of RA type that point downward.
   "Downward" means that I-source node for RA is anchored through YA in TA node which has a source L-node that appears earlier in the graph than the corresponding L-node that is connected (through YA and TA-nodes) to I-target node.
"""


def cleanup_nodeset(nodeset: Nodeset, nodeset_id: str) -> Nodeset:
    """Remove all edges from the nodeset that are not in valid transitions and remove isolated
    nodes.

    Args:
        nodeset: A Nodeset.
        nodeset_id: A Nodeset ID.

    Returns:
        Nodeset without isolated nodes and invalid transitions.
    """
    # node helper dictionary
    node_id2node = {node["nodeID"]: node for node in nodeset["nodes"]}

    # remove isolated nodes
    node_ids = get_node_ids(
        node_id2node=node_id2node, allowed_node_types=["L", "I", "YA", "TA", "MA", "RA", "CA"]
    )

    # collect valid I > S > I relations
    i_s_i_relations = get_binary_relations(
        node_id2node=node_id2node,
        edges=nodeset["edges"],
        allowed_node_types=["MA", "RA", "CA"],  # S nodes can be of type MA, RA, or CA
        allowed_source_types=["I"],
        allowed_target_types=["I"],
    )

    # collect valid L > TA > L relations
    l_ta_l_relations = get_binary_relations(
        node_id2node=node_id2node,
        edges=nodeset["edges"],
        allowed_node_types=["TA"],
        allowed_source_types=["L"],
        allowed_target_types=["L"],
    )

    # collect valid L > YA > I relations
    l_ya_i_relations = get_binary_relations(
        node_id2node=node_id2node,
        edges=nodeset["edges"],
        allowed_node_types=["YA"],
        allowed_source_types=["L"],
        allowed_target_types=["I"],
    )

    # collect valid TA > YA > S relations
    ta_ya_s_relations = get_binary_relations(
        node_id2node=node_id2node,
        edges=nodeset["edges"],
        allowed_node_types=["YA"],
        allowed_source_types=["TA"],
        allowed_target_types=["MA", "RA", "CA"],  # S nodes can be of type MA, RA, or CA
    )

    # create a copy of the nodeset to avoid modifying the original
    result = nodeset.copy()
    valid_relations = i_s_i_relations + l_ta_l_relations + l_ya_i_relations + ta_ya_s_relations

    # helper sets
    src_tgt_rel_nodes = set()
    for rel in valid_relations:
        src_tgt_rel_nodes.update({rel[0], rel[1], rel[2]})
    # remove isolated nodes
    valid_node_ids = remove_isolated_nodes(
        node_ids=list(src_tgt_rel_nodes), edges=nodeset["edges"]
    )
    # relation edges to keep
    relation_edges = {
        (rel[0], rel[2])
        for rel in valid_relations
        if rel[0] in valid_node_ids and rel[1] in valid_node_ids
    } | {
        (rel[2], rel[1])
        for rel in valid_relations
        if rel[2] in valid_node_ids and rel[1] in valid_node_ids
    }

    # nodes in valid relations
    result["nodes"] = [node for node in nodeset["nodes"] if (node["nodeID"] in valid_node_ids)]
    # edges in valid relations
    result["edges"] = [
        edge for edge in nodeset["edges"] if (edge["fromID"], edge["toID"]) in relation_edges
    ]

    # add RA-rev type to all relation nodes that point downward in dialogue flow
    for node in result["nodes"]:
        if node["type"] == "RA" and check_if_ra_points_down(node["nodeID"], result, nodeset_id):
            node["type"] = "RA-rev"

    return result


def check_if_ra_points_down(ra_node_id: str, nodeset: Nodeset, nodeset_id: str):
    """Check whether RA-node point down in the argument map.

    Args:
        ra_node_id: ID of the RA node.
        nodeset: Nodeset.
        nodeset_id: A Nodeset ID.

    Returns:
        Boolean that is set to True if the RA-node points downward.
    """
    points_down = False
    # node helper dictionary
    node_id2node = {node["nodeID"]: node for node in nodeset["nodes"]}
    src2targets: Dict[str, List[str]] = defaultdict(list)
    trg2sources: Dict[str, List[str]] = defaultdict(list)
    for edge in nodeset["edges"]:
        src_id = edge["fromID"]
        trg_id = edge["toID"]
        src2targets[src_id].append(trg_id)
        trg2sources[trg_id].append(src_id)

    node2type: Dict[str, str] = defaultdict(str)
    for node_id in node_id2node:
        node2type[node_id] = node_id2node[node_id]["type"]
    # find I-source for RA-node
    i_sources = [n for n in trg2sources[ra_node_id] if node2type[n] == "I"]
    # ambiguous case if we have multiple I-node connections
    if len(i_sources) == 1:
        i_source = i_sources[0]
        # for I-source we find the corresponding L-node (through YA-node) and track down whether it appears in the sources of TA-node that connects to RA-node, if it does then we have "downward" transition and need to assign RA-rev label
        ya_source = find_first_match_source_node(i_source, "YA", node2type, trg2sources)
        l_source = None
        ta_source = None
        ra_ya_source = None
        if ya_source is not None:
            l_source = find_first_match_source_node(ya_source, "L", node2type, trg2sources)
        # find RA < YA < TA relation
        ra_ya_source = find_first_match_source_node(ra_node_id, "YA", node2type, trg2sources)
        if ra_ya_source is not None:
            ta_source = find_first_match_source_node(ra_ya_source, "TA", node2type, trg2sources)
        if ta_source is not None and l_source in trg2sources[ta_source]:
            points_down = True
    elif len(i_sources) > 1:
        logger.warning(
            f"nodeset={nodeset_id}: RA-node {ra_node_id} is connected to multiple I-nodes! Source nodes: {i_sources}"
        )
    return points_down


def find_first_match_source_node(
    target_node: str, node_type: str, node2type: Dict[str, str], trg2sources: Dict[str, List[str]]
):
    """Find (first match) source node of a specific type given the target node ID. We assume that
    we can take the first match because the target node can be anchored only in a single node of
    the given type (e.g., I in YA, YA in L, RA in YA or YA in TA).

    Args:
        target_node: Target node ID.
        nodeset_type: Type of the source node we are looking for.
        node2type: Mapping between the node ID and the node type.
        trg2sources: Mapping from each target node to the source nodes (by ID).

    Returns:
        Source node ID or None if it was not found.
    """

    source_node = None
    for source in trg2sources[target_node]:
        if node2type[source] == node_type:
            return source
    return source_node


def main(
    input_dir: str,
    output_dir: str,
    show_progress: bool = True,
    nodeset_id: Optional[str] = None,
    **kwargs,
):
    # create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = cleanup_nodeset(nodeset=nodeset, nodeset_id=nodeset_id, **kwargs)
        write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=cleanup_nodeset,
            nodeset_dir=input_dir,
            show_progress=show_progress,
            **kwargs,
        ):
            if isinstance(result_or_error, Exception):
                logger.error(f"nodeset={nodeset_id}: Failed to process: {result_or_error}")
            else:
                write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result_or_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=HELP_TEXT, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="The input directory containing the nodesets."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory for the modified nodesets.",
    )
    parser.add_argument(
        "--nodeset_id",
        type=str,
        default=None,
        help="The ID of the nodeset to process. If not provided, all nodesets in the input directory will be processed.",
    )
    parser.add_argument(
        "--dont_show_progress",
        dest="show_progress",
        action="store_false",
        help="Whether to show a progress bar when processing multiple nodesets.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
