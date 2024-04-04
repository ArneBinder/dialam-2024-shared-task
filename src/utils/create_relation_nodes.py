import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.align_i2l_nodes import align_i_and_l_nodes
from src.utils.nodeset_utils import (
    Nodeset,
    Relation,
    create_edges_from_relations,
    create_relation_nodes_from_alignment,
    get_node_ids,
    get_relations,
    process_all_nodesets,
    read_nodeset,
    remove_isolated_nodes,
    remove_relation_nodes_and_edges,
    write_nodeset,
)

logger = logging.getLogger(__name__)

HELP_TEXT = """
Create S and YA relations from L- and I-nodes and TA relation nodes.

The algorithm works as follows:
0. Remove existing S and YA nodes and their edges if they exist.
1. Align I and L nodes based on the similarity of their texts.
2. Create S nodes and align them with TA nodes by mirroring TA relations between L nodes to
    the aligned I nodes (see step 1).
3. Create YA nodes and relations from I-L and S-TA alignments.

Important Disclaimer:
- This creates relation nodes with a generic text (and type) per S-/YA-node.
- The direction of the new S nodes may not be correct and need to be adjusted later on.
"""


def create_s_relations_and_nodes_from_ta_nodes_and_il_alignment(
    node_id2node: Dict[str, Any],
    ta_relations: List[Relation],
    il_node_alignment: List[Tuple[str, str]],
    s_node_type: str,
    s_node_text: str,
) -> Tuple[List[Relation], Dict[str, Any], List[Tuple[str, str]]]:
    """Create S nodes from TA nodes by mirroring TA relations to S relations.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        ta_relations: A list of TA relations.
        il_node_alignment: A list of tuples containing the alignment between L and I nodes.
        s_node_type: The type of the S nodes.
        s_node_text: The text of the S nodes.

    Returns:
        A tuple containing:
         - a list of new S relations,
         - a dictionary containing the newly created S nodes as a mapping from IDs to node content, and
         - a list of tuples containing the alignment between S and TA nodes.
    """

    # there are only 13 nodesets in the DialAM data that can have multiple I-nodes (no more than 2) for a single L-node, hence we can just revert the dictionary
    # helper dictionary to map L node IDs to aligned I node IDs
    l2i_node_id = {l_id: i_id for i_id, l_id in il_node_alignment}
    # we need to keep track of the biggest node ID to create new S nodes
    biggest_node_id = max([int(node_id) for node_id in node_id2node.keys()])
    # mirror TA relations to S relations
    s_relations: List[Relation] = []
    sat_node_alignment = []
    new_node_id2node = dict()
    for ta_relation in ta_relations:
        for src_or_trg_id in ta_relation["sources"] + ta_relation["targets"]:
            if src_or_trg_id not in l2i_node_id:
                # skip TA relations where the source or target L nodes are not aligned with I nodes
                continue
        # create a new S node
        biggest_node_id += 1
        s_node_id = str(biggest_node_id)
        new_node_id2node[s_node_id] = {
            "nodeID": s_node_id,
            "type": s_node_type,
            "text": s_node_text,
        }
        # map L-source and L-target node IDs to their corresponding S node IDs.
        # note that we swap the direction because for most of the S nodes (MA and CA), they point
        # in the opposite direction of the TA nodes
        # we also have to check whether node_id appears in l2i_node_id dictionary because not all L-nodes have a corresponding I-node (e.g., L-node 713369 in nodeset 21388)
        s_src_ids = [
            l2i_node_id[node_id] for node_id in ta_relation["targets"] if node_id in l2i_node_id
        ]
        s_trg_ids = [
            l2i_node_id[node_id] for node_id in ta_relation["sources"] if node_id in l2i_node_id
        ]
        s_relations.append({"sources": s_src_ids, "targets": s_trg_ids, "relation": s_node_id})
        # collect the alignment between the S and TA nodes
        sat_node_alignment.append((s_node_id, ta_relation["relation"]))

    return s_relations, new_node_id2node, sat_node_alignment


def remove_s_and_ya_nodes_with_edges(
    nodeset: Nodeset, nodeset_id: Optional[str] = None, verbose: bool = True
) -> Nodeset:
    """Remove S and YA nodes and their edges from the nodeset.

    Args:
        nodeset: A Nodeset.
        nodeset_id: The ID of the nodeset for better logging. Defaults to None.
        verbose: A boolean indicating whether to show warnings for nodesets with remaining S or YA

    Returns:
        Nodeset with S and YA nodes and their edges removed.
    """
    # collect S and YA relations
    s_relations = list(get_relations(nodeset=nodeset, relation_type="S"))
    ya_relations = list(get_relations(nodeset=nodeset, relation_type="YA"))
    # remove S and YA nodes and their edges
    result = remove_relation_nodes_and_edges(nodeset=nodeset, relations=s_relations + ya_relations)
    return result


def add_s_and_ya_nodes_with_edges(
    nodeset: Nodeset,
    s_node_text: str,
    ya_node_text: str,
    s_node_type: str = "S",
    similarity_measure: str = "lcsstr",
    nodeset_id: Optional[str] = None,
    remove_existing_s_and_ya_nodes: bool = True,
    verbose: bool = True,
) -> Nodeset:
    f"""{HELP_TEXT}

    Disclaimer:
    - This creates relation nodes with a generic text (and type) per S-/YA-node.
    - The direction of the new S nodes may not be correct and need to be adjusted later on.

    Args:
        nodeset: A Nodeset
        s_node_text: The text of the S nodes.
        ya_node_text: The text of the YA nodes.
        s_node_type: The type of the S nodes. Defaults to "S".
        similarity_measure: The similarity measure to use for creating YA nodes. Defaults to
            "lcsstr" (Longest common substring).
        nodeset_id: The ID of the nodeset for better logging. Defaults to None.
        remove_existing_s_and_ya_nodes: A boolean indicating whether to remove existing S and YA
            nodes and their edges before adding new S and YA nodes. Defaults to True.
        verbose: A boolean indicating whether to show warnings. Defaults to True.

    Returns:
        A Nodeset with S and YA nodes and their edges added.
    """
    if remove_existing_s_and_ya_nodes:
        # remove existing S and YA nodes and their edges
        nodeset = remove_s_and_ya_nodes_with_edges(nodeset=nodeset)
    else:
        # create a copy of the nodeset to avoid modifying the original nodeset
        nodeset = nodeset.copy()

    nodes = nodeset["nodes"]
    edges = nodeset["edges"]

    # node helper dictionary
    node_id2node = {node["nodeID"]: node for node in nodes}

    # sanity check: there should be no S or YA nodes in the input nodeset
    s_node_ids = get_node_ids(node_id2node=node_id2node, allowed_node_types=["MA", "RA", "CA"])
    if verbose and s_node_ids:
        logger.warning(f"nodeset={nodeset_id}: Input has still S nodes: {s_node_ids}")
    ya_node_ids = get_node_ids(node_id2node=node_id2node, allowed_node_types=["YA"])
    if verbose and ya_node_ids:
        logger.warning(f"nodeset={nodeset_id}: Input has still YA nodes: {ya_node_ids}")

    # get L and I node IDs
    l_node_ids_with_isolates = get_node_ids(node_id2node=node_id2node, allowed_node_types=["L"])
    # remove isolated L nodes
    l_node_ids = remove_isolated_nodes(node_ids=l_node_ids_with_isolates, edges=edges)
    i_node_ids = get_node_ids(node_id2node=node_id2node, allowed_node_types=["I"])
    # sanity check: all I nodes should be isolated
    i_nodes_without_isolates = remove_isolated_nodes(node_ids=i_node_ids, edges=edges)
    if verbose and i_nodes_without_isolates:
        logger.warning(
            f"nodeset={nodeset_id}: Input has still connected I nodes: {i_nodes_without_isolates}"
        )
    # align I and L nodes
    il_node_alignment = align_i_and_l_nodes(
        node_id2node=node_id2node,
        i_node_ids=i_node_ids,
        l_node_ids=l_node_ids,
        similarity_measure=similarity_measure,
        nodeset_id=nodeset_id,
    )
    # collect TA relations: (src_id, trg_id, ta_node_id) where src_id and trg_id are L nodes
    ta_relations = list(get_relations(nodeset=nodeset, relation_type="TA"))
    # copy the node_id2node dictionary to avoid modifying the original dictionary
    node_id2node = node_id2node.copy()
    # create S nodes and relations from TA nodes
    (
        s_relations,
        s_node_id2node,
        sat_node_alignment,
    ) = create_s_relations_and_nodes_from_ta_nodes_and_il_alignment(
        node_id2node=node_id2node,
        ta_relations=ta_relations,
        il_node_alignment=il_node_alignment,
        s_node_type=s_node_type,
        s_node_text=s_node_text,
    )
    node_id2node.update(s_node_id2node)
    # create YA nodes and relations I-L- and S-TA-alignments
    ya_relations, ya_node_id2node = create_relation_nodes_from_alignment(
        node_id2node=node_id2node,
        node_alignments=il_node_alignment + sat_node_alignment,
        node_type="YA",
        node_text=ya_node_text,
        # swap direction of the alignment to create the relations
        swap_direction=True,
    )
    node_id2node.update(ya_node_id2node)
    # create edges from S and YA relations
    new_edges = create_edges_from_relations(relations=s_relations + ya_relations, edges=edges)

    nodeset["nodes"] = list(node_id2node.values())
    nodeset["edges"] = edges + new_edges
    return nodeset


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
        result = add_s_and_ya_nodes_with_edges(nodeset=nodeset, nodeset_id=nodeset_id, **kwargs)
        write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=add_s_and_ya_nodes_with_edges,
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
        "--s_node_type",
        type=str,
        default="RA",
        help="The type of the new S nodes. Default is 'RA'.",
    )
    parser.add_argument(
        "--s_node_text",
        type=str,
        default="DUMMY",
        help="The text of the new S nodes. Default is 'DUMMY'.",
    )
    parser.add_argument(
        "--ya_node_text",
        type=str,
        default="DUMMY",
        help="The text of the new YA nodes. Default is 'DUMMY'.",
    )
    parser.add_argument(
        "--similarity_measure",
        type=str,
        default="lcsstr",
        help="The similarity measure to use for creating YA nodes. Default is 'lcsstr' (Longest common substring).",
    )
    parser.add_argument(
        "--nodeset_id",
        type=str,
        default=None,
        help="The ID of the nodeset to process. If not provided, all nodesets in the input directory will be processed.",
    )
    parser.add_argument(
        "--dont_remove_existing_s_and_ya_nodes",
        dest="remove_existing_s_and_ya_nodes",
        action="store_false",
        help="Whether to remove existing S and YA nodes and their edges before adding new S and YA nodes.",
    )
    parser.add_argument(
        "--dont_show_progress",
        dest="show_progress",
        action="store_false",
        help="Whether to show a progress bar when processing multiple nodesets.",
    )
    parser.add_argument(
        "--silent",
        dest="verbose",
        action="store_false",
        help="Whether to show warnings for nodesets with remaining S or YA nodes.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
