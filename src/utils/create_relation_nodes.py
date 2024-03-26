import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.align_i2l_nodes import align_i_and_l_nodes
from src.utils.nodeset_utils import (
    create_edges_from_relations,
    create_relation_nodes_from_alignment,
    get_binary_relations,
    get_node_ids,
    process_all_nodesets,
    read_nodeset,
    write_nodeset,
)

logger = logging.getLogger(__name__)


def get_binary_ta_relations(
    node_id2node: Dict[str, Any],
    edges: List[Dict[str, str]],
) -> List[Tuple[str, str, str]]:
    """Get TA relations from nodes.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        edges: A list of edge objects where each object contains the keys "fromID" and "toID".

    Returns:
        A list of binary TA relations: tuples containing the source node ID, target node ID, and TA node ID.
    """
    # collect TA relations: (src_id, trg_id, ta_node_id) where src_id and trg_id are L nodes
    return get_binary_relations(
        node_id2node=node_id2node,
        edges=edges,
        allowed_node_types=["TA"],
        # TA relations are always between L nodes
        allowed_source_types=["L"],
        allowed_target_types=["L"],
    )


def create_s_relations_and_nodes_from_ta_nodes_and_il_alignment(
    node_id2node: Dict[str, Any],
    ta_relations: List[Tuple[str, str, str]],
    il_node_alignment: List[Tuple[str, str]],
    s_node_type: str,
    s_node_text: str,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any], List[Tuple[str, str]]]:
    """Create S nodes from TA nodes by mirroring TA relations to S relations.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        ta_relations: A list of binary TA relations, i.e. tuples containing the source node ID, target node ID, and TA node ID.
        il_node_alignment: A list of tuples containing the alignment between L and I nodes.
        s_node_type: The type of the S nodes.
        s_node_text: The text of the S nodes.

    Returns:
        A tuple containing:
         - a list of binary S relations: tuples containing the source node ID, target node ID, and S node ID,
         - a dictionary containing the newly created S nodes as a mapping from IDs to node content, and
         - a list of tuples containing the alignment between S and TA nodes.
    """

    # TODO: is it fine to create a dictionary mapping L node IDs to I node IDs like that? or
    #  can there be multiple I nodes for a single L node?
    # helper dictionary to map L node IDs to aligned I node IDs
    l2i_node_id = {l_id: i_id for i_id, l_id in il_node_alignment}
    ta2s_id: Dict[str, str] = dict()
    # we need to keep track of the biggest node ID to create new S nodes
    biggest_node_id = max([int(node_id) for node_id in node_id2node.keys()])
    # mirror TA relations to S relations
    s_relations = []
    sat_node_alignment = []
    new_node_id2node = dict()
    for src_id, trg_id, ta_node_id in ta_relations:
        # we may have already created an S node for the TA node
        s_node_id = ta2s_id.get(ta_node_id, None)
        # if not, create a new S node
        if s_node_id is None:
            biggest_node_id += 1
            s_node_id = str(biggest_node_id)
            ta2s_id[ta_node_id] = s_node_id
            new_node_id2node[s_node_id] = {
                "nodeID": s_node_id,
                "type": s_node_type,
                "text": s_node_text,
            }
        # map L-source and L-target node IDs to their corresponding S node IDs
        s_src_id = l2i_node_id[src_id]
        s_trg_id = l2i_node_id[trg_id]
        s_relations.append((s_src_id, s_trg_id, s_node_id))
        # collect the alignment between the S and TA nodes
        sat_node_alignment.append((s_node_id, ta_node_id))

    return s_relations, new_node_id2node, sat_node_alignment


def add_s_and_ya_nodes_with_edges(
    nodeset: Dict[str, List[Dict[str, str]]],
    s_node_text: str,
    ya_node_text: str,
    s_node_type: str = "S",
    similarity_measure: str = "lcsstr",
    nodeset_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Create S and YA relations from L- and I-nodes and TA relations. The algorithm works as follows:
    1. Align I and L nodes based on the similarity of their texts.
    2. Create S nodes and align them with TA nodes by mirroring TA relations between L nodes to
        the aligned I nodes (see step 1).
    3. Create YA nodes and relations from I-L- and S-TA-add_s_and_ya_nodes_with_edgesalignments.

    Disclaimer:
    - This creates relation nodes with a generic text (and type) per S-/YA-node.
    - The direction of the new S nodes may not be correct and need to be adjusted later on.

    Args:
        nodeset: A dictionary containing the keys "nodes" and "edges" where "nodes" is a list of
            node objects (each entry with "nodeID", "type", and "text") and "edges" is a list of
            edge objects (each entry with "fromID" and "toID").
        s_node_text: The text of the S nodes.
        ya_node_text: The text of the YA nodes.
        s_node_type: The type of the S nodes. Defaults to "S".
        similarity_measure: The similarity measure to use for creating YA nodes. Defaults to
            "lcsstr" (Longest common substring).
        nodeset_id: The ID of the nodeset for better logging. Defaults to None.

    Returns:
        A tuple containing:
        - a list of node objects containing the newly created S and YA nodes, and
        - a list of edge objects containing the newly created edges.
    """
    nodes = nodeset["nodes"]
    edges = nodeset["edges"]

    # node helper dictionary
    node_id2node = {node["nodeID"]: node for node in nodes}
    # get I and L node IDs
    i_node_ids = get_node_ids(node_id2node=node_id2node, allowed_node_types=["I"])
    l_node_ids = get_node_ids(node_id2node=node_id2node, allowed_node_types=["L"])
    # align I and L nodes
    il_node_alignment = align_i_and_l_nodes(
        node_id2node=node_id2node,
        i_node_ids=i_node_ids,
        l_node_ids=l_node_ids,
        similarity_measure=similarity_measure,
        nodeset_id=nodeset_id,
    )
    # collect TA relations: (src_id, trg_id, ta_node_id) where src_id and trg_id are L nodes
    ta_relations = get_binary_ta_relations(node_id2node=node_id2node, edges=edges)
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

    new_nodeset = nodeset.copy()
    new_nodeset["nodes"] = list(node_id2node.values())
    new_nodeset["edges"] = edges + new_edges
    return new_nodeset


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
        description="Create S and YA nodes from I, L and TA nodes. See add_s_and_ya_nodes_with_edges() "
        "for more details."
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
        "--s_node_type", type=str, default="S", help="The type of the new S nodes."
    )
    parser.add_argument(
        "--s_node_text", type=str, default="DUMMY", help="The text of the new S nodes."
    )
    parser.add_argument(
        "--ya_node_text", type=str, default="DUMMY", help="The text of the new YA nodes."
    )
    parser.add_argument(
        "--similarity_measure",
        type=str,
        default="lcsstr",
        help="The similarity measure to use for creating YA nodes.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)