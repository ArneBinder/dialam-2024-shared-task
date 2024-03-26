from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def create_relation_nodes_from_alignments(
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


def create_s_nodes_from_ta_nodes(
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
                "id": s_node_id,
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
