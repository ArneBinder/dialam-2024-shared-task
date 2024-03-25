from typing import Any, Dict, List, Tuple


def create_s_nodes_from_ta_nodes(
    node_id2node: Dict[str, Any],
    src2targets: Dict[str, List[str]],
    trg2sources: Dict[str, List[str]],
    ta_node_ids: list[str],
    il_node_alignment: List[Tuple[str, str]],
    s_node_type: str = "S",
    s_node_text: str = "DUMMY",
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """Create S nodes from TA nodes by mirroring TA relations to S relations.

    Args:
        node_id2node: A dictionary mapping node IDs to node objects.
        src2targets: A dictionary mapping source node IDs to target node IDs.
        trg2sources: A dictionary mapping target node IDs to source node IDs.
        ta_node_ids: A list of TA node IDs.
        il_node_alignment: A list of tuples containing the alignment between L and I nodes.
        s_node_type: The type of the S nodes.
        s_node_text: The text of the S nodes.

    Returns:
        A tuple containing:
         - a list of binary S relations, and
         - a dictionary mapping S node IDs to S node objects.
    """
    # collect TA relations: (src_id, trg_id, ta_node_id) where src_id and trg_id are L nodes
    ta_relations = []
    for ta_node_id in ta_node_ids:
        for src_id in trg2sources[ta_node_id]:
            src_node = node_id2node[src_id]
            if src_node["type"] == "L":
                for trg_id in src2targets[src_id]:
                    trg_node = node_id2node[trg_id]
                    if trg_node["type"] == "L":
                        ta_relations.append((src_id, trg_id, ta_node_id))

    # TODO: is it fine to create a dictionary mapping L node IDs to I node IDs like that? or
    #  can there be multiple I nodes for a single L node?
    l2i_node_id = {l_id: i_id for i_id, l_id in il_node_alignment}
    ta2s_id: Dict[str, str] = dict()
    biggest_node_id = max([int(node_id) for node_id in node_id2node.keys()])
    # mirror TA relations to S relations
    s_relations = []
    new_node_id2node = dict()
    for src_id, trg_id, ta_node_id in ta_relations:
        s_src_id = l2i_node_id[src_id]
        s_trg_id = l2i_node_id[trg_id]
        s_dummy_id = ta2s_id.get(ta_node_id, None)
        if s_dummy_id is None:
            biggest_node_id += 1
            s_dummy_id = str(biggest_node_id)
            ta2s_id[ta_node_id] = s_dummy_id
            new_node_id2node[s_dummy_id] = {
                "id": s_dummy_id,
                "type": s_node_type,
                "text": s_node_text,
            }
        s_relations.append((s_src_id, s_trg_id, s_dummy_id))

    return s_relations, new_node_id2node
