import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import copy
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from src.utils.nodeset_utils import (
    Nodeset,
    Relation,
    get_relations,
    process_all_nodesets,
    read_nodeset,
    remove_isolated_nodes,
    write_nodeset,
)

logger = logging.getLogger(__name__)

HELP_TEXT = """
Clean-up the data as follows:

0. Remove isolated nodes disconnected from the graph.
1. Remove invalid relation edges. We only allow the following transitions:
   I > S > I
   L > TA > L
   L > YA > I
   TA > YA > S
2. Swap the edges for S-nodes that point downwards.
"""


def cleanup_nodeset(
    nodeset: Nodeset, nodeset_id: str, normalize_relation_direction: bool, verbose: bool = True
) -> Nodeset:
    """Remove all edges from the nodeset that are not in valid transitions and remove isolated
    nodes. Optionally, normalize the relation direction.

    Args:
        nodeset: A Nodeset.
        nodeset_id: A Nodeset ID.
        normalize_relation_direction: Whether to set all relations in the same direction (this affects RA-nodes).
        verbose: Whether to show verbose output.

    Returns:
        Nodeset without isolated nodes and invalid transitions.
    """
    # node helper dictionary
    node_id2node = {node["nodeID"]: node for node in nodeset["nodes"]}

    # collect valid I > S > I relations
    s_relations = list(get_relations(nodeset, "S", enforce_cardinality=True))

    # collect valid L > TA > L relations
    # collect valid L > YA > I relations
    ya_relations = list(get_relations(nodeset, "YA", enforce_cardinality=True))

    # collect valid TA > YA > S relations
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))

    relations_to_keep = s_relations + ya_relations + ta_relations

    # helper sets
    src_tgt_rel_nodes = set()
    for rel in relations_to_keep:
        for node_id in rel["sources"] + rel["targets"] + [rel["relation"]]:
            src_tgt_rel_nodes.add(node_id)

    # remove isolated nodes
    valid_node_ids = remove_isolated_nodes(
        node_ids=list(src_tgt_rel_nodes), edges=nodeset["edges"]
    )

    # remove invalid relations
    valid_edges = get_invalid_edges(relations_to_keep, valid_node_ids=valid_node_ids)

    # create a copy of the nodeset to avoid modifying the original
    result = nodeset.copy()

    # nodes in valid relations
    result["nodes"] = [node for node in nodeset["nodes"] if (node["nodeID"] in valid_node_ids)]
    # edges in valid relations
    result["edges"] = [
        edge for edge in nodeset["edges"] if (edge["fromID"], edge["toID"]) in valid_edges
    ]

    if normalize_relation_direction:
        # reverse S-node relations

        reversed_s_relations = get_reversed_s_relations(
            nodeset=result,
            nodeset_id=nodeset_id,
            verbose=verbose,
            node_id2node=node_id2node,
        )
        result = reverse_relations_nodes(
            relations=reversed_s_relations,
            nodeset=result,
            nodeset_id=nodeset_id,
            reversed_text_suffix="-rev",
            redo=False,
        )
    return result


def get_invalid_edges(
    relations_to_keep: List[Relation], valid_node_ids: List[str]
) -> List[Tuple[str, str]]:
    """Remove all relations that do not correspond to the patterns specified in relations_to_keep.

    Args:
        relations_to_keep: List of relations to keep: (source, target, relation).
        valid_node_ids: Which node IDs are allowed (i.e., not isolated from the rest).

    Returns:
        Set of allowed relation edges.
    """
    # relation edges to keep
    valid_edges = []
    for rel in relations_to_keep:
        if all(
            node_id in valid_node_ids
            for node_id in rel["sources"] + rel["targets"] + [rel["relation"]]
        ):
            for src_id in rel["sources"]:
                valid_edges.append((src_id, rel["relation"]))
            for trg_id in rel["targets"]:
                valid_edges.append((rel["relation"], trg_id))

    return valid_edges


def reverse_relations_nodes(
    relations: Iterator[Relation],
    nodeset: Nodeset,
    nodeset_id: str,
    reversed_text_suffix: str = "-rev",
    redo: bool = False,
) -> Nodeset:
    """Reverse the direction of the relations in the nodeset.

    Args:
        relations: Iterator over relations.
        nodeset: Nodeset.
        nodeset_id: Nodeset ID.
        reversed_text_suffix: Suffix to append to the type of the reversed relation node.
        redo: If True, the function will reverse the reversed relations back to the original state.

    Returns:
        Nodeset with reversed relations.
    """
    # create a copy of the nodeset to avoid modifying the original
    result = copy.deepcopy(nodeset)
    # helper constructs
    node_id2nodes = {node["nodeID"]: node for node in result["nodes"]}
    edges_dict = {(edge["fromID"], edge["toID"]): edge for edge in result["edges"]}
    # we want to reverse each edge only once
    reversed_edges: Set[Tuple[str, str]] = set()
    # we want to reverse each relation node type only once
    # reversed_rel_types: Set[str] = set()
    for rel in relations:
        # append (or remove) -rev to the text of the relation node
        # if rel_id not in reversed_rel_types:
        rel_id = rel["relation"]
        node_text = node_id2nodes[rel_id]["text"]
        if not redo:
            node_id2nodes[rel_id]["text"] = f"{node_text}{reversed_text_suffix}"
        else:
            if not node_text.endswith(reversed_text_suffix):
                raise ValueError(f"nodeset={nodeset_id}: Node {rel_id} is not reversed!")
            node_id2nodes[rel_id]["text"] = node_text[: -len(reversed_text_suffix)]
        reversed_rel_node_type = node_id2nodes[rel_id]["type"]
        # warn if the reversed S-node type is not RA (should not happen!)
        if reversed_rel_node_type != "RA":
            logger.warning(
                f"nodeset={nodeset_id}: Relation node {rel_id} of type {reversed_rel_node_type} was reversed."
            )
        # reversed_rel_types.add(rel_id)
        current_edges = [(src_id, rel_id) for src_id in rel["sources"]] + [
            (rel_id, trg_id) for trg_id in rel["targets"]
        ]
        # swap edges
        for src_trg in current_edges:
            if src_trg not in reversed_edges:
                edge = edges_dict[src_trg]
                edge["fromID"], edge["toID"] = edge["toID"], edge["fromID"]
                reversed_edges.add(src_trg)

    return result


def get_l_anchor_nodes(
    i_node_id: str,
    ya_trg2sources: Dict[str, List[str]],
) -> List[str]:
    mapped_ids = ya_trg2sources.get(i_node_id, [])
    result = []
    for mapped_id in mapped_ids:
        # (in-)direct speech is encoded as another step of YA-relation
        if mapped_id in ya_trg2sources:
            result.extend(ya_trg2sources[mapped_id])
        else:
            result.append(mapped_id)
    # if len(result) != 1:
    #    logger.warning(
    #        f"Could not find a single anchor node for I-node {i_node_id} "
    #        f"(found {len(result)} anchor nodes)!"
    #    )
    return result


def get_reversed_s_relations(
    nodeset: Nodeset,
    nodeset_id: str,
    node_id2node: Dict[str, Any],
    verbose: bool = True,
) -> Iterator[Relation]:
    """Collect all S-node relations that need to be reversed (this affects RA-nodes).

    Args:
        nodeset: Nodeset.
        nodeset_id: Nodeset ID.
        node_id2node: A dictionary mapping node IDs to Node objects.
        verbose: Whether to show verbose output.

    Returns:
        Iterator over the S-node relations that need to be reversed.
    """
    s_relations = list(get_relations(nodeset, "S", enforce_cardinality=True))
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))
    ya_relations = list(get_relations(nodeset, "YA", enforce_cardinality=True))

    # helper structures
    ya_trg2sources = defaultdict(list)
    for rel in ya_relations:
        trg_id = rel["targets"][0]
        src_id = rel["sources"][0]
        ya_trg2sources[trg_id].append(src_id)
    ta_src_trg = {(rel["sources"][0], rel["targets"][0]) for rel in ta_relations}

    # collect for each S-node all source-anchor and target-anchor pairs
    already_checked: Dict[str, bool] = dict()
    for s_rel in s_relations:
        rel_id = s_rel["relation"]
        # get all anchors (L-nodes) for I-source nodes
        i_source_multi_anchor_nodes = [
            get_l_anchor_nodes(src_id, ya_trg2sources) for src_id in s_rel["sources"]
        ]
        # get all anchors (L-nodes) for I-target nodes
        i_target_multi_anchor_nodes = [
            get_l_anchor_nodes(trg_id, ya_trg2sources) for trg_id in s_rel["targets"]
        ]
        if any(
            len(anchors) != 1
            for anchors in i_source_multi_anchor_nodes + i_target_multi_anchor_nodes
        ):
            # notes regarding the s-relation arguments in the (cleaned) data:
            #  - 22408 have one anchor node
            #  - 3236 do not have any anchor node
            #  - 469 have two anchor nodes
            #  - 14 have three anchor nodes
            # TODO: we will ignore these cases for now
            continue
        i_source_anchor_nodes = [anchors[0] for anchors in i_source_multi_anchor_nodes]
        i_target_anchor_nodes = [anchors[0] for anchors in i_target_multi_anchor_nodes]

        # TODO: correctly un-binarize?
        for i_source_anchor in i_source_anchor_nodes:
            for i_target_anchor in i_target_anchor_nodes:
                # keep only pairs in s_node2source_target_pairs that appear in binary TA-relations
                if (i_source_anchor, i_target_anchor) in ta_src_trg:
                    if rel_id in already_checked and not already_checked[rel_id]:
                        raise ValueError(f"direction of S-node {rel_id} is ambiguous!")
                    already_checked[rel_id] = True
                    yield s_rel
                elif (i_target_anchor, i_source_anchor) in ta_src_trg:
                    if rel_id in already_checked and already_checked[rel_id]:
                        raise ValueError(f"direction of S-node {rel_id} is ambiguous!")
                    already_checked[rel_id] = False
                # else:
                #    raise ValueError(
                #        f"nodeset={nodeset_id}: Could not find TA-relation for S-node {rel_id}!"
                #    )

    for s_rel in s_relations:
        rel_id = s_rel["relation"]
        if verbose and rel_id not in already_checked:
            rel_node = node_id2node[rel_id]
            logger.warning(
                f"nodeset={nodeset_id}: could not determine direction of S-node {rel_id} "
                f"(type: {rel_node['type']}) because of missing source/target anchor node(s)!"
            )


def main(
    input_dir: str,
    output_dir: str,
    show_progress: bool = True,
    normalize_relation_direction: bool = False,
    nodeset_id: Optional[str] = None,
    nodeset_blacklist: Optional[List[str]] = None,
    **kwargs,
):
    # create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = cleanup_nodeset(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            normalize_relation_direction=normalize_relation_direction,
            **kwargs,
        )
        write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=cleanup_nodeset,
            nodeset_dir=input_dir,
            normalize_relation_direction=normalize_relation_direction,
            show_progress=show_progress,
            nodeset_blacklist=nodeset_blacklist,
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
        "--nodeset_blacklist",
        "--list",
        type=str,
        default=None,
        help="List of nodeset IDs that should be ignored.",
    )
    parser.add_argument(
        "--normalize_relation_direction",
        dest="normalize_relation_direction",
        action="store_true",
        help="Whether to normalize the direction of edges in the graph.",
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
        help="Whether to show verbose output.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
