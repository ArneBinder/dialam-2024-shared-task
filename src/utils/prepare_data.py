import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import copy
import logging
import os
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from src.utils.create_relation_nodes import (
    add_s_and_ya_nodes_with_edges,
    remove_s_and_ya_nodes_with_edges,
)
from src.utils.nodeset_utils import (
    Nodeset,
    Relation,
    get_id2node,
    get_node_ids_by_type,
    get_relations,
    merge_other_into_nodeset,
    process_all_nodesets,
    read_nodeset,
    remove_relation_nodes_and_edges,
    write_nodeset,
)

logger = logging.getLogger(__name__)

HELP_TEXT = """
Clean-up the data as follows:

0. Remove isolated nodes disconnected from the graph.
1. Remove invalid relation edges. We only allow the following transitions:
   - I > {MA, RA, CA} > I
   - L > TA > L
   - {L, TA} > YA > {I, L, MA, RA, CA}
   and also, we allow just one source and target node for each relation,
   except for S-relations where we allow multiple sources.
2. Swap the edges for S-nodes that point downwards.
"""


def get_valid_relations(nodeset: Nodeset) -> List[Relation]:
    # collect L > TA > L relations
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))

    # collect valid I > {MA, RA, CA} > I relations
    s_relations = list(get_relations(nodeset, "S", enforce_cardinality=True))

    # collect {L, TA} > YA > {I, L, MA, RA, CA} relations
    ya_relations = list(get_relations(nodeset, "YA", enforce_cardinality=True))

    return s_relations + ya_relations + ta_relations


def normalize_ra_relation_direction(nodeset: Nodeset, nodeset_id: str) -> Nodeset:
    """Normalize the direction of the relations in the nodeset to point in the opposite direction
    as their anchoring TA-relation.

    Args:
        nodeset: Nodeset.
        nodeset_id: Nodeset ID.

    Returns:
        Nodeset with normalized relations.
    """
    reversed_ra_relations = get_reversed_ra_relations(
        nodeset=nodeset,
        nodeset_id=nodeset_id,
        verbose=True,
    )
    result = reverse_relations_nodes(
        relations=reversed_ra_relations,
        nodeset=nodeset,
        nodeset_id=nodeset_id,
        reversed_text_suffix="-rev",
        redo=False,
    )
    return result


def get_arguments(relation: Relation) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Get the source and target node ids from the relation as hashable tuples.

    Args:
        relation: A relation.

    Returns:
        Tuple of source and target nodes.
    """
    return tuple(sorted(set(relation["sources"]))), tuple(sorted(set(relation["targets"])))


def match_relation_nodes_by_arguments(
    nodeset: Nodeset,
    other: Nodeset,
    relation_type: str,
    node_matching: List[Tuple[str, str]],
    nodeset_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Match relation nodes from the nodeset to relation nodes from the other nodeset based on
    their arguments. This assumes that all relations in nodeset and other have unique argument
    sets.

    Args:
        nodeset: Nodeset.
        other: Other Nodeset.
        relation_type: Relation type.
        node_matching: Node matching pairs (node_id_in_nodeset, node_id_in_other).
        nodeset_id: Nodeset ID for better error messages.

    Returns:
        List of relation node matching pairs (node_id_in_nodeset, node_id_in_other).
    """
    this2other = {
        node_id_in_nodeset: node_id_in_other
        for node_id_in_nodeset, node_id_in_other in node_matching
    }
    result = []
    nodeset_relations = list(get_relations(nodeset, relation_type))
    other_relations = list(get_relations(other, relation_type))
    nodeset_arguments2node_id = {get_arguments(rel): rel["relation"] for rel in nodeset_relations}
    # sanity check if arguments are unique
    if len(nodeset_arguments2node_id) != len(nodeset_relations):
        nodeset_arguments2node_id_values = set(nodeset_arguments2node_id.values())
        missing_relations = [
            rel
            for rel in nodeset_relations
            if rel["relation"] not in nodeset_arguments2node_id_values
        ]
        raise ValueError(
            f"nodeset={nodeset_id}: S-node arguments are not unique! missing relations: {missing_relations}"
        )
    other_arguments2node_id = {get_arguments(rel): rel["relation"] for rel in other_relations}
    for (sources, targets), node_id in nodeset_arguments2node_id.items():
        mapped_sources = tuple(this2other.get(src, src) for src in sources)
        mapped_targets = tuple(this2other.get(trg, trg) for trg in targets)
        if (mapped_sources, mapped_targets) in other_arguments2node_id:
            other_node_id = other_arguments2node_id[(mapped_sources, mapped_targets)]
            result.append((node_id, other_node_id))
    return result


def get_node_matching(
    nodeset: Nodeset,
    other: Nodeset,
    nodeset_id: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Match nodes from the nodeset to nodes from the other nodeset. The workflow is as follows:
    1. We assume that L-nodes, and TA-nodes are identical, i.e. they can be matched by their ids.
    2. We assume that I-nodes can be matched by exact string matching.
    3. Then, S-nodes can be matched by their (already matched) source and target nodes.
    4. Finally, YA-nodes can be matched by their (already matched) source and target nodes.

    Args:
        nodeset: Nodeset.
        other: Other Nodeset.
        nodeset_id: Nodeset ID.
        verbose: Whether to show verbose output.

    Returns:
        List of node matching pairs (node_id_in_nodeset, node_id_in_other).
    """
    result: List[Tuple[str, str]] = []

    nodeset_id2node = get_id2node(nodeset)
    other_id2node = get_id2node(other)

    # 1. match L-nodes and TA-nodes by their ids
    l_node_ids = get_node_ids_by_type(nodeset, ["L"])
    other_l_node_ids = get_node_ids_by_type(other, ["L"])
    # sanity check
    if not set(l_node_ids) == set(other_l_node_ids):
        raise ValueError(
            f"nodeset={nodeset_id}: L-nodes do not match: {l_node_ids} vs. {other_l_node_ids}"
        )
    result.extend((node_id, node_id) for node_id in l_node_ids)
    ta_node_ids = get_node_ids_by_type(nodeset, ["TA"])
    other_ta_node_ids = get_node_ids_by_type(other, ["TA"])
    # sanity check
    if not set(ta_node_ids) == set(other_ta_node_ids):
        raise ValueError(
            f"nodeset={nodeset_id}: TA-nodes do not match: {ta_node_ids} vs. {other_ta_node_ids}"
        )
    result.extend((node_id, node_id) for node_id in ta_node_ids)

    # 2. match I-nodes by exact string matching
    i_node_ids = get_node_ids_by_type(nodeset, ["I"])
    other_i_node_ids = get_node_ids_by_type(other, ["I"])
    i_text2node_id = {nodeset_id2node[node_id]["text"]: node_id for node_id in i_node_ids}
    other_i_text2node_id = {
        other_id2node[node_id]["text"]: node_id for node_id in other_i_node_ids
    }
    # sanity check: check if texts are the same
    if not set(i_text2node_id) == set(other_i_text2node_id):
        raise ValueError(
            f"nodeset={nodeset_id}: I-nodes do not match: {i_text2node_id} vs. {other_i_text2node_id}"
        )
    # sanity check: check if texts are unique
    if len(i_text2node_id) != len(i_node_ids):
        raise ValueError(f"nodeset={nodeset_id}: I-node texts are not unique!")
    # match I-nodes by their text
    for i_text, node_id in i_text2node_id.items():
        other_node_id = other_i_text2node_id[i_text]
        result.append((node_id, other_node_id))

    # 3. match S-nodes by their (already matched) source and target nodes
    s_node_mapping = match_relation_nodes_by_arguments(
        nodeset=nodeset,
        other=other,
        relation_type="S",
        node_matching=result,
        nodeset_id=nodeset_id,
    )
    result.extend(s_node_mapping)

    # 4. match YA-nodes by their (already matched) source and target nodes
    ya_node_mapping = match_relation_nodes_by_arguments(
        nodeset=nodeset,
        other=other,
        relation_type="YA",
        node_matching=result,
        nodeset_id=nodeset_id,
    )
    result.extend(ya_node_mapping)

    return result


def cleanup_nodeset(nodeset: Nodeset, nodeset_id: str, verbose: bool = True) -> Nodeset:
    """Remove all edges from the nodeset that are not in valid transitions and remove isolated
    nodes. Optionally, normalize the relation direction.

    Args:
        nodeset: A Nodeset.
        nodeset_id: A Nodeset ID.
        verbose: Whether to show verbose output.

    Returns:
        Nodeset without isolated nodes and invalid transitions.
    """

    valid_relations = get_valid_relations(nodeset)

    # remove invalid relations
    valid_src_trg, valid_node_ids = get_valid_src_trg_and_node_ids_from_relations(valid_relations)

    # create a copy of the nodeset to avoid modifying the original
    result = nodeset.copy()

    # nodes in valid relations
    result["nodes"] = [node for node in nodeset["nodes"] if (node["nodeID"] in valid_node_ids)]
    # edges in valid relations
    result["edges"] = [
        edge for edge in nodeset["edges"] if (edge["fromID"], edge["toID"]) in valid_src_trg
    ]

    return result


def get_valid_src_trg_and_node_ids_from_relations(
    relations_to_keep: List[Relation], valid_node_ids: Optional[List[str]] = None
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """Remove all relations that do not correspond to the patterns specified in relations_to_keep.

    Args:
        relations_to_keep: List of relations to keep: (source, target, relation).
        valid_node_ids: Which node IDs are allowed (i.e., not isolated from the rest).

    Returns:
        Set of allowed relation edges.
    """
    # relation edges to keep
    valid_src_trg = []
    for rel in relations_to_keep:
        if valid_node_ids is None or all(
            node_id in valid_node_ids
            for node_id in rel["sources"] + rel["targets"] + [rel["relation"]]
        ):
            for src_id in rel["sources"]:
                valid_src_trg.append((src_id, rel["relation"]))
            for trg_id in rel["targets"]:
                valid_src_trg.append((rel["relation"], trg_id))

    valid_node_ids = []
    for src_id, trg_id in valid_src_trg:
        valid_node_ids.append(src_id)
        valid_node_ids.append(trg_id)

    return set(valid_src_trg), set(valid_node_ids)


def reverse_relations_nodes(
    relations: Iterable[Relation],
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


def get_reversed_ra_relations(
    nodeset: Nodeset,
    nodeset_id: str,
    verbose: bool = True,
) -> Iterator[Relation]:
    """Collect all S-node relations that need to be reversed (this affects RA-nodes).

    Args:
        nodeset: Nodeset.
        nodeset_id: Nodeset ID.
        verbose: Whether to show verbose output.

    Returns:
        Iterator over the S-node relations that need to be reversed.
    """
    ra_relations = list(get_relations(nodeset, "RA", enforce_cardinality=True))
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))
    ya_relations = list(get_relations(nodeset, "YA", enforce_cardinality=True))

    # helper structures
    ya_trg2sources = defaultdict(list)
    for rel in ya_relations:
        trg_id = rel["targets"][0]
        src_id = rel["sources"][0]
        ya_trg2sources[trg_id].append(src_id)
    ta_src_trg = set()
    for rel in ta_relations:
        for src_id in rel["sources"]:
            for trg_id in rel["targets"]:
                ta_src_trg.add((src_id, trg_id))

    # collect for each S-node all source-anchor and target-anchor pairs
    already_checked: Dict[str, bool] = dict()
    for rel in ra_relations:
        rel_id = rel["relation"]
        # get all anchors (L-nodes) for S-source nodes
        i_source_multi_anchor_nodes = [
            get_l_anchor_nodes(src_id, ya_trg2sources) for src_id in rel["sources"]
        ]
        # get all anchors (L-nodes) for S-target nodes
        i_target_multi_anchor_nodes = [
            get_l_anchor_nodes(trg_id, ya_trg2sources) for trg_id in rel["targets"]
        ]
        if any(
            len(anchors) == 0
            for anchors in i_source_multi_anchor_nodes + i_target_multi_anchor_nodes
        ):
            logger.warning(
                f"nodeset={nodeset_id}: Could not find anchor node for any argument of the RA-node {rel_id}!"
            )
            continue

        # we check all combinations of source and target anchors for all anchors per argument
        for i_source_multi_anchor in i_source_multi_anchor_nodes:
            for i_source_anchor in i_source_multi_anchor:
                for i_target_multi_anchor in i_target_multi_anchor_nodes:
                    for i_target_anchor in i_target_multi_anchor:
                        # keep only pairs in s_node2source_target_pairs that appear in binary TA-relations
                        if (i_source_anchor, i_target_anchor) in ta_src_trg:
                            if rel_id in already_checked and not already_checked[rel_id]:
                                raise ValueError(f"direction of RA-node {rel_id} is ambiguous!")
                            already_checked[rel_id] = True
                            yield rel
                        elif (i_target_anchor, i_source_anchor) in ta_src_trg:
                            if rel_id in already_checked and already_checked[rel_id]:
                                raise ValueError(f"direction of RA-node {rel_id} is ambiguous!")
                            already_checked[rel_id] = False
                        # else:
                        #    raise ValueError(
                        #        f"nodeset={nodeset_id}: Could not find TA-relation for RA-node {rel_id}!"
                        #    )

    if verbose:
        missing = []
        for rel in ra_relations:
            rel_id = rel["relation"]
            if rel_id not in already_checked:
                missing.append(rel_id)
        if len(missing) > 0:
            raise ValueError(
                f"nodeset={nodeset_id}: could not determine direction of RA-nodes {missing} "
                f"because there is no TA relation between any combination of anchoring I-nodes!"
            )


def prepare_nodeset(
    nodeset: Nodeset,
    nodeset_id: str,
    s_node_text: str,
    ya_node_text: str,
    s_node_type: str = "S",
    l2i_similarity_measure: str = "lcsstr",
    add_gold_data: bool = False,
    re_revert_ra_relations: bool = False,
    re_remove_none_relations: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> Nodeset:
    """Prepare the nodeset for further processing:
    1. Clean up the nodeset by removing isolated nodes and invalid transitions.
    2. Remove S- and YA-nodes with edges.
    3. Add dummy S- and YA-nodes with edges by matching L- and I-nodes based on the similarity measure.
    4. Optionally, add cleaned gold data (from output of step 1):
        a. Normalize the direction of the RA-relation nodes.
        b. Update the text and type of the result relation nodes with matching gold data.
        c. Add remaining nodes and edges from the gold data that were not matched.

    Args:
        nodeset: A Nodeset.
        s_node_text: Text for the dummy S-node.
        ya_node_text: Text for the dummy YA-node.
        s_node_type: Type for the dummy S-node.
        l2i_similarity_measure: Similarity measure to use for matching L- and I-nodes.
        add_gold_data: Whether to add gold data to the dummy relation nodes.
        re_revert_ra_relations: Whether to revert the normalized RA-relation nodes back to the original state.
        re_remove_none_relations: Whether to re-remove the new S and YA relations.
        nodeset_id: A Nodeset ID for better error messages.
        verbose: Whether to show verbose output.
        debug: Whether to execute in debug mode.

    Returns:
        Prepared nodeset.
    """

    nodeset_clean = cleanup_nodeset(nodeset=nodeset, nodeset_id=nodeset_id, verbose=verbose)
    nodeset_without_relations = remove_s_and_ya_nodes_with_edges(nodeset_clean, nodeset_id)
    nodeset_with_dummy_relations = add_s_and_ya_nodes_with_edges(
        nodeset=nodeset_without_relations,
        nodeset_id=nodeset_id,
        s_node_text=s_node_text,
        ya_node_text=ya_node_text,
        s_node_type=s_node_type,
        similarity_measure=l2i_similarity_measure,
        verbose=verbose,
    )

    if add_gold_data:
        # normalize the direction of the RA-relation nodes in the cleaned (gold) data
        nodeset_normalized_relations = normalize_ra_relation_direction(nodeset_clean, nodeset_id)
        # match the dummy relation nodes with the gold data
        dummy2gold_node_matching = get_node_matching(
            nodeset=nodeset_with_dummy_relations,
            other=nodeset_normalized_relations,
            nodeset_id=nodeset_id,
            verbose=verbose,
        )
        # update the text and type of the dummy relation nodes with the gold data and
        nodeset_with_dummy_relations = merge_other_into_nodeset(
            nodeset=nodeset_with_dummy_relations,
            other=nodeset_normalized_relations,
            node_matching=dummy2gold_node_matching,
            id_suffix_other="-gold",
            nodeset_id=nodeset_id,
            verbose=verbose,
            debug=debug,
            add_nodes_from_other=False,
        )

        if re_revert_ra_relations:
            node_id2node = get_id2node(nodeset_with_dummy_relations)
            normalized_re_relations = [
                ra_relation
                for ra_relation in get_relations(
                    nodeset_with_dummy_relations, "RA", enforce_cardinality=True
                )
                if node_id2node[ra_relation["relation"]]["text"].endswith("-rev")
            ]
            nodeset_with_dummy_relations = reverse_relations_nodes(
                relations=normalized_re_relations,
                nodeset=nodeset_with_dummy_relations,
                nodeset_id=nodeset_id,
                reversed_text_suffix="-rev",
                redo=True,
            )
        if re_remove_none_relations:
            node_id2node = get_id2node(nodeset_with_dummy_relations)
            # collect S and YA relations
            s_relations = list(
                get_relations(nodeset=nodeset_with_dummy_relations, relation_type="S")
            )
            new_s_relations = [
                rel for rel in s_relations if node_id2node[rel["relation"]]["text"] == s_node_text
            ]
            ya_relations = list(
                get_relations(nodeset=nodeset_with_dummy_relations, relation_type="YA")
            )
            new_ya_relations = [
                rel
                for rel in ya_relations
                if node_id2node[rel["relation"]]["text"] == ya_node_text
            ]
            nodeset_with_dummy_relations = remove_relation_nodes_and_edges(
                nodeset=nodeset_with_dummy_relations, relations=new_s_relations + new_ya_relations
            )
    else:
        if re_revert_ra_relations:
            logger.warning(
                "re_revert_ra_relations=True is only supported in combination with add_gold_data=True."
            )
        if re_remove_none_relations:
            logger.warning(
                "re_remove_none_relations=True is only supported in combination with add_gold_data=True."
            )
    return nodeset_with_dummy_relations


def main(
    input_dir: str,
    output_dir: str,
    show_progress: bool = True,
    nodeset_id: Optional[str] = None,
    nodeset_blacklist: Optional[List[str]] = None,
    **kwargs,
):
    # create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = prepare_nodeset(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            **kwargs,
        )
        write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=prepare_nodeset,
            nodeset_dir=input_dir,
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
        "--s_node_type",
        type=str,
        default="RA",
        help="The type of the new S nodes. Default is 'RA'.",
    )
    parser.add_argument(
        "--s_node_text",
        type=str,
        default="NONE",
        help="The text of the new S nodes. Default is 'DUMMY'.",
    )
    parser.add_argument(
        "--ya_node_text",
        type=str,
        default="NONE",
        help="The text of the new YA nodes. Default is 'DUMMY'.",
    )
    parser.add_argument(
        "--add_gold_data",
        action="store_true",
        help="Whether to add gold data to the dummy relation nodes and missing nodes and edges.",
    )
    parser.add_argument(
        "--re_revert_ra_relations",
        action="store_true",
        help="Whether to revert the normalized RA-relation nodes back to the original state.",
    )
    parser.add_argument(
        "--re_remove_none_relations",
        action="store_true",
        help="Whether to re-remove the new S and YA relations.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to execute in debug mode.",
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
