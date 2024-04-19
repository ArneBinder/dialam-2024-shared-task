import argparse
import dataclasses
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import datasets
import pyrootutils
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan, NaryRelation
from pytorch_ie.documents import TextBasedDocument

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

from src.utils.nodeset_utils import (
    Nodeset,
    get_id2node,
    get_node_ids_by_type,
    get_relations,
    process_all_nodesets,
    read_nodeset,
    sort_nodes_by_hierarchy,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SimplifiedQT30Document(TextBasedDocument):
    l_nodes: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    ya_i2l_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")
    ya_s2ta_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")
    s_nodes: AnnotationLayer[NaryRelation] = annotation_field(target="l_nodes")


def convert_to_document(
    nodeset: Nodeset, nodeset_id: str, text_mode: str = "l-nodes", text_sep: str = " "
) -> SimplifiedQT30Document:

    # 1. create document text and L-node-spans
    l_node_ids = get_node_ids_by_type(nodeset, node_types=["L"])
    sorted_l_node_ids: List[str] = sort_nodes_by_hierarchy(l_node_ids, edges=nodeset["edges"])
    node_id2node = get_id2node(nodeset)
    text = ""
    l_node_spans = dict()
    for l_node_id in sorted_l_node_ids:
        if text != "":
            text += text_sep
        l_node = node_id2node[l_node_id]
        if text_mode == "l-nodes":
            node_text = l_node["text"]
            l_node_spans[l_node_id] = LabeledSpan(
                start=len(text), end=len(text) + len(node_text), label=l_node["type"]
            )
        else:
            raise ValueError(f"Unsupported text mode: {text_mode}")
        text += node_text

    doc = SimplifiedQT30Document(text=text, id=nodeset_id)
    doc.l_nodes.extend([l_node_spans[node_id] for node_id in sorted_l_node_ids])
    doc.metadata["l_node_ids"] = sorted_l_node_ids

    # 2. encode YA relations between I and L nodes
    ya_i2l_relations = list(get_relations(nodeset, "YA-L2I", enforce_cardinality=True))
    doc.metadata["ya_i2l_relations"] = []
    for ya_12l_relation in ya_i2l_relations:
        ya_12l_relation_node = node_id2node[ya_12l_relation["relation"]]
        if len(ya_12l_relation["sources"]) != 1 or len(ya_12l_relation["targets"]) != 1:
            logger.warning(
                f"YA-relation {ya_12l_relation['relation']} has more than one source or target node!"
            )
            continue
        source_id = ya_12l_relation["sources"][0]
        source_span = l_node_spans[source_id]
        i_ya_nary_relation = NaryRelation(
            arguments=(source_span,),
            roles=("source",),
            label=f"YA-I2L:{ya_12l_relation_node['text']}",
        )
        doc.ya_i2l_nodes.append(i_ya_nary_relation)
        doc.metadata["ya_i2l_relations"].append(ya_12l_relation)

    # 3. encode S relations (between I nodes)
    # get anchor mapping from ya_i2l_relations
    i2l_ya_trg2sources = dict()
    for rel in ya_i2l_relations:
        if len(rel["targets"]) != 1 or len(rel["sources"]) != 1:
            logger.warning(
                f"YA-relation {rel['relation']} has more than one source or target node!"
            )
            continue
        trg_id = rel["targets"][0]
        src_id = rel["sources"][0]
        if trg_id in i2l_ya_trg2sources:
            logger.warning(f"YA-relation {rel['relation']} has multiple sources!")
            continue
        i2l_ya_trg2sources[trg_id] = src_id

    s_relations = list(get_relations(nodeset, "S", enforce_cardinality=True))
    doc.metadata["s_relations"] = []
    for s_relation in s_relations:
        s_relation_node = node_id2node[s_relation["relation"]]
        # get anchors
        source_anchor_ids = [i2l_ya_trg2sources[src_id] for src_id in s_relation["sources"]]
        target_anchor_ids = [i2l_ya_trg2sources[trg_id] for trg_id in s_relation["targets"]]
        # get spans
        source_spans = [l_node_spans[src_id] for src_id in source_anchor_ids]
        target_spans = [l_node_spans[trg_id] for trg_id in target_anchor_ids]
        # get roles
        source_roles = ["source"] * len(source_spans)
        target_roles = ["target"] * len(target_spans)
        # create nary relation
        s_nary_relation = NaryRelation(
            arguments=tuple(source_spans + target_spans),
            roles=tuple(source_roles + target_roles),
            label=f"S:{s_relation_node['text']}",
        )
        doc.s_nodes.append(s_nary_relation)
        doc.metadata["s_relations"].append(s_relation)

    # 4. encode YA relations between S and TA nodes
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))
    ta_id2relation = {rel["relation"]: rel for rel in ta_relations}
    s2ta_ya_relations = list(get_relations(nodeset, "YA-TA2S", enforce_cardinality=True))
    doc.metadata["ya_s2ta_relations"] = []
    for ya_s2ta_relation in s2ta_ya_relations:
        ya_s2ta_relation_node = node_id2node[ya_s2ta_relation["relation"]]
        # there should be exactly one source which is the TA relation node
        if len(ya_s2ta_relation["sources"]) != 1:
            logger.warning(
                f"YA-relation {ya_s2ta_relation['relation']} has more than one source node!"
            )
            continue
        src_id = ya_s2ta_relation["sources"][0]
        ta_relation = ta_id2relation.get(src_id)
        if ta_relation is None:
            # silent skip, because it is fine that an S-node is anchored in a none-TA node
            continue
        # get spans
        source_spans = [l_node_spans[src_id] for src_id in ta_relation["sources"]]
        target_spans = [l_node_spans[trg_id] for trg_id in ta_relation["targets"]]
        # get roles
        source_roles = ["source"] * len(source_spans)
        target_roles = ["target"] * len(target_spans)

        ya_s2ta_nary_relation = NaryRelation(
            arguments=tuple(source_spans + target_spans),
            roles=tuple(source_roles + target_roles),
            label=f"YA-S2TA:{ya_s2ta_relation_node['text']}",
        )
        doc.ya_s2ta_nodes.append(ya_s2ta_nary_relation)
        doc.metadata["ya_s2ta_relations"].append(ya_s2ta_relation)

    i_nodes = get_node_ids_by_type(nodeset, node_types=["I"])
    doc.metadata["i_node_ids"] = i_nodes
    ta_nodes = get_node_ids_by_type(nodeset, node_types=["TA"])
    doc.metadata["ta_node_ids"] = ta_nodes

    validate_document(nodeset=nodeset, document=doc)

    return doc


def validate_document(nodeset: Nodeset, document: SimplifiedQT30Document):
    metadata = document.metadata
    original = nodeset.copy()
    # check that we have the same L-nodes as in the original nodeset
    orig_l_node_id2text = dict()
    for n in original["nodes"]:
        if n["type"] == "L":
            orig_l_node_id2text[n["nodeID"]] = n["text"]
    l_node_ids = metadata["l_node_ids"]
    l_node_spans = {l_node_id: l_span for l_node_id, l_span in zip(l_node_ids, document.l_nodes)}
    converted_l_node_id2text = dict()
    for l_node_id in l_node_ids:
        l_span = l_node_spans[l_node_id]
        converted_l_node_id2text[l_node_id] = document.text[l_span.start : l_span.end]
    for l_id in orig_l_node_id2text:
        if l_id in converted_l_node_id2text:
            assert converted_l_node_id2text[l_id] == orig_l_node_id2text[l_id]
        else:
            raise ValueError(f"L-node missing in the converted document, nodeID: {l_id}")
    # check that we have the same TA-nodes as in the original nodeset
    ta_node_ids = metadata["ta_node_ids"]
    orig_ta_node_ids = [n["nodeID"] for n in original["nodes"] if n["type"] == "TA"]
    ta_node_diff = set(orig_ta_node_ids) - set(ta_node_ids)
    if len(ta_node_diff) != 0:
        raise ValueError(f"TA nodes missing in the converted document: {ta_node_diff}")

    # helper structures
    orig_src2trg = defaultdict(list)
    orig_trg2src = defaultdict(list)
    for e in original["edges"]:
        orig_src2trg[e["fromID"]].append(e["toID"])
    for e in original["edges"]:
        orig_trg2src[e["toID"]].append(e["fromID"])
    node_id2type = {n["nodeID"]: n["type"] for n in original["nodes"]}

    orig_mappings = {
        "orig_src2trg": orig_src2trg,
        "orig_trg2src": orig_trg2src,
        "node_id2type": node_id2type,
    }

    # check YA I2L relations are the same as in the original nodeset
    ya_i2l_relations = metadata["ya_i2l_relations"]
    allowed_types = {
        "allowed_node_types": ["YA"],
        "allowed_source_types": ["L"],
        "allowed_target_types": ["I"],
    }
    check_relations(
        original=original,
        relations=ya_i2l_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )
    # check YA S2TA relations are the same as in the original nodeset
    ya_s2ta_relations = metadata["ya_s2ta_relations"]
    allowed_types = {
        "allowed_node_types": ["YA"],
        "allowed_source_types": ["TA"],
        "allowed_target_types": ["S"],
    }
    check_relations(
        original=original,
        relations=ya_s2ta_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )
    # check S relations are the same as in the original nodeset
    s_relations = metadata["s_relations"]
    allowed_types = {
        "allowed_node_types": ["RA", "MA", "CA"],
        "allowed_source_types": ["I"],
        "allowed_target_types": ["I"],
    }
    check_relations(
        original=original,
        relations=s_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )


def check_relations(
    original: Nodeset,
    relations: List[Dict[str, Any]],
    orig_mappings: Dict[str, Any],
    allowed_types: Dict[str, List[str]],
):
    orig_src2trg = orig_mappings["orig_src2trg"]
    orig_trg2src = orig_mappings["orig_trg2src"]
    node_id2type = orig_mappings["node_id2type"]
    allowed_node_types = allowed_types["allowed_node_types"]
    allowed_source_types = allowed_types["allowed_source_types"]
    allowed_target_types = allowed_types["allowed_target_types"]
    # filter out only those node IDs where sources and targets have the correct (allowed) types
    orig_node_ids = []
    for n in original["nodes"]:
        if n["type"] in allowed_node_types:
            n_id = n["nodeID"]
            n_sources = orig_src2trg[n_id]
            n_targets = orig_trg2src[n_id]
            n_sources_filtered = [
                n_s for n_s in n_sources if node_id2type[n_s] in allowed_source_types
            ]
            n_targets_filtered = [
                n_t for n_t in n_targets if node_id2type[n_t] in allowed_target_types
            ]
            if len(n_sources_filtered) > 0 and len(n_targets_filtered) > 0:
                orig_node_ids.append(n_id)
    node_src_trg = dict()
    for rel in relations:
        node_src_trg[rel["relation"]] = {"sources": rel["sources"], "targets": rel["targets"]}
    # check for each relation whether we are missing the original source or target nodes
    for n_id in orig_node_ids:
        if not (n_id in node_src_trg):
            raise ValueError(
                f"{node_id2type[n_id]} node is not in the converted document, nodeID: {n_id}"
            )
        orig_sources = orig_trg2src[n_id]
        for orig_source in orig_sources:
            if node_id2type[orig_source] in allowed_source_types and not (
                orig_source in node_src_trg[n_id]["sources"]
            ):
                raise ValueError(f"Source is missing, nodeID: {orig_source} for nodeID: {n_id}")
        orig_targets = orig_src2trg[n_id]
        for orig_target in orig_targets:
            if node_id2type[orig_target] in allowed_target_types and not (
                orig_target in node_src_trg[n_id]["targets"]
            ):
                raise ValueError(f"Target is missing, nodeID: {orig_target} for nodeID: {n_id}")


def main(
    input_dir: str,
    show_progress: bool = True,
    verbose: bool = True,
    nodeset_id: Optional[str] = None,
    nodeset_blacklist: Optional[List[str]] = None,
    **kwargs,
):
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = convert_to_document(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            **kwargs,
        )
        # write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
        # result.asdict()
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=convert_to_document,
            nodeset_dir=input_dir,
            show_progress=show_progress,
            nodeset_blacklist=nodeset_blacklist,
            **kwargs,
        ):
            if isinstance(result_or_error, Exception):
                logger.error(f"nodeset={nodeset_id}: Failed to process: {result_or_error}")
            else:
                # write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result_or_error)
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_dir", type=str, required=True, help="The input directory containing the nodesets."
    )
    parser.add_argument(
        "--nodeset_id",
        type=str,
        default=None,
        help="The ID of the nodeset to process. If not provided, all nodesets in the input directory will be processed.",
    )
    parser.add_argument(
        "--nodeset_blacklist",
        # split by comma and remove leading/trailing whitespaces
        type=lambda x: [nid.strip() for nid in x.split(",")] if x else None,
        default=None,
        help="List of nodeset IDs that should be ignored.",
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
