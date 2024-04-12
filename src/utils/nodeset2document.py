import dataclasses

import argparse
import logging
import os
from typing import List, Optional
import pyrootutils
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan, NaryRelation
from pytorch_ie.documents import (
    TextBasedDocument,
)

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
    sorted_l_nodes: List[str] = sort_nodes_by_hierarchy(l_node_ids, edges=nodeset["edges"])
    node_id2node = get_id2node(nodeset)
    text = ""
    l_node_spans = dict()
    for l_node_id in sorted_l_nodes:
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
    doc.l_nodes.extend(l_node_spans.values())

    # 2. encode YA relations between I and L nodes
    ya_i2l_relations = list(get_relations(nodeset, "YA1", enforce_cardinality=True))
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

    # 4. encode YA relations between S and TA nodes
    ta_relations = list(get_relations(nodeset, "TA", enforce_cardinality=True))
    ta_id2relation = {rel["relation"]: rel for rel in ta_relations}
    s2ta_ya_relations = list(get_relations(nodeset, "YA2", enforce_cardinality=True))
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

    return doc


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
        result = convert_to_document(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            **kwargs,
        )
        #write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
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
                #write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result_or_error)
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawTextHelpFormatter
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
    parser.add_argument(
        "--silent",
        dest="verbose",
        action="store_false",
        help="Whether to show verbose output.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
