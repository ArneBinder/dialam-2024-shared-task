import logging
from typing import List

import datasets
from pie_datasets import GeneratorBasedBuilder
from pytorch_ie.annotations import LabeledSpan, NaryRelation

from src.document.types import SimplifiedDialAM2024Document
from src.utils.nodeset_utils import (
    Nodeset,
    get_id2node,
    get_node_ids_by_type,
    get_relations,
    sort_nodes_by_hierarchy,
)
from src.utils.prepare_data import prepare_nodeset

logger = logging.getLogger(__name__)


def dictoflists_to_listofdicts(data):
    return [dict(zip(data, t)) for t in zip(*data.values())]


def convert_to_document(
    nodeset: Nodeset, nodeset_id: str, text_mode: str = "l-nodes", text_sep: str = " "
) -> SimplifiedDialAM2024Document:

    # 1. create document text and L-node-spans
    l_node_ids = get_node_ids_by_type(nodeset, node_types=["L"])
    sorted_l_node_ids: List[str] = sort_nodes_by_hierarchy(
        l_node_ids, edges=nodeset["edges"], nodeset_id=nodeset_id
    )
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

    doc = SimplifiedDialAM2024Document(text=text, id=nodeset_id)
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
            label=ya_12l_relation_node["text"],
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
            label=s_relation_node["text"],
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
            label=ya_s2ta_relation_node["text"],
        )
        doc.ya_s2ta_nodes.append(ya_s2ta_nary_relation)
        doc.metadata["ya_s2ta_relations"].append(ya_s2ta_relation)

    i_nodes = get_node_ids_by_type(nodeset, node_types=["I"])
    doc.metadata["i_node_ids"] = i_nodes
    ta_nodes = get_node_ids_by_type(nodeset, node_types=["TA"])
    doc.metadata["ta_node_ids"] = ta_nodes

    # validate_document(nodeset=nodeset, document=doc)

    return doc


class PieDialAM2024(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SimplifiedDialAM2024Document

    # BASE_DATASET_PATH = "ArneBinder/dialam2024"
    # BASE_DATASET_REVISION = "dc0c50174e0f1d4d697be5d017c390054aa8aa9d"
    BASE_DATASET_PATH = "dataset_builders/hf/dialam2024"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dialam2024",
            version=datasets.Version("1.0.0"),
            description="PIE-wrapped DialAM-2024 dataset",
        ),
    ]

    def _generate_document(self, example, **kwargs):
        nodeset_id = example["id"]
        # because of the tabular format that backs HuggingFace datasets, the sequential data fields, like
        # nodes / edges / locutions, are stored as dict of lists, where each key is a field name and the value
        # is a list of values for that field; however, the nodeset2document conversion function expects the
        # data to be stored as list of dicts, where each dict is a record; therefore, before converting the
        # nodeset to a document, we need to convert the nodes / edges / locutions back to lists of dicts
        nodeset = {
            "nodes": dictoflists_to_listofdicts(example["nodes"]),
            "edges": dictoflists_to_listofdicts(example["edges"]),
            "locutions": dictoflists_to_listofdicts(example["locutions"]),
        }
        cleaned_nodeset = prepare_nodeset(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            s_node_text="NONE",
            ya_node_text="NONE",
            s_node_type="RA",
            l2i_similarity_measure="lcsstr",
            add_gold_data=True,
        )
        doc = convert_to_document(nodeset=cleaned_nodeset, nodeset_id=nodeset_id)
        return doc
