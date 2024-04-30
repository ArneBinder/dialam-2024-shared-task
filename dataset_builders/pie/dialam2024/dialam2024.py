import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import datasets
from pie_datasets import GeneratorBasedBuilder
from pytorch_ie import Document
from pytorch_ie.annotations import LabeledSpan, NaryRelation
from pytorch_ie.documents import TextBasedDocument

from src.document.types import (
    SimplifiedDialAM2024Document,
    TextDocumentWithLabeledEntitiesAndNaryRelations,
)
from src.utils.nodeset_utils import (
    Nodeset,
    get_id2node,
    get_node_ids_by_type,
    get_relations,
    sort_nodes_by_hierarchy,
)
from src.utils.prepare_data import prepare_nodeset

logger = logging.getLogger(__name__)

NONE_LABEL = "NONE"
PREFIX_SEPARATOR = ":"
REVERSE_SUFFIX = "-rev"


def dictoflists_to_listofdicts(data):
    return [dict(zip(data, t)) for t in zip(*data.values())]


def listofdicts_to_dictoflists(data):
    return {k: [t[k] for t in data] for k in data[0]}


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
            # avoid multiple spaces since they will be removed later by the tokenizer and will cause offset mismatch
            node_text = " ".join(l_node["text"].split())
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

    # add all original data
    doc.metadata["nodes"] = nodeset["nodes"]
    doc.metadata["edges"] = nodeset["edges"]
    doc.metadata["locutions"] = nodeset["locutions"]

    return doc


def get_roles_and_arguments(annotation: NaryRelation) -> Tuple[Tuple[str, LabeledSpan], ...]:
    return tuple(sorted(zip(annotation.roles, annotation.arguments)))


# NOTE: this does not exactly do the inverse of convert_to_document, because it:
# - produces the format expected by the DialAM2024 shared task evaluation script, but
#   not the HuggingFace examples, i.e. the nodes / edges / locutions are not stored in lists of dicts
# - re-creates edges, i.e. the ids of any original edges are not preserved
# - requires to set use_predictions to either True or False
def convert_to_example(
    document: SimplifiedDialAM2024Document,
    use_predictions: Union[bool, List[str]],
    denormalize_relation_direction: bool = True,
) -> Dict[str, Any]:

    edge_set = set((edge["fromID"], edge["toID"]) for edge in document.metadata["edges"])

    node_ids2original = {node["nodeID"]: node for node in document.metadata["nodes"]}

    i_node_ids = document.metadata["i_node_ids"]
    i_nodes = [node_ids2original[i_node_id] for i_node_id in i_node_ids]
    ta_node_ids = document.metadata["ta_node_ids"]
    ta_nodes = [node_ids2original[ta_node_id] for ta_node_id in ta_node_ids]
    l_node_ids = document.metadata["l_node_ids"]
    l_nodes = [node_ids2original[l_node_id] for l_node_id in l_node_ids]

    # NOTE: The order of the relation layers is important here because we check for all new relation
    #   nodes if their source and target nodes are already in the document and skip the relation node
    #   if not. Therefore, the order should be such that the source and target nodes are added before
    #   the respective relation nodes.
    rel_layer_names = ["ya_i2l_nodes", "s_nodes", "ya_s2ta_nodes"]

    annotation2predicted_label: Dict[str, Dict[NaryRelation, str]] = defaultdict(dict)
    if use_predictions:
        if isinstance(use_predictions, bool):
            prediction_layer_names = rel_layer_names
        elif isinstance(use_predictions, (list, tuple)):
            prediction_layer_names = use_predictions
        else:
            raise ValueError(
                f"Unsupported value for use_predictions: {use_predictions}. Expected bool or list of layer names."
            )
        for layer_name in prediction_layer_names:
            rel_args2annotation = {
                get_roles_and_arguments(rel): rel for rel in document[layer_name]
            }
            rel_args2predictions = {
                get_roles_and_arguments(rel): rel for rel in document[layer_name].predictions
            }
            for args, annotation in rel_args2annotation.items():
                if args in rel_args2predictions:
                    predicted_label = rel_args2predictions[args].label
                    annotation2predicted_label[layer_name][annotation] = predicted_label
                else:
                    annotation2predicted_label[layer_name][annotation] = NONE_LABEL

    nodes = i_nodes + ta_nodes + l_nodes
    node_ids = set(node["nodeID"] for node in nodes)
    # create ya_i2l_nodes / ya_s2ta_nodes / s_nodes from the nary relations
    for layer_name in rel_layer_names:
        metadata_key = layer_name.replace("_nodes", "_relations")
        relations = document.metadata[metadata_key]
        for annotation, relation in zip(document[layer_name], relations):
            label = annotation2predicted_label[layer_name].get(annotation, annotation.label)
            rel_node_id = relation["relation"]
            for src in relation["sources"]:
                edge_set.remove((src, rel_node_id))
            for tgt in relation["targets"]:
                edge_set.remove((rel_node_id, tgt))
            if label != NONE_LABEL:
                new_node = dict(node_ids2original[rel_node_id])
                # if the relation was normalized, re-reverse it
                if denormalize_relation_direction and label.endswith(REVERSE_SUFFIX):
                    label = label[: -len(REVERSE_SUFFIX)]
                    source_ids = relation["targets"]
                    target_ids = relation["sources"]
                else:
                    source_ids = relation["sources"]
                    target_ids = relation["targets"]
                new_node["text"] = label
                # check if all source and target nodes are in the document
                if not all(
                    source_or_target in node_ids for source_or_target in source_ids + target_ids
                ):
                    logger.warning(
                        f"doc={document.id}: Skipping relation node {relation} because not all source or target "
                        f"nodes are yet available."
                    )
                    continue
                nodes.append(new_node)
                node_ids.add(rel_node_id)
                for src in source_ids:
                    edge_set.add((src, rel_node_id))
                for tgt in target_ids:
                    edge_set.add((rel_node_id, tgt))

    edge_set_filtered = set(
        (src, tgt) for src, tgt in edge_set if src in node_ids and tgt in node_ids
    )
    edges = [
        {"edgeID": str(idx + 1), "fromID": src, "toID": tgt}
        for idx, (src, tgt) in enumerate(sorted(edge_set_filtered))
    ]
    result = {
        "id": document.id,
        "nodes": nodes,
        "edges": edges,
        "locutions": document.metadata["locutions"],  # TODO: better copy.deepcopy?
    }

    return result


def prefix_nary_relation(nary_relation: NaryRelation, prefix: str, sep: str = ":") -> NaryRelation:
    # prefix the label and roles of the nary relation
    new_label = f"{prefix}{sep}{nary_relation.label}"
    new_roles = tuple(f"{prefix}{sep}{role}" for role in nary_relation.roles)
    new_arguments = nary_relation.arguments
    return NaryRelation(arguments=new_arguments, roles=new_roles, label=new_label)


def unprefix_nary_relation(
    nary_relation: NaryRelation, sep: str = ":"
) -> Tuple[str, NaryRelation]:
    """Unprefixes the label and roles of a n-ary relation. The prefix is assumed to be the same for
    the label and all roles. If the roles have different prefixes, a ValueError is raised. If the
    label prefix does not match the role prefix, a warning is logged and the label is set to
    NONE_LABEL.

    Args:
        nary_relation: The n-ary relation to unprefix
        sep: The separator used to separate the prefix from the label and roles
    """
    new_roles = []
    role_prefixes = set()
    for role in nary_relation.roles:
        if sep not in role:
            raise ValueError(f'Role "{role}" does not contain separator "{sep}".')
        role_prefix, new_role = role.split(sep, maxsplit=1)
        role_prefixes.add(role_prefix)
        new_roles.append(new_role)
    if len(role_prefixes) != 1:
        raise ValueError(f"Roles have different prefixes: {role_prefixes}.")
    role_prefix = role_prefixes.pop()
    label_prefix, new_label = nary_relation.label.split(sep, maxsplit=1)
    if label_prefix != role_prefix:
        logger.warning(
            f'Label prefix "{label_prefix}" does not match role prefix "{role_prefix}". '
            f'Set label to NONE_LABEL="{NONE_LABEL}".'
        )
        new_label = NONE_LABEL
    return role_prefix, NaryRelation(
        arguments=nary_relation.arguments, roles=tuple(new_roles), label=new_label
    )


def merge_relations(
    document: TextBasedDocument,
    labeled_span_layer: str,
    nary_relation_layers: List[str],
    sep: str = ":",
) -> TextDocumentWithLabeledEntitiesAndNaryRelations:
    """Merges the relations from multiple n-ary relation layers into a single layer. The labels and
    roles of the n-ary relations are prefixed with the name of the layer.

    Note that this is destructive and will remove the original span and relation annotations
    from the input document!

    Args:
        document: The input document
        labeled_span_layer: The name of the layer containing the labeled spans
        nary_relation_layers: The names of the n-ary relation layers to merge
        sep: The separator used to separate the prefix from the label and roles

    Returns: A new document with the relation layers merged
    """
    new_document = TextDocumentWithLabeledEntitiesAndNaryRelations(
        text=document.text, id=document.id, metadata=document.metadata
    )
    # to allow reconstructing the original document, store the original labeled span layer name
    # and the n-ary relation layer names
    new_document.metadata["labeled_span_layer"] = labeled_span_layer
    new_document.metadata["nary_relation_layers"] = nary_relation_layers
    # get labeled spans and detach them from the document
    labeled_span = document[labeled_span_layer].clear()
    # add the labeled spans to the new document
    new_document.labeled_spans.extend(labeled_span)
    # iterate over the nary relation layers and prefix the labels and roles
    for nary_relation_layer in nary_relation_layers:
        if document[nary_relation_layer].target_name != labeled_span_layer:
            raise ValueError(
                f"Expected target name of nary relation layer {nary_relation_layer} to be "
                f"{labeled_span_layer}, got {document[nary_relation_layer].target_name}."
            )
        nary_relations = document[nary_relation_layer].clear()
        prefixed_nary_relations = [
            prefix_nary_relation(rel, prefix=nary_relation_layer, sep=sep)
            for rel in nary_relations
        ]
        new_document.nary_relations.extend(prefixed_nary_relations)

    return new_document


def unmerge_relations(
    document: TextDocumentWithLabeledEntitiesAndNaryRelations, sep: str = ":"
) -> SimplifiedDialAM2024Document:
    """Unmerges the relations from a single n-ary relation layer into multiple layers. The labels
    and roles of the n-ary relations are un-prefixed.

    Note that this is destructive and will remove the original span and relation annotations
    from the input document!

    Args:
        document: The input document
        sep: The separator used to separate the prefix from the label and roles

    Returns: A new document with the relation layers unmerged
    """
    new_document = SimplifiedDialAM2024Document(
        text=document.text, id=document.id, metadata=document.metadata
    )
    # get the labeled spans and detach them from the document
    labeled_spans = document.labeled_spans.clear()
    # add the labeled spans to the new document
    new_document.l_nodes.extend(labeled_spans)
    # iterate over the nary relations and un-prefix the labels and roles
    for nary_relation in document.nary_relations:
        prefix, new_nary_relation = unprefix_nary_relation(nary_relation, sep=sep)
        if prefix not in new_document.metadata["nary_relation_layers"]:
            raise ValueError(f"Unknown n-ary relation layer prefix {prefix}.")
        new_document[prefix].append(new_nary_relation)
    # also handle predictions
    for nary_relation in document.nary_relations.predictions:
        prefix, new_nary_relation = unprefix_nary_relation(nary_relation, sep=sep)
        if prefix not in new_document.metadata["nary_relation_layers"]:
            raise ValueError(f"Unknown n-ary relation layer prefix {prefix}.")
        new_document[prefix].predictions.append(new_nary_relation)

    return new_document


class PieDialAM2024(GeneratorBasedBuilder):
    # BASE_DATASET_PATH = "ArneBinder/dialam2024"
    # BASE_DATASET_REVISION = ???  # fill in the revision of the dataset
    BASE_DATASET_PATH = "dataset_builders/hf/dialam2024"
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="PIE-wrapped DialAM-2024 dataset",
        ),
        datasets.BuilderConfig(
            name="merged_relations",
            version=datasets.Version("1.0.0"),
            description="PIE-wrapped DialAM-2024 dataset with merged relations",
        ),
    ]
    DOCUMENT_TYPES = {
        "default": SimplifiedDialAM2024Document,
        "merged_relations": TextDocumentWithLabeledEntitiesAndNaryRelations,
    }
    # both configs use the same base config
    BASE_CONFIG_KWARGS_DICT = {
        "default": {"name": "default"},
        "merged_relations": {"name": "default"},
    }
    DEFAULT_CONFIG_NAME = "default"

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
        # this is a bit hacky, but we can use the nodeset_id to determine if the example is a test example
        is_test_example = nodeset_id.startswith("test_map")
        cleaned_nodeset = prepare_nodeset(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            s_node_text=NONE_LABEL,
            ya_node_text=NONE_LABEL,
            s_node_type="RA",
            reversed_text_suffix=REVERSE_SUFFIX,
            l2i_similarity_measure="lcsstr",
            add_gold_data=not is_test_example,
        )
        doc = convert_to_document(nodeset=cleaned_nodeset, nodeset_id=nodeset_id)
        if self.config.name == "merged_relations":
            doc = merge_relations(
                document=doc,
                labeled_span_layer="l_nodes",
                nary_relation_layers=["ya_i2l_nodes", "ya_s2ta_nodes", "s_nodes"],
                sep=PREFIX_SEPARATOR,
            )
        return doc

    def _generate_example(self, document: Document, **kwargs) -> Dict[str, Any]:
        if isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations):
            document = unmerge_relations(document, sep=PREFIX_SEPARATOR)
        elif not isinstance(document, SimplifiedDialAM2024Document):
            raise ValueError(f"Unsupported document type {type(document)}")
        converted = convert_to_example(document, use_predictions=False)
        # convert the nodes / edges / locutions back to dict of lists (see _generate_document)
        result = {k: listofdicts_to_dictoflists(v) for k, v in converted.items()}
        return result
