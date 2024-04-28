import json
from typing import Dict, Union

import datasets
import pytest
from pie_datasets import load_dataset
from pytorch_ie import Annotation
from pytorch_ie.annotations import LabeledSpan, NaryRelation

from dataset_builders.pie.dialam2024.dialam2024 import PieDialAM2024
from src.document.types import (
    SimplifiedDialAM2024Document,
    TextDocumentWithLabeledEntitiesAndNaryRelations,
)

DATA_DIR = None
# To use local data, set DATA_DIR to a local directory containing the nodeset files.
# But note that the blacklist is applied nevertheless!
# (change it in dataset_builder/hf/dialam2024/dialam2024.py file, if necessary)
# DATA_DIR = "data/dataset_excerpt"

SPLIT_SIZES = {"train": 1399, "sample_test": 3, "test": 11}
# for the test set, we want to have the map "test_map10" as example because it contains complex stuff
SPLIT2IDX = {"train": 0, "test": 1}


@pytest.fixture(scope="module")
def hf_dataset():
    ds = datasets.load_dataset(PieDialAM2024.BASE_DATASET_PATH, data_dir=DATA_DIR)
    return ds


def test_hf_dataset(hf_dataset):
    split_sizes = {split: len(hf_dataset[split]) for split in hf_dataset}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module", params=["train", "test"])
def split_name(request):
    return request.param


@pytest.fixture(scope="module")
def hf_example(hf_dataset, split_name):
    return hf_dataset[split_name][SPLIT2IDX[split_name]]


def test_hf_example(hf_example, split_name):
    fixture_data_path = (
        f"tests/fixtures/dataset_builders/pie/dialam2024/{split_name}/{hf_example['id']}.json"
    )
    expected_data = json.load(open(fixture_data_path, "r"))
    assert hf_example == expected_data


@pytest.fixture(scope="module", params=[config.name for config in PieDialAM2024.BUILDER_CONFIGS])
def config_name(request):
    return request.param


@pytest.fixture(scope="module")
def builder(config_name):
    return PieDialAM2024(name=config_name)


NODE_IDS = {
    "17918": {
        "l_nodes": {
            "510959",
            "510963",
            "510968",
            "510973",
            "510980",
            "510985",
            "510992",
            "510996",
            "511000",
            "511005",
            "511010",
            "511015",
            "511022",
            "511029",
            "511034",
            "511041",
            "511046",
            "511051",
        },
        "s_nodes": {
            "511063",
            "511066",
            "511068",
            "511075",
            "511060",
            "511065",
            "511069",
            "511074",
            "511061",
            "511067",
            "511070",
            "511059",
            "511073",
            "511062",
            "511072",
            "511071",
            "511064",
        },
        "ya_i2l_nodes": {
            "511084",
            "511083",
            "511088",
            "511081",
            "511092",
            "511087",
            "511093",
            "511091",
            "511077",
            "511079",
            "511089",
            "511078",
            "511086",
            "511076",
            "511082",
            "511090",
            "511085",
            "511080",
        },
        "ya_s2ta_nodes": {
            "511101",
            "511098",
            "511107",
            "511099",
            "511096",
            "511110",
            "511105",
            "511108",
            "511104",
            "511100",
            "511103",
            "511095",
            "511102",
            "511094",
            "511097",
            "511109",
            "511106",
        },
    },
    "test_map10": {
        "l_nodes": {
            "13_168141153402989746",
            "18_168141153402989746",
            "23_168141153402989746",
            "28_168141153402989746",
            "38_168141153402989746",
            "3_168141153402989746",
            "43_168141153402989746",
            "48_168141153402989746",
            "83_168141153402989746",
            "88_168141153402989746",
            "8_168141153402989746",
        },
        "s_nodes": {
            "113168141153402989747",
            "113168141153402989748",
            "113168141153402989749",
            "113168141153402989750",
            "113168141153402989751",
            "113168141153402989752",
            "113168141153402989753",
            "113168141153402989754",
            "113168141153402989755",
            "113168141153402989756",
            "113168141153402989757",
        },
        "ya_i2l_nodes": {
            "113168141153402989758",
            "113168141153402989759",
            "113168141153402989760",
            "113168141153402989761",
            "113168141153402989762",
            "113168141153402989763",
            "113168141153402989764",
            "113168141153402989765",
            "113168141153402989766",
            "113168141153402989767",
            "113168141153402989768",
        },
        "ya_s2ta_nodes": {
            "113168141153402989769",
            "113168141153402989770",
            "113168141153402989771",
            "113168141153402989772",
            "113168141153402989773",
            "113168141153402989774",
            "113168141153402989775",
            "113168141153402989776",
            "113168141153402989777",
            "113168141153402989778",
            "113168141153402989779",
        },
    },
}


def get_relations_from_merged_layer(
    document: TextDocumentWithLabeledEntitiesAndNaryRelations,
) -> Dict[str, Dict[str, NaryRelation]]:
    node_types = document.metadata["nary_relation_layers"]
    start_idx = 0
    node_ids2annotations = {}
    for node_type in node_types:
        relation_metadata_key = node_type.replace("_nodes", "_relations")
        relation_metadata = document.metadata[relation_metadata_key]
        relation_node_ids = [rel["relation"] for rel in relation_metadata]
        end_idx = start_idx + len(relation_metadata)
        relation_annotations = document.nary_relations[start_idx:end_idx]
        # sanity check
        assert len(relation_node_ids) == len(relation_annotations)
        node_ids2annotations[node_type] = {
            node_id: ann for node_id, ann in zip(relation_node_ids, relation_annotations)
        }
        start_idx = end_idx
    return node_ids2annotations


def get_relations_from_separate_layers(
    document: SimplifiedDialAM2024Document,
) -> Dict[str, Dict[str, NaryRelation]]:
    node_ids2annotations = {}
    for node_type in ["ya_i2l_nodes", "ya_s2ta_nodes", "s_nodes"]:
        relation_metadata_key = node_type.replace("_nodes", "_relations")
        relation_metadata = document.metadata[relation_metadata_key]
        relation_node_ids = [rel["relation"] for rel in relation_metadata]
        relation_annotations = getattr(document, node_type)
        # sanity check
        assert len(relation_node_ids) == len(relation_annotations)
        node_ids2annotations[node_type] = {
            node_id: ann for node_id, ann in zip(relation_node_ids, relation_annotations)
        }
    return node_ids2annotations


def get_relations(
    document: Union[SimplifiedDialAM2024Document, TextDocumentWithLabeledEntitiesAndNaryRelations]
) -> Dict[str, Dict[str, NaryRelation]]:
    if isinstance(document, SimplifiedDialAM2024Document):
        return get_relations_from_separate_layers(document)
    elif isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations):
        return get_relations_from_merged_layer(document)
    else:
        raise ValueError(f"Unknown document type {type(document)}")


def get_labeled_spans(
    document: Union[SimplifiedDialAM2024Document, TextDocumentWithLabeledEntitiesAndNaryRelations]
) -> Dict[str, LabeledSpan]:
    if isinstance(document, SimplifiedDialAM2024Document):
        return {
            node_id: ann for node_id, ann in zip(document.metadata["l_node_ids"], document.l_nodes)
        }
    elif isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations):
        return {
            node_id: ann
            for node_id, ann in zip(document.metadata["l_node_ids"], document.labeled_spans)
        }
    else:
        raise ValueError(f"Unknown document type {type(document)}")


def assert_document(document, config_name, split_name):
    node_ids2annotations: Dict[str, Dict[str, Annotation]] = {
        "l_nodes": get_labeled_spans(document),
    }
    relation_node_ids2annotations = get_relations(document)
    node_ids2annotations.update(relation_node_ids2annotations)

    if split_name == "train":
        if config_name == "default":
            assert isinstance(document, SimplifiedDialAM2024Document)
            assert document.id == "17918"
            last_s_node = node_ids2annotations["s_nodes"]["511063"]
            assert last_s_node.resolve() == (
                "Default Rephrase",
                (
                    ("source", ("L", "Andy Burnham : It is a mixed picture")),
                    ("target", ("L", "Fiona Bruce : What's happening where you are")),
                ),
            )
        elif config_name == "merged_relations":
            assert isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations)
            assert document.id == "17918"
            # s_node with id '511063' is the last nary relation
            last_s_node = node_ids2annotations["s_nodes"]["511063"]
            assert last_s_node.resolve() == (
                "s_nodes:Default Rephrase",
                (
                    ("s_nodes:source", ("L", "Andy Burnham : It is a mixed picture")),
                    ("s_nodes:target", ("L", "Fiona Bruce : What's happening where you are")),
                ),
            )
        else:
            raise ValueError(f"Unknown config name {config_name}")
        fist_l_node = node_ids2annotations["l_nodes"]["510959"]
        assert isinstance(fist_l_node, LabeledSpan)
        assert (
            str(fist_l_node)
            == "Claire Cooper : Even if some children do get back to school before the end of the summer term, "
            "their experience is more likely to be about social distancing and hygiene rather than anything "
            "of any educational value"
        )

    elif split_name == "test":
        if config_name == "default":
            assert isinstance(document, SimplifiedDialAM2024Document)
            assert document.id == "test_map10"
            last_s_node = node_ids2annotations["s_nodes"]["113168141153402989757"]
            assert last_s_node.resolve() == (
                "NONE",
                (
                    ("source", ("L", "David Davies: this is what’s in store for us")),
                    (
                        "target",
                        (
                            "L",
                            "David Davies: unless they’re amongst the extra Assembly members who’ll be "
                            "able to have a full slap-up English breakfast for £3.50 in the Senate",
                        ),
                    ),
                ),
            )
        elif config_name == "merged_relations":
            assert isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations)
            assert document.id == "test_map10"
            last_s_node = node_ids2annotations["s_nodes"]["113168141153402989757"]
            assert last_s_node.resolve() == (
                "s_nodes:NONE",
                (
                    ("s_nodes:source", ("L", "David Davies: this is what’s in store for us")),
                    (
                        "s_nodes:target",
                        (
                            "L",
                            "David Davies: unless they’re amongst the extra Assembly members who’ll be able "
                            "to have a full slap-up English breakfast for £3.50 in the Senate",
                        ),
                    ),
                ),
            )
        else:
            raise ValueError(f"Unknown config name {config_name}")
        fist_l_node = node_ids2annotations["l_nodes"]["23_168141153402989746"]
        assert isinstance(fist_l_node, LabeledSpan)
        assert (
            str(fist_l_node)
            == "David Davies: £100 million being spent on a whole load of extra Welsh Assembly members"
        )
    else:
        raise ValueError(f"Unknown split name {split_name}")

    for node_type, current_node_ids2annotations in node_ids2annotations.items():
        assert set(current_node_ids2annotations) == NODE_IDS[document.id][node_type]


def test_convert_document(builder, hf_example, split_name):
    # test nodeset cleanup and conversion to document
    document = builder._generate_document(hf_example)
    assert_document(document, config_name=builder.config.name, split_name=split_name)


@pytest.fixture(scope="module")
def dataset(config_name):
    return load_dataset("dataset_builders/pie/dialam2024", name=config_name, data_dir=DATA_DIR)


def test_dataset(dataset):
    split_sizes = {split: len(dataset[split]) for split in dataset}
    assert split_sizes == SPLIT_SIZES


@pytest.fixture(scope="module")
def document(dataset, split_name):
    return dataset[split_name][SPLIT2IDX[split_name]]


def test_document(document, config_name, split_name):
    assert_document(document, config_name, split_name)
