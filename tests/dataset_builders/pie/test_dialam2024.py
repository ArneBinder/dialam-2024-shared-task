import json

import datasets
import pytest
from pie_datasets import load_dataset
from pytorch_ie.annotations import LabeledSpan

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

SPLIT_SIZES = {"train": 1399, "test": 11}
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


def assert_document(document, config_name, split_name):
    if split_name == "train":
        if config_name == "default":
            assert isinstance(document, SimplifiedDialAM2024Document)
            assert document.id == "17918"
            assert len(document.l_nodes) == 18
            fist_l_node = document.l_nodes[0]
            assert len(document.ya_i2l_nodes) == 18
            assert len(document.ya_s2ta_nodes) == 17
            assert len(document.s_nodes) == 17
            last_s_node = document.s_nodes[-1]
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
            assert len(document.labeled_spans) == 18
            fist_l_node = document.labeled_spans[0]
            assert len(document.nary_relations) == 52
            last_nary_relation = document.nary_relations[-1]
            assert last_nary_relation.resolve() == (
                "s_nodes:Default Rephrase",
                (
                    ("s_nodes:source", ("L", "Andy Burnham : It is a mixed picture")),
                    ("s_nodes:target", ("L", "Fiona Bruce : What's happening where you are")),
                ),
            )
        else:
            raise ValueError(f"Unknown config name {config_name}")

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
            assert len(document.l_nodes) == 11
            # TODO: it looks like the order of the nodes is not deterministic. Does this matter?
            #   At least, the tests should be fixed to not rely on the order.
            assert document.metadata["l_node_ids"] == [
                "28_168141153402989746",
                "18_168141153402989746",
                "13_168141153402989746",
                "23_168141153402989746",
                "3_168141153402989746",
                "8_168141153402989746",
                "38_168141153402989746",
                "83_168141153402989746",
                "88_168141153402989746",
                "43_168141153402989746",
                "48_168141153402989746",
            ]
            fist_l_node = document.l_nodes[0]
            assert len(document.ya_i2l_nodes) == 11
            assert len(document.ya_s2ta_nodes) == 11
            assert len(document.s_nodes) == 11
            last_s_node = document.s_nodes[-1]
            assert last_s_node.resolve() == (
                "NONE",
                (
                    ("source", ("L", "David Davies: this is what’s in store for us")),
                    (
                        "target",
                        (
                            "L",
                            "David Davies: unless they’re amongst the extra Assembly members who’ll be able to have a full slap-up English breakfast for £3.50 in the Senate",
                        ),
                    ),
                ),
            )
        elif config_name == "merged_relations":
            assert isinstance(document, TextDocumentWithLabeledEntitiesAndNaryRelations)
            assert document.id == "test_map10"
            assert len(document.labeled_spans) == 11
            # TODO: it looks like the order of the nodes is not deterministic. Does this matter?
            #   At least, the tests should be fixed to not rely on the order.
            assert document.metadata["l_node_ids"] == [
                "18_168141153402989746",
                "28_168141153402989746",
                "8_168141153402989746",
                "3_168141153402989746",
                "23_168141153402989746",
                "13_168141153402989746",
                "38_168141153402989746",
                "83_168141153402989746",
                "88_168141153402989746",
                "43_168141153402989746",
                "48_168141153402989746",
            ]
            fist_l_node = document.labeled_spans[0]
            assert len(document.nary_relations) == 33
            last_nary_relation = document.nary_relations[-1]
            assert last_nary_relation.resolve() == (
                "s_nodes:NONE",
                (
                    ("s_nodes:source", ("L", "David Davies: this is what’s in store for us")),
                    (
                        "s_nodes:target",
                        (
                            "L",
                            "David Davies: unless they’re amongst the extra Assembly members who’ll be able to have a full slap-up English breakfast for £3.50 in the Senate",
                        ),
                    ),
                ),
            )
        else:
            raise ValueError(f"Unknown config name {config_name}")
        assert isinstance(fist_l_node, LabeledSpan)
        assert (
            str(fist_l_node)
            == "David Davies: £100 million being spent on a whole load of extra Welsh Assembly members"
        )
    else:
        raise ValueError(f"Unknown split name {split_name}")


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
