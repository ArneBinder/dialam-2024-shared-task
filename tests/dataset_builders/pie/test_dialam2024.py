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


@pytest.fixture(scope="module")
def hf_dataset():
    ds = datasets.load_dataset(PieDialAM2024.BASE_DATASET_PATH, data_dir=DATA_DIR)
    return ds


def test_hf_dataset(hf_dataset):
    assert set(hf_dataset) == {"train"}
    assert len(hf_dataset["train"]) == 1400


@pytest.fixture(scope="module")
def hf_example():
    ds = datasets.load_dataset(PieDialAM2024.BASE_DATASET_PATH, data_dir=DATA_DIR)
    return ds["train"][0]


def test_hf_example(hf_example):
    fixture_data_path = (
        f"tests/fixtures/dataset_builders/pie/dialam2024/nodeset{hf_example['id']}.json"
    )
    expected_data = json.load(open(fixture_data_path, "r"))
    assert hf_example == expected_data


@pytest.fixture(scope="module", params=[config.name for config in PieDialAM2024.BUILDER_CONFIGS])
def config_name(request):
    return request.param


@pytest.fixture(scope="module")
def builder(config_name):
    return PieDialAM2024(name=config_name)


def assert_document(document, config_name):
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


def test_convert_document(builder, hf_example):
    # test nodeset cleanup and conversion to document
    document = builder._generate_document(hf_example)
    assert_document(document, config_name=builder.config.name)


@pytest.fixture(scope="module")
def dataset(config_name):
    return load_dataset("dataset_builders/pie/dialam2024", name=config_name, data_dir=DATA_DIR)


def test_dataset(dataset):
    assert set(dataset) == {"train"}
    assert len(dataset["train"]) == 1400


@pytest.fixture(scope="module")
def document(dataset):
    return dataset["train"][0]


def test_document(document, config_name):
    assert_document(document, config_name)
