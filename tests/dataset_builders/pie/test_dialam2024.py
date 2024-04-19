import json

import datasets
import pytest
from pie_datasets import load_dataset
from pytorch_ie.annotations import LabeledSpan

from dataset_builders.pie.dialam2024.dialam2024 import PieDialAM2024
from src.utils.nodeset2document import SimplifiedDialAM2024Document

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
    assert len(hf_dataset["train"]) == 1381


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


@pytest.fixture(scope="module")
def builder():
    return PieDialAM2024()


def test_convert_document(builder, hf_example):
    document = builder._generate_document(hf_example)
    assert isinstance(document, SimplifiedDialAM2024Document)
    assert document.id == "17918"
    assert len(document.l_nodes) > 0
    fist_l_node = document.l_nodes[0]
    assert isinstance(fist_l_node, LabeledSpan)
    assert (
        str(fist_l_node)
        == "Claire Cooper : Even if some children do get back to school before the end of the summer term, their experience is more likely to be about social distancing and hygiene rather than anything of any educational value"
    )


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("dataset_builders/pie/dialam2024", data_dir=DATA_DIR)


def test_dataset(dataset):
    assert set(dataset) == {"train"}
    assert len(dataset["train"]) == 1381


@pytest.fixture(scope="module")
def document(dataset):
    return dataset["train"][0]


def test_document(document):
    assert isinstance(document, SimplifiedDialAM2024Document)
    assert document.id == "23156"
    assert len(document.l_nodes) > 0
    fist_l_node = document.l_nodes[0]
    assert isinstance(fist_l_node, LabeledSpan)
    assert (
        str(fist_l_node)
        == "Kate Forbes : I don't expect everybody on the panel to agree with me on independence"
    )
