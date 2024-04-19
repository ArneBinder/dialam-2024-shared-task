import pytest
from pie_datasets import load_dataset
from pytorch_ie.annotations import LabeledSpan

from src.utils.nodeset2document import SimplifiedQT30Document


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(
        "dataset_builders/pie/qt30",
        # to use local data:
        # data_dir="data/dataset",
    )


def test_dataset(dataset):
    assert set(dataset) == {"train"}
    assert len(dataset["train"]) == 1381


@pytest.fixture(scope="module")
def document(dataset):
    return dataset["train"][0]


def test_document(document):
    assert isinstance(document, SimplifiedQT30Document)
    assert len(document.l_nodes) > 0
    fist_l_node = document.l_nodes[0]
    assert isinstance(fist_l_node, LabeledSpan)
    # assert str(fist_l_node) == ...
