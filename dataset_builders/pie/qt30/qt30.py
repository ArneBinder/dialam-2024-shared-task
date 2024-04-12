import json
import logging
import os

import datasets
from pie_datasets import GeneratorBasedBuilder

from src.utils.prepare_data import SimplifiedQT30Document, convert_to_document

logger = logging.getLogger(__name__)


class SimplifiedQT30(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SimplifiedQT30Document

    BASE_DATASET_PATH = None
    BASE_DATASET_REVISION = None

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="qt30", version=datasets.Version("1.0.0"), description="QT30 dataset"
        ),
    ]

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_dir}
            ),
        ]

    def _generate_examples(self, filepath):
        # get all json files in the directory
        for file_name in os.listdir(filepath):
            if file_name.endswith(".json"):
                # get the nodeset_id from the file_name name, e.g. nodeset24234.json
                if not file_name.startswith("nodeset"):
                    raise ValueError(
                        f"Expected file name to start with 'nodeset', got {file_name}"
                    )
                nodeset_id = file_name[len("nodeset") : -len(".json")]
                with open(os.path.join(filepath, file_name), "r") as f:
                    nodeset = json.load(f)
                    nodeset["id"] = nodeset_id
                    yield nodeset_id, nodeset

    def _generate_document(self, example, **kwargs):
        return convert_to_document(nodeset=example, nodeset_id=example["id"])
