"""This module defines a HuggingFace dataset builder for the QT30 dataset used in the DialAM-2024
shared task. See http://dialam.arg.tech/ for more information about the DialAM-2024 shared task.

Unfortunately, there are some nodesets that are not suitable for conversion to documents. These nodesets are
excluded from the dataset. The following nodesets are excluded:
- excluded by the organizers (23): 24255, 24807, 24808, 24809, 24903, 24905, 24992, 25045, 25441, 25442,
    25443, 25444, 25445, 25452, 25461, 25462, 25463, 25465, 25468, 25472, 25473, 25474, 25475
- excluded because of warning (6): "Could not align I-node (dummy-L-node was selected)": 21083, 18888,
    23701, 18484, 17938, 19319
- excluded because of error "could not determine direction of RA-nodes ... because there is no TA
    relation between any combination of anchoring I-nodes!" (26): 25411, 25510, 25516, 25901, 25902,
    25904, 25906, 25907, 25936, 25937, 25938, 25940, 26066, 26067, 26068, 26087, 17964, 18459, 19091,
    19146, 19149, 19757, 19761, 19908, 21449, 23749
- excluded because of error "S-node arguments are not unique!" (7): 25552, 19165, 22969, 21342, 25400,
    21681, 23710
- excluded because of error "direction of RA-node 587841 is ambiguous!" (16): 19059, 19217, 19878, 20479,
    20507, 20510, 20766, 20844, 20888, 20992, 21401, 21477, 21588, 23114, 23766, 23891
- excluded because of error "I-node texts are not unique!" (1): 19911
- still problematic (19): 19897, 18321, 18877, 18874, 19174, 23552, 23799, 23517, 20729, 25691, 21023,
    23144, 23120, 23560, 23892, 23959, 19173, 19918, 25511
"""
import glob
import json
import logging
import os

import datasets
from datasets import Features
from pie_datasets import GeneratorBasedBuilder

from src.utils.nodeset2document import SimplifiedQT30Document, convert_to_document
from src.utils.nodeset_utils import get_node_id_from_filename
from src.utils.prepare_data import prepare_nodeset

logger = logging.getLogger(__name__)

DATA_URL = "http://dialam.arg.tech/res/files/dataset.zip"
SUBDIR = "dataset"
NODESET_BLACKLIST = [
    "24255",
    "24807",
    "24808",
    "24809",
    "24903",
    "24905",
    "24992",
    "25045",
    "25441",
    "25442",
    "25443",
    "25444",
    "25445",
    "25452",
    "25461",
    "25462",
    "25463",
    "25465",
    "25468",
    "25472",
    "25473",
    "25474",
    "25475",
    "21083",
    "18888",
    "23701",
    "18484",
    "17938",
    "19319",
    "25411",
    "25510",
    "25516",
    "25901",
    "25902",
    "25904",
    "25906",
    "25907",
    "25936",
    "25937",
    "25938",
    "25940",
    "26066",
    "26067",
    "26068",
    "26087",
    "17964",
    "18459",
    "19091",
    "19146",
    "19149",
    "19757",
    "19761",
    "19908",
    "21449",
    "23749",
    "25552",
    "19165",
    "22969",
    "21342",
    "25400",
    "21681",
    "23710",
    "19059",
    "19217",
    "19878",
    "20479",
    "20507",
    "20510",
    "20766",
    "20844",
    "20888",
    "20992",
    "21401",
    "21477",
    "21588",
    "23114",
    "23766",
    "23891",
    "19911",
    "19897",
    "18321",
    "18877",
    "18874",
    "19174",
    "23552",
    "23799",
    "23517",
    "20729",
    "25691",
    "21023",
    "23144",
    "23120",
    "23560",
    "23892",
    "23959",
    "19173",
    "19918",
    "25511",
]


def is_blacklisted(nodeset_filename: str) -> bool:
    nodeset_id = get_node_id_from_filename(nodeset_filename)
    return nodeset_id in NODESET_BLACKLIST


def dictoflists_to_listofdicts(data):
    return [dict(zip(data, t)) for t in zip(*data.values())]


class SimplifiedQT30(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SimplifiedQT30Document

    BASE_DATASET_PATH = None
    BASE_DATASET_REVISION = None

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="qt30", version=datasets.Version("1.0.0"), description="QT30 dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=Features(
                {
                    "id": datasets.Value("string"),
                    "nodes": datasets.Sequence(
                        {
                            "nodeID": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "timestamp": datasets.Value("string"),
                            # Since optional fields are not supported in HuggingFace datasets, we exclude the
                            # scheme and schemeID fields from the dataset. Note that the scheme field has the
                            # same value as the text field where it is present.
                            # "scheme": datasets.Value("string"),
                            # "schemeID": datasets.Value("string"),
                        }
                    ),
                    "edges": datasets.Sequence(
                        {
                            "edgeID": datasets.Value("string"),
                            "fromID": datasets.Value("string"),
                            "toID": datasets.Value("string"),
                            "formEdgeID": datasets.Value("string"),
                        }
                    ),
                    "locutions": datasets.Sequence(
                        {
                            "nodeID": datasets.Value("string"),
                            "personID": datasets.Value("string"),
                            "timestamp": datasets.Value("string"),
                            "start": datasets.Value("string"),
                            "end": datasets.Value("string"),
                            "source": datasets.Value("string"),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles."""
        if dl_manager.manual_dir is None:
            data_dir = os.path.join(dl_manager.download_and_extract(DATA_URL), SUBDIR)
        else:
            # make absolute path of the manual_dir
            data_dir = os.path.abspath(dl_manager.manual_dir)
        # collect all json files in the data_dir with glob
        file_names = glob.glob(os.path.join(data_dir, "*.json"))
        file_names_filtered = [fn for fn in file_names if not is_blacklisted(fn)]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_names": file_names_filtered},
            )
        ]

    def _generate_examples(self, file_names):
        idx = 0
        for file_name in file_names:
            with open(file_name, encoding="utf-8", errors=None) as f:
                data = json.load(f)
            data["id"] = get_node_id_from_filename(file_name)
            # delete optional node fields: scheme, schemeID
            for node in data["nodes"]:
                node.pop("scheme", None)
                node.pop("schemeID", None)

            yield idx, data
            idx += 1

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
