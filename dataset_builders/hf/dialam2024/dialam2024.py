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
- excluded because of error "I-node texts are not unique!" (1): 18293

Problematic, but still included:
- warnings because of TA-loops (42): 18321, 18795, 18874, 18877, 19173, 19174, 19773, 19897, 19918, 20729, 20894, 21022, 21023, 21039, 21275, 21279, 23120, 23144, 23391, 23479, 23517, 23533, 23551, 23552, 23560, 23599, 23688, 23696, 23789, 23799, 23809, 23837, 23849, 23853, 23878, 23892, 23959, 25511, 25526, 25528, 25691, 25723
"""
import glob
import json
import logging
import os

import datasets
from datasets import Features, GeneratorBasedBuilder

logger = logging.getLogger(__name__)

DATA_URL = "http://dialam.arg.tech/res/files/dataset.zip"
SAMPLE_TEST_DATA_URL = "http://dialam.arg.tech/res/files/sample_test.zip"
TEST_DATA_URL = "http://dialam.arg.tech/res/files/test-data.zip"
SUBDIR = "dataset"
SAMPLE_TEST_SUBDIR = "sample_test"
TEST_SUBDIR = "test"
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
    "18293",
]


def is_blacklisted(nodeset_filename: str) -> bool:
    nodeset_id = get_node_id_from_filename(nodeset_filename)
    return nodeset_id in NODESET_BLACKLIST


def get_node_id_from_filename(filename: str) -> str:
    """Get the ID of a nodeset from a filename."""

    fn_no_ext = os.path.splitext(os.path.basename(filename))[0]
    if "nodeset" in fn_no_ext:
        return fn_no_ext.split("nodeset")[1]
    else:
        return fn_no_ext


def _construct_split_generator(data_dir: str, split_name: str) -> datasets.SplitGenerator:
    # collect all json files in the data_dir with glob
    sample_test_file_names = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    sample_test_file_names_filtered = [
        fn for fn in sample_test_file_names if not is_blacklisted(fn)
    ]
    return datasets.SplitGenerator(
        name=split_name,
        gen_kwargs={"file_names": sample_test_file_names_filtered},
    )


class DialAM2024(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="DialAM-2024 dataset",
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
            test_data_dir = os.path.join(
                dl_manager.download_and_extract(TEST_DATA_URL), TEST_SUBDIR
            )
            sample_test_data_dir = os.path.join(
                dl_manager.download_and_extract(SAMPLE_TEST_DATA_URL), SAMPLE_TEST_SUBDIR
            )
        else:
            # make absolute path of the manual_dir
            data_dir = os.path.abspath(dl_manager.manual_dir)
            test_data_dir = None
            sample_test_data_dir = None
        result = [_construct_split_generator(data_dir, datasets.Split.TRAIN)]
        if test_data_dir is not None:
            result.append(_construct_split_generator(test_data_dir, datasets.Split.TEST))
        if sample_test_data_dir is not None:
            result.append(_construct_split_generator(sample_test_data_dir, "sample_test"))

        return result

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

            # set missing entries to None
            for key, subkey in [
                # ("nodes", "scheme"),
                # ("nodes", "schemeID"),
                ("nodes", "timestamp"),
                ("locutions", "timestamp"),
                ("locutions", "source"),
                ("edges", "formEdgeID"),
            ]:
                for entry in data[key]:
                    if subkey not in entry:
                        entry[subkey] = None

            yield idx, data
            idx += 1
