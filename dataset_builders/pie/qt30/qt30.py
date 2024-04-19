import logging

import datasets
from pie_datasets import ArrowBasedBuilder

from src.utils.nodeset2document import SimplifiedQT30Document, convert_to_document
from src.utils.nodeset_utils import get_node_id_from_filename
from src.utils.prepare_data import prepare_nodeset

logger = logging.getLogger(__name__)

"""
Nodesets that are not suitable for conversion to documents:
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


class SimplifiedQT30(ArrowBasedBuilder):
    DOCUMENT_TYPE = SimplifiedQT30Document

    BASE_DATASET_PATH = "json"
    BASE_DATASET_REVISION = None

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="qt30", version=datasets.Version("1.0.0"), description="QT30 dataset"
        ),
    ]

    # This is the same as _split_generators from the HF json builder, see
    # https://github.com/huggingface/datasets/blob/2.15.0/src/datasets/packaged_modules/json/json.py,
    # but with the blacklist check added.
    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles."""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files if not is_blacklisted(file)]
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files if not is_blacklisted(file)]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_document(self, example, **kwargs):
        # TODO: use real nodeset_id
        nodeset_id = None
        nodeset = dict(example)
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
