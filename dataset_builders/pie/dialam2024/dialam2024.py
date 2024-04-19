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
import logging

import datasets
from pie_datasets import GeneratorBasedBuilder

from src.utils.nodeset2document import SimplifiedDialAM2024Document, convert_to_document
from src.utils.prepare_data import prepare_nodeset

logger = logging.getLogger(__name__)


def dictoflists_to_listofdicts(data):
    return [dict(zip(data, t)) for t in zip(*data.values())]


class PieDialAM2024(GeneratorBasedBuilder):
    DOCUMENT_TYPE = SimplifiedDialAM2024Document

    BASE_DATASET_PATH = "ArneBinder/dialam2024"
    BASE_DATASET_REVISION = "e484998b3822746a232a4faa882cdb3e533ab9f4"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dialam2024",
            version=datasets.Version("1.0.0"),
            description="PIE-wrapped DialAM-2024 dataset",
        ),
    ]

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
