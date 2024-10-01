import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import json
import os

from dataset_builders.pie.dialam2024.dialam2024 import convert_to_example, unmerge_relations
from src.document.types import TextDocumentWithLabeledEntitiesAndNaryRelations
from src.serializer import JsonSerializer


def main(args):
    docs = JsonSerializer.read(
        path=args.input_dir,
        file_name="documents.jsonl",
        document_type=TextDocumentWithLabeledEntitiesAndNaryRelations,
    )

    output_dir = args.output_dir
    for doc in docs:
        # convert to SimplifiedDialAM2024Document
        unmerged_document = unmerge_relations(doc)
        # convert to shared task format
        result = convert_to_example(unmerged_document, use_predictions=True)
        # get and remove the doc id, it should not be part of the file content
        doc_id = result.pop("id")
        # create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{doc.id}.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, help="path to the directory with serialized JSON documents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the directory with nodesets where each nodeset is stored in a separate JSON file (in the format required by the DialAM Shared Task)",
    )
    args = parser.parse_args()
    main(args)
