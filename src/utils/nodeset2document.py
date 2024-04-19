import argparse
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

from dataset_builders.pie.dialam2024.dialam2024 import convert_to_document
from src.document.types import SimplifiedDialAM2024Document
from src.utils.nodeset_utils import Nodeset, process_all_nodesets, read_nodeset

logger = logging.getLogger(__name__)


def validate_document(nodeset: Nodeset, document: SimplifiedDialAM2024Document):
    metadata = document.metadata
    original = nodeset.copy()
    # check that we have the same L-nodes as in the original nodeset
    orig_l_node_id2text = dict()
    for n in original["nodes"]:
        if n["type"] == "L":
            orig_l_node_id2text[n["nodeID"]] = n["text"]
    l_node_ids = metadata["l_node_ids"]
    l_node_spans = {l_node_id: l_span for l_node_id, l_span in zip(l_node_ids, document.l_nodes)}
    converted_l_node_id2text = dict()
    for l_node_id in l_node_ids:
        l_span = l_node_spans[l_node_id]
        converted_l_node_id2text[l_node_id] = document.text[l_span.start : l_span.end]
    for l_id in orig_l_node_id2text:
        if l_id in converted_l_node_id2text:
            assert converted_l_node_id2text[l_id] == orig_l_node_id2text[l_id]
        else:
            raise ValueError(f"L-node missing in the converted document, nodeID: {l_id}")
    # check that we have the same TA-nodes as in the original nodeset
    ta_node_ids = metadata["ta_node_ids"]
    orig_ta_node_ids = [n["nodeID"] for n in original["nodes"] if n["type"] == "TA"]
    ta_node_diff = set(orig_ta_node_ids) - set(ta_node_ids)
    if len(ta_node_diff) != 0:
        raise ValueError(f"TA nodes missing in the converted document: {ta_node_diff}")

    # helper structures
    orig_src2trg = defaultdict(list)
    orig_trg2src = defaultdict(list)
    for e in original["edges"]:
        orig_src2trg[e["fromID"]].append(e["toID"])
    for e in original["edges"]:
        orig_trg2src[e["toID"]].append(e["fromID"])
    node_id2type = {n["nodeID"]: n["type"] for n in original["nodes"]}

    orig_mappings = {
        "orig_src2trg": orig_src2trg,
        "orig_trg2src": orig_trg2src,
        "node_id2type": node_id2type,
    }

    # check YA I2L relations are the same as in the original nodeset
    ya_i2l_relations = metadata["ya_i2l_relations"]
    allowed_types = {
        "allowed_node_types": ["YA"],
        "allowed_source_types": ["L"],
        "allowed_target_types": ["I"],
    }
    check_relations(
        original=original,
        relations=ya_i2l_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )
    # check YA S2TA relations are the same as in the original nodeset
    ya_s2ta_relations = metadata["ya_s2ta_relations"]
    allowed_types = {
        "allowed_node_types": ["YA"],
        "allowed_source_types": ["TA"],
        "allowed_target_types": ["S"],
    }
    check_relations(
        original=original,
        relations=ya_s2ta_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )
    # check S relations are the same as in the original nodeset
    s_relations = metadata["s_relations"]
    allowed_types = {
        "allowed_node_types": ["RA", "MA", "CA"],
        "allowed_source_types": ["I"],
        "allowed_target_types": ["I"],
    }
    check_relations(
        original=original,
        relations=s_relations,
        orig_mappings=orig_mappings,
        allowed_types=allowed_types,
    )


def check_relations(
    original: Nodeset,
    relations: List[Dict[str, Any]],
    orig_mappings: Dict[str, Any],
    allowed_types: Dict[str, List[str]],
):
    orig_src2trg = orig_mappings["orig_src2trg"]
    orig_trg2src = orig_mappings["orig_trg2src"]
    node_id2type = orig_mappings["node_id2type"]
    allowed_node_types = allowed_types["allowed_node_types"]
    allowed_source_types = allowed_types["allowed_source_types"]
    allowed_target_types = allowed_types["allowed_target_types"]
    # filter out only those node IDs where sources and targets have the correct (allowed) types
    orig_node_ids = []
    for n in original["nodes"]:
        if n["type"] in allowed_node_types:
            n_id = n["nodeID"]
            n_sources = orig_src2trg[n_id]
            n_targets = orig_trg2src[n_id]
            n_sources_filtered = [
                n_s for n_s in n_sources if node_id2type[n_s] in allowed_source_types
            ]
            n_targets_filtered = [
                n_t for n_t in n_targets if node_id2type[n_t] in allowed_target_types
            ]
            if len(n_sources_filtered) > 0 and len(n_targets_filtered) > 0:
                orig_node_ids.append(n_id)
    node_src_trg = dict()
    for rel in relations:
        node_src_trg[rel["relation"]] = {"sources": rel["sources"], "targets": rel["targets"]}
    # check for each relation whether we are missing the original source or target nodes
    for n_id in orig_node_ids:
        if not (n_id in node_src_trg):
            raise ValueError(
                f"{node_id2type[n_id]} node is not in the converted document, nodeID: {n_id}"
            )
        orig_sources = orig_trg2src[n_id]
        for orig_source in orig_sources:
            if node_id2type[orig_source] in allowed_source_types and not (
                orig_source in node_src_trg[n_id]["sources"]
            ):
                raise ValueError(f"Source is missing, nodeID: {orig_source} for nodeID: {n_id}")
        orig_targets = orig_src2trg[n_id]
        for orig_target in orig_targets:
            if node_id2type[orig_target] in allowed_target_types and not (
                orig_target in node_src_trg[n_id]["targets"]
            ):
                raise ValueError(f"Target is missing, nodeID: {orig_target} for nodeID: {n_id}")


def convert_and_validate(nodeset: Nodeset, nodeset_id: str, **kwargs) -> None:
    document = convert_to_document(nodeset=nodeset, nodeset_id=nodeset_id, **kwargs)
    validate_document(nodeset=nodeset, document=document)


def main(
    input_dir: str,
    show_progress: bool = True,
    verbose: bool = True,
    nodeset_id: Optional[str] = None,
    nodeset_blacklist: Optional[List[str]] = None,
    **kwargs,
):
    if nodeset_id is not None:
        nodeset = read_nodeset(nodeset_dir=input_dir, nodeset_id=nodeset_id)
        result = convert_to_document(
            nodeset=nodeset,
            nodeset_id=nodeset_id,
            **kwargs,
        )
        # write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result)
        # result.asdict()
    else:
        # if no nodeset ID is provided, process all nodesets in the input directory
        for nodeset_id, result_or_error in process_all_nodesets(
            func=convert_to_document,
            nodeset_dir=input_dir,
            show_progress=show_progress,
            nodeset_blacklist=nodeset_blacklist,
            **kwargs,
        ):
            if isinstance(result_or_error, Exception):
                logger.error(f"nodeset={nodeset_id}: Failed to process: {result_or_error}")
            else:
                # write_nodeset(nodeset_dir=output_dir, nodeset_id=nodeset_id, data=result_or_error)
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_dir", type=str, required=True, help="The input directory containing the nodesets."
    )
    parser.add_argument(
        "--nodeset_id",
        type=str,
        default=None,
        help="The ID of the nodeset to process. If not provided, all nodesets in the input directory will be processed.",
    )
    parser.add_argument(
        "--nodeset_blacklist",
        # split by comma and remove leading/trailing whitespaces
        type=lambda x: [nid.strip() for nid in x.split(",")] if x else None,
        default=None,
        help="List of nodeset IDs that should be ignored.",
    )
    parser.add_argument(
        "--dont_show_progress",
        dest="show_progress",
        action="store_false",
        help="Whether to show a progress bar when processing multiple nodesets.",
    )
    parser.add_argument(
        "--silent",
        dest="verbose",
        action="store_false",
        help="Whether to show verbose output.",
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    main(**args)
