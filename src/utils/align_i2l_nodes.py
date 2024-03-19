"""How to use the script for checking node alignments:

(1) Install SentenceTransformers with `pip install sentence-transformers` (see requirements.txt)

(2) Install textdistance with `pip install textdistance` (see requirements.txt)

(3) Run the script as follows:

$ python3 src/utils/align_i2l_nodes.py path_to_nodesets similarity_measure nodeset_id

For example:

$ python3 src/utils/align_i2l_nodes.py data cossim 17940

`data` is the path to the dataset with the nodesets in JSON format

`cossim` is the similarity measure to use, it can have the following values:
 - cossim: cosine similarity with SentenceTransformers (embedding-based)
 - jaccard: Jaccard index (token-based)
 - tversky: Tversky index (token-based)
 - sorensen: Sorensen-Dice coefficient (token-based)
 - tanimoto: Tanimoto distance (token-based)
 - overlap: Overlap coefficient (token-based)
 - bag: Bag distance (token-based)
 - lcsstr: Longest common substring similarity (sequence-based)
For more details on textdistance metrics see: https://pypi.org/project/textdistance/
For more details on SentenceTransformers models see: https://www.sbert.net/docs/pretrained_models.html

`21388` is the nodeset id (in this example for `nodeset21388.json`).

Note: If no nodeset id is provided, the script will compute the matches and mismatches between the I and L-ndoes for all nodesets in the directory.
"""

import argparse
import datetime
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import textdistance
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

DISTANCE_BASED_SIMILARITY = ["bag"]
SIMILARITY_METRICS = ["jaccard", "tversky", "sorensen", "tanimoto", "overlap", "bag", "lcsstr"]

# text distance metrics based on https://pypi.org/project/textdistance/
metric_mapping = {
    "jaccard": textdistance.jaccard,
    "tversky": textdistance.tversky,
    "sorensen": textdistance.sorensen,
    "tanimoto": textdistance.tanimoto,
    "overlap": textdistance.overlap,
    "bag": textdistance.bag,
    "lcsstr": textdistance.lcsstr,
}


# returns number of matched and mismatched I-L alignments (through YA-nodes)
def align_nodes(
    nodeset_dir: str,
    similarity_measure: str,
    nodeset: Optional[str] = None,
    smodel: Optional[Any] = None,
):
    # if no nodeset id is provided, check all nodesets in the directory
    if nodeset is None:
        total_matched = 0
        total_mismatched = 0
        nodeset_ids = [
            f.split("nodeset")[1].split(".json")[0]
            for f in os.listdir(nodeset_dir)
            if f.endswith(".json")
        ]
        for nodeset in nodeset_ids:
            try:
                nodeset_matched, nodeset_mismatched = align_nodes(
                    nodeset_dir=nodeset_dir,
                    similarity_measure=similarity_measure,
                    nodeset=nodeset,
                    smodel=smodel,
                )
                total_matched += nodeset_matched
                total_mismatched += nodeset_mismatched
            except Exception as e:
                logger.error(f"nodeset={nodeset}: Failed to process: {e}")
        return total_matched, total_mismatched

    # read the JSON data
    filename = os.path.join(nodeset_dir, f"nodeset{nodeset}.json")
    with open(filename) as f:
        data = json.load(f)

    # edge related helper data structures
    src2targets = defaultdict(list)
    trg2sources = defaultdict(list)
    edges = set()
    for edge_dict in data["edges"]:
        src2targets[edge_dict["fromID"]].append(edge_dict["toID"])
        trg2sources[edge_dict["toID"]].append(edge_dict["fromID"])
        edges.add((edge_dict["fromID"], edge_dict["toID"]))

    # node related helper data structures
    node_id2node = {n["nodeID"]: n for n in data["nodes"]}
    node_id2locution = {n["nodeID"]: n for n in data["locutions"]}
    node_types2node_ids: Dict[str, Set[str]] = defaultdict(set)
    disconnected_node_ids = set()
    duplicate_node_ids = set()
    for n in data["nodes"]:
        node_type = n["type"]
        if node_type in ["RA", "CA", "MA"]:
            node_type = "S"

        # only collect connected nodes
        if n["nodeID"] in src2targets or n["nodeID"] in trg2sources:
            if n["nodeID"] not in node_types2node_ids[node_type]:
                node_types2node_ids[node_type].add(n["nodeID"])
            else:
                duplicate_node_ids.add(n["nodeID"])
        else:
            disconnected_node_ids.add(n["nodeID"])

    if len(duplicate_node_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Duplicate nodes: {duplicate_node_ids}")

    # sort L-nodes by timestamp that comes from locutions because other timestamps can be arbitrary
    l_node_ids_sorted = sorted(
        [n for n in node_types2node_ids["L"] if n in node_id2locution],
        key=lambda x: datetime.datetime.fromisoformat(node_id2locution[x]["timestamp"]),
    )

    # sort I-nodes by timestamp
    i_node_ids_sorted = sorted(
        node_types2node_ids["I"],
        key=lambda x: datetime.datetime.fromisoformat(node_id2node[x]["timestamp"]),
    )

    # extract text for sorted L-nodes and I-nodes
    l_node_texts_sorted = []
    i_node_texts_sorted = []
    for l_node in l_node_ids_sorted:
        l_node_text = node_id2node[l_node]["text"].lower()
        # remove the speaker prefix
        if ":" in l_node_text:
            l_node_text = l_node_text[l_node_text.index(":") + 1 :].strip()
        l_node_texts_sorted.append(l_node_text)
    for i_node in i_node_ids_sorted:
        i_node_text = node_id2node[i_node]["text"].lower()
        i_node_texts_sorted.append(i_node_text)

    if similarity_measure == "cossim" and smodel is not None:
        # use SentenceEmbeddings to align the nodes (text-based)
        l_node_embeds = smodel.encode(l_node_texts_sorted)
        i_node_embeds = smodel.encode(i_node_texts_sorted)
        similarity_matrix = util.cos_sim(i_node_embeds, l_node_embeds)
    else:
        if similarity_measure in SIMILARITY_METRICS:
            # use textdistance to align the nodes (text-based)
            similarity_matrix = []
            metric = metric_mapping.get(similarity_measure, "Invalid metric")
            for i in range(len(i_node_texts_sorted)):
                new_row = []
                i_node_text = i_node_texts_sorted[i]
                i_node_text_str = "".join(i_node_text)

                for j in range(len(l_node_texts_sorted)):
                    l_node_text = l_node_texts_sorted[j]
                    l_node_text_str = "".join(l_node_text)
                    # longest common substring similarity
                    if similarity_measure == "lcsstr":
                        similarity = 1.0 * len(metric(i_node_text_str, l_node_text_str))
                        similarity /= max(len(i_node_text_str), len(l_node_text_str))
                    else:
                        similarity = metric(i_node_text, l_node_text)
                    new_row.append(similarity)
                similarity_matrix.append(new_row)
        else:
            raise NotImplementedError(f"Unknown similarity measure: {similarity_measure}")
    aligned_il_nodes = []
    # align each I-node with L-node
    # Q: why I -> L and not L -> I alignment?
    # A: there can be L-nodes that are not aligned to any I-nodes but each I-node must be aligned to some L-node
    for i in range(len(i_node_texts_sorted)):
        l_candidates = []
        for l_candidate in range(len(l_node_texts_sorted)):
            l_candidates.append([similarity_matrix[i][l_candidate], l_candidate])
        if similarity_measure in DISTANCE_BASED_SIMILARITY:
            reverse = False
        else:
            reverse = True
        l_candidates = sorted(l_candidates, key=lambda x: x[0], reverse=reverse)
        best_l_candidate = l_candidates[0][-1]
        aligned_il_nodes.append((i_node_ids_sorted[i], l_node_ids_sorted[best_l_candidate]))

    total_matched = 0
    total_mismatched = 0
    for i_node, l_node in aligned_il_nodes:
        # find the gold alignment between the I-node and L-node
        matched = False
        gold_l_node = None
        for ya_node in node_types2node_ids["YA"]:
            if ya_node in trg2sources[i_node]:
                ya_sources = trg2sources[ya_node]
                for ya_source in ya_sources:
                    if node_id2node[ya_source]["type"] == "L":
                        gold_l_node = ya_source
                        break
            # check whether aligned L-node is the same as gold L-node
            if str(gold_l_node) == l_node:
                matched = True
                break
        if matched:
            total_matched += 1
        else:
            total_mismatched += 1

    assert total_matched + total_mismatched == len(aligned_il_nodes)
    return total_matched, total_mismatched


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    parser.add_argument(
        "similarity_measure",
        type=str,
        help="similarity measure for comparing I and L-nodes. Available options: cossim, jaccard, tversky, sorensen, tanimoto, overlap, bag, lcsstr",
    )
    parser.add_argument(
        "nodeset",
        nargs="?",
        type=str,
        help="nodeset (argument map) id. This is optional, if not provided, all nodesets of in the "
        "nodeset_dir are processed",
        default=None,
    )
    args = vars(parser.parse_args())
    if args["similarity_measure"] == "cossim":
        # Options: # "all-distilroberta-v1", "all-mpnet-base-v2", "all-MiniLM-L6-v2"
        # See the complete list here: https://www.sbert.net/docs/pretrained_models.html
        smodel = SentenceTransformer("all-distilroberta-v1")
    else:
        smodel = None

    total_matched, total_mismatched = align_nodes(
        nodeset_dir=args["nodeset_dir"],
        similarity_measure=args["similarity_measure"],
        nodeset=args["nodeset"],
        smodel=smodel,
    )
    print(
        "Result:",
        "matched:",
        total_matched,
        "mismatched:",
        total_mismatched,
        "aligned I-L nodes:",
        total_matched + total_mismatched,
    )
