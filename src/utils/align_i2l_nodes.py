"""How to use the script for checking node alignments:

(1) Install SentenceTransformers with `pip install sentence-transformers` (see requirements.txt)

(2) Install textdistance with `pip install textdistance` (see requirements.txt)

(3) Run the script as follows:

$ python3 src/utils/align_i2l_nodes.py path_to_nodesets similarity_measure nodeset_id

For example:

$ python3 src/utils/align_i2l_nodes.py data cossim 17940

`data` is the path to the dataset with the nodesets in JSON format

`cossim` is the similarity measure to use, similarity measure can have the following values:
 - cossim: cosine similarity with SentenceTransformers (embedding-based)
 - jaccard: Jaccard index (token-based)
 - tversky: Tversky index (token-based)
 - sorensen: Sorensen-Dice coefficient (token-based)
 - tanimoto: Tanimoto distance (token-based)
 - overlap: Overlap coefficient (token-based)
 - bag: Bag distance (token-based)
 - lcsstr: Longest common substring similarity (sequence-based)
 - ratcliff_obershelp: Ratcliff-Obershelp similarity (sequence-based)
For more details on textdistance metrics see: https://pypi.org/project/textdistance/
For more details on SentenceTransformers models see: https://www.sbert.net/docs/pretrained_models.html

`21388` is the nodeset id (in this example for `nodeset21388.json`).

Note: If no nodeset id is provided, the script will compute the matches and mismatches between
the I- and L-nodes for all nodesets in the directory.
"""

import argparse
import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import textdistance
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

from src.utils.nodeset_utils import process_all_nodesets, read_nodeset

logger = logging.getLogger(__name__)

DISTANCE_BASED_SIMILARITY = ["bag"]
SEQUENCE_BASED_SIMILARITY = ["lcsstr", "ratcliff_obershelp"]
SIMILARITY_METRICS = [
    "jaccard",
    "tversky",
    "sorensen",
    "tanimoto",
    "overlap",
    "bag",
    "lcsstr",
    "ratcliff_obershelp",
]

# text distance metrics based on https://pypi.org/project/textdistance/
metric_mapping = {
    "jaccard": textdistance.jaccard,
    "tversky": textdistance.tversky,
    "sorensen": textdistance.sorensen,
    "tanimoto": textdistance.tanimoto,
    "overlap": textdistance.overlap,
    "bag": textdistance.bag,
    "lcsstr": textdistance.lcsstr,
    "ratcliff_obershelp": textdistance.ratcliff_obershelp,
}


def align_i_and_l_nodes(
    node_id2node: Dict[str, Dict],
    l_node_ids: List[str],
    i_node_ids: List[str],
    similarity_measure: str,
    nodeset_id: str,
    smodel=None,
) -> List[Tuple[str, str]]:
    # extract text for sorted L-nodes and I-nodes
    l_node_texts = []
    i_node_texts = []
    for l_node_id in l_node_ids:
        l_node_text = node_id2node[l_node_id]["text"].lower()
        # remove the speaker prefix
        if ":" in l_node_text:
            l_node_text = l_node_text[l_node_text.index(":") + 1 :].strip()
        l_node_texts.append(l_node_text)
    for i_node_id in i_node_ids:
        i_node_text = node_id2node[i_node_id]["text"].lower()
        i_node_texts.append(i_node_text)

    if similarity_measure == "cossim" and smodel is not None:
        # use SentenceEmbeddings to align the nodes (text-based)
        l_node_embeds = smodel.encode(l_node_texts)
        i_node_embeds = smodel.encode(i_node_texts)
        similarity_matrix = util.cos_sim(i_node_embeds, l_node_embeds).tolist()
    else:
        if similarity_measure in SIMILARITY_METRICS:
            # use textdistance to align the nodes (text-based)
            similarity_matrix = []
            metric = metric_mapping.get(similarity_measure, "Invalid metric")
            for i in range(len(i_node_texts)):
                new_row = []
                i_node_text = i_node_texts[i]
                i_node_text_str = "".join(i_node_text)

                for j in range(len(l_node_texts)):
                    l_node_text = l_node_texts[j]
                    l_node_text_str = "".join(l_node_text)
                    # longest common substring similarity
                    if similarity_measure in SEQUENCE_BASED_SIMILARITY:
                        if similarity_measure == "lcsstr":
                            similarity = 1.0 * len(metric(i_node_text_str, l_node_text_str))
                        else:
                            similarity = metric(i_node_text_str, l_node_text_str)
                        similarity /= max(len(i_node_text_str), len(l_node_text_str))
                    else:
                        similarity = metric(i_node_text, l_node_text)
                    new_row.append(similarity)
                similarity_matrix.append(new_row)
        else:
            raise NotImplementedError(f"Unknown similarity measure: {similarity_measure}")

    max_value = max([max(row) for row in similarity_matrix])
    if similarity_measure in DISTANCE_BASED_SIMILARITY:
        maximize = False
        dummy_value = max_value + 1
    else:
        maximize = True
        dummy_value = -1

    # make sure that we have a square similarity matrix
    # if we have more I-nodes than L-nodes
    if len(similarity_matrix) > len(similarity_matrix[0]):
        # modified_matrix = True
        for row in range(len(similarity_matrix)):
            while len(similarity_matrix[row]) < len(similarity_matrix):
                similarity_matrix[row].append(dummy_value)
    # if we have more-L nodes than I-nodes
    elif len(similarity_matrix) < len(similarity_matrix[0]):
        # modified_matrix = True
        while len(similarity_matrix) < len(similarity_matrix[0]):
            dummy_row = [dummy_value for _ in range(len(similarity_matrix[0]))]
            similarity_matrix.append(dummy_row)

    assignments = linear_sum_assignment(similarity_matrix, maximize=maximize)
    # align each I-node with L-node
    # Q: why I -> L and not L -> I alignment?
    # A: there can be L-nodes that are not aligned to any I-nodes but each I-node must be aligned to some L-node
    aligned_il_nodes = []
    for i in range(len(i_node_ids)):
        best_l_candidate = assignments[1][i]
        if best_l_candidate < len(l_node_ids):
            aligned_il_nodes.append((i_node_ids[i], l_node_ids[best_l_candidate]))
        else:
            # warn about failed I-to-L alignment ("dummy" L-node was selected)
            logger.warning(f"nodeset={nodeset_id}: Could not align I-node: {i_node_ids[i]}")

    return aligned_il_nodes


# returns number of matched and mismatched I-L alignments (through YA-nodes)
def evaluate_align_nodes(
    nodeset_dir: str,
    similarity_measure: str,
    nodeset_id: str,
    smodel: Optional[Any] = None,
):

    # read the JSON data
    data = read_nodeset(nodeset_dir=nodeset_dir, nodeset_id=nodeset_id)

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
    node_ids2node_type: Dict[str, str] = defaultdict(str)
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
                node_ids2node_type[n["nodeID"]] = node_type
            else:
                duplicate_node_ids.add(n["nodeID"])
        else:
            disconnected_node_ids.add(n["nodeID"])

    # warn about duplicates
    if len(duplicate_node_ids) > 0:
        logger.warning(f"nodeset={nodeset_id}: Duplicate nodes: {duplicate_node_ids}")

    # warn about missing L-nodes in locutions
    missing_l_nodes_in_locutions = []
    for n in node_types2node_ids["L"]:
        if not (n in node_id2locution):
            missing_l_nodes_in_locutions.append(n)
    if len(missing_l_nodes_in_locutions) > 0:
        logger.warning(
            f"nodeset={nodeset_id}: Missing L-nodes in locutions: {missing_l_nodes_in_locutions}"
        )

    # do not align by "locution" timestamps since those are missing for some L-nodes (e.g., L-node 70682 in nodeset 21303)!
    l_node_ids_sorted = sorted(
        node_types2node_ids["L"],
        key=lambda x: datetime.datetime.fromisoformat(node_id2node[x]["timestamp"]),
    )

    # sort I-nodes by timestamp
    i_node_ids_sorted = sorted(
        node_types2node_ids["I"],
        key=lambda x: datetime.datetime.fromisoformat(node_id2node[x]["timestamp"]),
    )

    # align I and L-nodes
    aligned_il_nodes = align_i_and_l_nodes(
        node_id2node=node_id2node,
        l_node_ids=l_node_ids_sorted,
        i_node_ids=i_node_ids_sorted,
        similarity_measure=similarity_measure,
        nodeset_id=nodeset_id,
        smodel=smodel,
    )

    total_matched = 0
    total_mismatched = 0
    aligned_i2l_nodes = {i_node: l_node for i_node, l_node in aligned_il_nodes}
    i2l_gold_nodes = defaultdict(list)
    for i_node in node_types2node_ids["I"]:
        # L and I-nodes must be connected via YA-node
        for ya_node in trg2sources[i_node]:
            for ya_source in trg2sources[ya_node]:
                if node_ids2node_type[ya_source] == "L":
                    i2l_gold_nodes[i_node].append(ya_source)
    for i_node in i2l_gold_nodes:
        if i_node in aligned_i2l_nodes and aligned_i2l_nodes[i_node] in i2l_gold_nodes[i_node]:
            total_matched += 1
        else:
            total_mismatched += 1

    # check whether L-node can be aligned to multiple I-nodes in the gold data
    l2i_gold_nodes = defaultdict(list)
    for l_node in node_types2node_ids["L"]:
        # L and I-nodes must be connected via YA-node
        for ya_node in src2targets[l_node]:
            for ya_target in src2targets[ya_node]:
                if node_ids2node_type[ya_target] == "I":
                    l2i_gold_nodes[l_node].append(ya_target)
    l2i_gold_multiple_alignments = sum(
        [len(l2i_gold_nodes[l_node]) - 1 for l_node in l2i_gold_nodes]
    )

    # check whether our alignments have the same L-node aligned to multiple I-nodes
    # this should NOT happen if we use linear_sum_assignment
    l2i_assigned_nodes = defaultdict(list)
    for i_node, l_node in aligned_il_nodes:
        l2i_assigned_nodes[l_node].append(i_node)
    l2i_assigned_multiple_alignments = sum(
        [len(l2i_assigned_nodes[l_node]) - 1 for l_node in l2i_assigned_nodes]
    )

    if l2i_gold_multiple_alignments > 0:
        # warn about multiple alignments
        gold_multiple_alignments = [
            "L-Node: " + l_node + " I-Nodes: " + " ".join(l2i_gold_nodes[l_node])
            for l_node in l2i_gold_nodes
            if len(l2i_gold_nodes[l_node]) > 1
        ]
        logger.warning(
            f"nodeset={nodeset_id}: Multiple alignments between L-node and I-nodes {gold_multiple_alignments}"
        )

    assert total_matched + total_mismatched == len(i2l_gold_nodes)

    alignments = {
        "aligned_il_nodes": aligned_il_nodes,
        "total_matched": total_matched,
        "total_mismatched": total_mismatched,
        "l2i_gold_multiple_alignments": l2i_gold_multiple_alignments,
        "l2i_assigned_multiple_alignments": l2i_assigned_multiple_alignments,
    }

    return alignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    parser.add_argument(
        "similarity_measure",
        type=str,
        help="similarity measure for comparing I and L-nodes. Available options: "
        + " ".join(SIMILARITY_METRICS),
    )
    parser.add_argument(
        "nodeset",
        nargs="?",
        type=str,
        help="nodeset (argument map) id. This is optional, if not provided, all nodesets in the "
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

    kwargs = {
        "nodeset_dir": args["nodeset_dir"],
        "similarity_measure": args["similarity_measure"],
        "smodel": smodel,
    }

    if args["nodeset"] is not None:
        alignments = evaluate_align_nodes(nodeset_id=args["nodeset"], **kwargs)
    else:
        alignments = dict()
        for nodeset, result in process_all_nodesets(func=evaluate_align_nodes, **kwargs):
            if isinstance(result, Exception):
                logger.error(f"nodeset={nodeset}: Failed to process: {result}")
            else:
                for key in result:
                    if key in alignments:
                        alignments[key] += result[key]
                    else:
                        alignments[key] = result[key]

    total_matched = alignments["total_matched"]
    total_mismatched = alignments["total_mismatched"]
    l2i_gold_multiple_alignments = alignments["l2i_gold_multiple_alignments"]
    l2i_assigned_multiple_alignments = alignments["l2i_assigned_multiple_alignments"]

    print(
        "Result:",
        "\nmatched I-L pairs:",
        total_matched,
        "\nmismatched I-L pairs:",
        total_mismatched,
        "\naligned I-L pairs:",
        total_matched + total_mismatched,
        "\nL-I multiple alignments (gold)",
        l2i_gold_multiple_alignments,
        "\nL-I multiple alignments (automatically assigned)",
        l2i_assigned_multiple_alignments,
    )
