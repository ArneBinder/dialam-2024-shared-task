import argparse
import itertools
import logging
from typing import Dict, Optional

import pyrootutils
from sklearn.metrics import precision_recall_fscore_support

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

from src.utils.nodeset_utils import Nodeset, read_nodeset


def eval_arguments(
    nodeset_id: str,
    predictions_dir: str,
    gold_dir: str,
    nodeset: Optional[Nodeset] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:

    preds = read_nodeset(predictions_dir, nodeset_id) if nodeset is None else nodeset
    truth = read_nodeset(gold_dir, nodeset_id)

    proposition_dict = {}
    proposition_list = []
    true_inference_list = []
    true_conflict_list = []
    true_rephrase_list = []
    pred_inference_list = []
    pred_conflict_list = []
    pred_rephrase_list = []

    # Get the list of proposition nodes
    for node in truth["nodes"]:
        if node["type"] == "I":
            proposition_list.append(node["nodeID"])
            proposition_dict[node["nodeID"]] = node["text"]

    # Check truth relations
    for node in truth["nodes"]:
        if node["type"] == "RA":
            inference_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == inference_id:
                    conclusion_id = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == inference_id:
                            premise_id = edge["fromID"]
                            if premise_id in proposition_list:
                                true_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node["type"] == "CA":
            conflict_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == conflict_id:
                    conf_to = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == conflict_id:
                            conf_from = edge["fromID"]
                            if conf_from in proposition_list:
                                true_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node["type"] == "MA":
            rephrase_id = node["nodeID"]

            for edge in truth["edges"]:
                if edge["fromID"] == rephrase_id:
                    reph_to = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == rephrase_id:
                            reph_from = edge["fromID"]
                            if reph_from in proposition_list:
                                true_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # Check predicted relation
    for node in preds["nodes"]:
        if node["type"] == "RA":
            inference_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == inference_id:
                    conclusion_id = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == inference_id:
                            premise_id = edge["fromID"]
                            if premise_id in proposition_list:
                                pred_inference_list.append([premise_id, conclusion_id, 0])
                    break

        elif node["type"] == "CA":
            conflict_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == conflict_id:
                    conf_to = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == conflict_id:
                            conf_from = edge["fromID"]
                            if conf_from in proposition_list:
                                pred_conflict_list.append([conf_from, conf_to, 1])
                    break

        elif node["type"] == "MA":
            rephrase_id = node["nodeID"]

            for edge in preds["edges"]:
                if edge["fromID"] == rephrase_id:
                    reph_to = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == rephrase_id:
                            reph_from = edge["fromID"]
                            if reph_from in proposition_list:
                                pred_rephrase_list.append([reph_from, reph_to, 2])
                    break

    # ID to text
    """
    text_inference_list = []
    for inference in inference_list:
        text_inference_list.append([proposition_dict[inference[0]], proposition_dict[inference[1]], 1])
    text_conflict_list = []
    for conflict in conflict_list:
        text_conflict_list.append([proposition_dict[conflict[0]], proposition_dict[conflict[1]], 2])
    text_rephrase_list = []
    for rephrase in rephrase_list:
        text_rephrase_list.append([proposition_dict[rephrase[0]], proposition_dict[rephrase[1]], 3])
    """

    p_c = itertools.permutations(proposition_list, 2)
    proposition_combinations = []
    for p in p_c:
        proposition_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proposition_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for inference in true_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_true.append(0)
                added_true = True
                break
        if not added_true:
            for conflict in true_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_true.append(1)
                    added_true = True
                    break
        if not added_true:
            for rephrase in true_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_true.append(2)
                    added_true = True
                    break
        if not added_true:
            y_true.append(3)

        # Prepare Y pred
        for inference in pred_inference_list:
            if inference[0] == comb[0] and inference[1] == comb[1]:
                y_pred.append(0)
                added_pred = True
                break
        if not added_pred:
            for conflict in pred_conflict_list:
                if conflict[0] == comb[0] and conflict[1] == comb[1]:
                    y_pred.append(1)
                    added_pred = True
                    break
        if not added_pred:
            for rephrase in pred_rephrase_list:
                if rephrase[0] == comb[0] and rephrase[1] == comb[1]:
                    y_pred.append(2)
                    added_pred = True
                    break
        if not added_pred:
            y_pred.append(3)

    if verbose:
        print(y_true)
        print(y_pred)
    focused_true = []
    focused_pred = []
    for i in range(len(y_true)):
        if y_true[i] != 3:
            focused_true.append(y_true[i])
            focused_pred.append(y_pred[i])

    if verbose:
        print(focused_true)
        print(focused_pred)

    result_general = precision_recall_fscore_support(y_true, y_pred, average="macro")
    result_focused = precision_recall_fscore_support(focused_true, focused_pred, average="macro")
    if verbose:
        print("General", result_general)
        print("Focused", result_focused)

    return {
        "general": {"p": result_general[0], "r": result_general[1], "f1": result_general[2]},
        "focused": {"p": result_focused[0], "r": result_focused[1], "f1": result_focused[2]},
    }


def eval_illocutions(
    nodeset_id: str,
    predictions_dir: str,
    gold_dir: str,
    nodeset: Optional[Nodeset] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:

    preds = read_nodeset(predictions_dir, nodeset_id) if nodeset is None else nodeset
    truth = read_nodeset(gold_dir, nodeset_id)

    proposition_dict = {}
    proposition_list = []
    locution_dict = {}
    locution_list = []
    true_illocution_list = []
    pred_illocution_list = []

    # Get the list of proposition and locution nodes
    for node in truth["nodes"]:
        if node["type"] == "I":
            proposition_list.append(node["nodeID"])
            proposition_dict[node["nodeID"]] = node["text"]
        elif node["type"] == "L":
            locution_list.append(node["nodeID"])
            locution_dict[node["nodeID"]] = node["text"]

    proploc_list = proposition_list + locution_list

    # Check truth illocutions
    for node in truth["nodes"]:
        if node["type"] == "YA":
            illocution_id = node["nodeID"]
            illocution_type = node["text"]

            for edge in truth["edges"]:
                if edge["fromID"] == illocution_id:
                    target_id = edge["toID"]
                    for edge in truth["edges"]:
                        if edge["toID"] == illocution_id:
                            source_id = edge["fromID"]
                            if source_id in proploc_list and target_id in proploc_list:
                                true_illocution_list.append(
                                    [source_id, target_id, illocution_type]
                                )
                    break

    # Check predicted illocutions
    for node in preds["nodes"]:
        if node["type"] == "YA":
            illocution_id = node["nodeID"]
            illocution_type = node["text"]

            for edge in preds["edges"]:
                if edge["fromID"] == illocution_id:
                    target_id = edge["toID"]
                    for edge in preds["edges"]:
                        if edge["toID"] == illocution_id:
                            source_id = edge["fromID"]
                            if source_id in proploc_list and target_id in proploc_list:
                                pred_illocution_list.append(
                                    [source_id, target_id, illocution_type]
                                )
                    break

    # print(true_illocution_list)
    # print(pred_illocution_list)

    p_c = itertools.product(locution_list, proposition_list)
    proploc_combinations = []
    for p in p_c:
        proploc_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proploc_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for illocution in true_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_true.append(illocution[2])
                added_true = True
                break

        if not added_true:
            y_true.append("None")

        # Prepare Y pred
        for illocution in pred_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_pred.append(illocution[2])
                added_pred = True
                break

        if not added_pred:
            y_pred.append("None")

    if verbose:
        print(y_true)
        print(y_pred)

    focused_true = []
    focused_pred = []
    for i in range(len(y_true)):
        if y_true[i] != "None":
            focused_true.append(y_true[i])
            focused_pred.append(y_pred[i])

    if verbose:
        print(focused_true)
        print(focused_pred)

    result_general = precision_recall_fscore_support(y_true, y_pred, average="macro")
    result_focused = precision_recall_fscore_support(focused_true, focused_pred, average="macro")
    if verbose:
        print("General", result_general)
        print("Focused", result_focused)

    return {
        "general": {"p": result_general[0], "r": result_general[1], "f1": result_general[2]},
        "focused": {"p": result_focused[0], "r": result_focused[1], "f1": result_focused[2]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate arguments")
    parser.add_argument(
        "--mode", type=str, required=True, help="Mode of evaluation (arguments/illocutions)"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to the directory containing the gold nodesets",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        required=True,
        help="Path to the directory containing the predicted nodesets",
    )
    parser.add_argument("--nodeset_id", type=str, help="ID of the nodeset to evaluate")
    parser.add_argument(
        "--silent", dest="verbose", action="store_false", help="Whether to show verbose output"
    )

    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO)
    mode = args.pop("mode")

    if mode == "arguments":
        print(eval_arguments(**args))
    elif mode == "illocutions":
        print(eval_illocutions(**args))
    else:
        raise ValueError(f"Unknown mode: {mode}")
