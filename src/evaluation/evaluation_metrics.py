"""How to use the evaluation script:

(1) Make sure that you have installed the following libraries (also specified in requirements.txt):
  - networkx # graphs
  - pycm # confusion matrix
  - segeval # segmentation
  - fuzzywuzzy # fuzzy string matching
  - numpy # used in matching.py
  - bs4 # used in matching.py

(2) Install GMatch4py (needed for graph matching) either directly with pip:
  pip install git+https://github.com/Jacobe2169/GMatch4py.git

  Or by cloning the repository first:
  git clone https://github.com/Jacobe2169/GMatch4py
  cd GMatch4py
  pip install .


(3) Run the script as follows:

$ python3 src/evaluation/evaluation_metrics.py predicted_nodeset_path gold_nodeset_path

For example:
$ python3 src/evaluation/evaluation_metrics.py data/predicted_nodeset1.json data/gold_nodeset1.json

The script prints to the console the following metrics: Kappa, CASS, F1, Accuracy, U-Alpha

You can find more detail about the CASS metric here: https://aclanthology.org/W16-2805.pdf

Note: This script is based on the following code (some adaptations were made to make it work with the DialAM data): https://github.com/arg-tech/AMF-Evaluation-Metrics
"""

import argparse
import json

from matching import match
from pycm import ConfusionMatrix


class evaluation:
    @staticmethod
    def matching_calculation(predicted_data, gold_data):
        json1 = predicted_data
        json2 = gold_data
        # Graph construction
        graph1, graph2 = match.get_graphs(json1, json2)
        # creating proposition similarity matrix relations
        prop_rels = match.get_prop_sim_matrix(graph1, graph2)

        # creating locution similarity matrix relations
        loc_rels = match.get_loc_sim_matrix(graph1, graph2)
        # anchoring on s-nodes (RA/CA/MA) and combining them
        ra_a = match.ra_anchor(graph1, graph2)
        ma_a = match.ma_anchor(graph1, graph2)
        ca_a = match.ca_anchor(graph1, graph2)
        all_a = match.combine_s_node_matrix(ra_a, ca_a, ma_a)
        all_s_a_dict = match.convert_to_dict(all_a)
        # propositional relation comparison
        prop_rels_comp_conf = match.prop_rels_comp(prop_rels, graph1, graph2)
        prop_rels_comp_dict = match.convert_to_dict(prop_rels_comp_conf)
        # getting all YAs anchored in Locutions
        loc_ya_rels_comp_conf = match.loc_ya_rels_comp(loc_rels, graph1, graph2)
        loc_ya_rels_comp_dict = match.convert_to_dict(loc_ya_rels_comp_conf)
        # getting all YAs in propositions
        prop_ya_comp_conf = match.prop_ya_comp(prop_rels, graph1, graph2)
        prop_ya_comp_dict = match.convert_to_dict(prop_ya_comp_conf)
        # getting all TAs anchored in Locutions
        loc_ta_conf = match.loc_ta_rels_comp(loc_rels, graph1, graph2)
        loc_ta_dict = match.convert_to_dict(loc_ta_conf)
        # getting all YAs anchored in propositions
        prop_ya_conf = match.prop_ya_anchor_comp(prop_rels, graph1, graph2)
        prop_ya_dict = match.convert_to_dict(prop_ya_conf)

        # creating confusion matrix for s-nodes/YA/TA
        all_s_a_cm = ConfusionMatrix(matrix=all_s_a_dict)
        prop_rels_comp_cm = ConfusionMatrix(matrix=prop_rels_comp_dict)
        loc_ya_rels_comp_cm = ConfusionMatrix(matrix=loc_ya_rels_comp_dict)
        prop_ya_comp_cm = ConfusionMatrix(matrix=prop_ya_comp_dict)
        loc_ta_cm = ConfusionMatrix(matrix=loc_ta_dict)
        prop_ya_cm = ConfusionMatrix(matrix=prop_ya_dict)

        return (
            all_s_a_cm,
            prop_rels_comp_cm,
            loc_ya_rels_comp_cm,
            prop_ya_comp_cm,
            loc_ta_cm,
            prop_ya_cm,
        )

    # Kappa range from -1 to +1
    @staticmethod
    def kappa_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    ):
        # kappa calculation
        s_node_kapp = all_s_a_cm.Kappa
        prop_rel_kapp = prop_rels_comp_cm.Kappa
        loc_rel_kapp = loc_ya_rels_comp_cm.Kappa
        prop_ya_kapp = prop_ya_comp_cm.Kappa
        loc_ta_kapp = loc_ta_cm.Kappa
        prop_ya_an_kapp = prop_ya_cm.Kappa

        if match.check_none(s_node_kapp):
            s_node_kapp = all_s_a_cm.KappaNoPrevalence
        if match.check_none(prop_rel_kapp):
            prop_rel_kapp = prop_rels_comp_cm.KappaNoPrevalence
        if match.check_none(loc_rel_kapp):
            loc_rel_kapp = loc_ya_rels_comp_cm.KappaNoPrevalence
        if match.check_none(prop_ya_kapp):
            prop_ya_kapp = prop_ya_comp_cm.KappaNoPrevalence
        if match.check_none(loc_ta_kapp):
            loc_ta_kapp = loc_ta_cm.KappaNoPrevalence
        if match.check_none(prop_ya_an_kapp):
            prop_ya_an_kapp = prop_ya_cm.KappaNoPrevalence

        score_list = [
            s_node_kapp,
            prop_rel_kapp,
            loc_rel_kapp,
            prop_ya_kapp,
            loc_ta_kapp,
            prop_ya_an_kapp,
        ]
        k_graph = sum(score_list) / float(len(score_list))

        return k_graph
        # CASS calculation

    @staticmethod
    def text_similarity(predicted_data, gold_data):
        text1 = predicted_data["text"]
        text2 = gold_data["text"]
        # Check if text1 is a dictionary with 'txt' key
        if isinstance(text1, dict) and "txt" in text1:
            text1 = text1["txt"]

        # Check if text2 is a dictionary with 'txt' key
        if isinstance(text2, dict) and "txt" in text2:
            text2 = text2["txt"]
        # Similarity between two texts
        ss = match.get_similarity(text1, text2)
        if (
            ss == "Error Text Input Is Empty"
            or ss == "None:Error! Source Text Was Different as Segmentations differ in length"
        ):
            return ss
        else:
            return ss

    @staticmethod
    def CASS_calculation(text_sim_ss, k_graph):
        if (
            text_sim_ss == "Error Text Input Is Empty"
            or text_sim_ss
            == "None:Error! Source Text Was Different as Segmentations differ in length"
        ):
            overall_sim = "None"
        else:

            # overall_sim = (k_graph + text_sim_ss) / 2
            overall_sim = (float(k_graph) + float(text_sim_ss)) / 2

        return overall_sim

    # f1 0-1
    @staticmethod
    def F1_Macro_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    ):
        # Get F1 macro scores from confusion matrices for each category/class
        s_node_F1_macro = all_s_a_cm.F1_Macro
        prop_rel_F1_macro = prop_rels_comp_cm.F1_Macro
        loc_rel_F1_macro = loc_ya_rels_comp_cm.F1_Macro
        prop_ya_F1_macro = prop_ya_comp_cm.F1_Macro
        loc_ta_F1_macro = loc_ta_cm.F1_Macro
        prop_ya_an_F1_macro = prop_ya_cm.F1_Macro

        if match.check_none(s_node_F1_macro):
            s_node_F1_macro = 1.0
        if match.check_none(prop_rel_F1_macro):
            prop_rel_F1_macro = 1.0
        if match.check_none(loc_rel_F1_macro):
            loc_rel_F1_macro = 1.0
        if match.check_none(prop_ya_F1_macro):
            prop_ya_F1_macro = 1.0
        if match.check_none(loc_ta_F1_macro):
            loc_ta_F1_macro = 1.0
        if match.check_none(prop_ya_an_F1_macro):
            prop_ya_an_F1_macro = 1.0
        score_list = [
            s_node_F1_macro,
            prop_rel_F1_macro,
            loc_rel_F1_macro,
            prop_ya_F1_macro,
            loc_ta_F1_macro,
            prop_ya_an_F1_macro,
        ]

        F1_macro = sum(score_list) / float(len(score_list))

        return F1_macro

    #  accuracy 0-1
    @staticmethod
    def accuracy_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    ):
        # Get accuracy scores from confusion matrices for each category/class
        s_node_accuracy = all_s_a_cm.ACC
        prop_rel_accuracy = prop_rels_comp_cm.ACC
        loc_rel_accuracy = loc_ya_rels_comp_cm.ACC
        prop_ya_accuracy = prop_ya_comp_cm.ACC
        loc_ta_accuracy = loc_ta_cm.ACC
        prop_ya_an_accuracy = prop_ya_cm.ACC

        # Handle cases where accuracy is None
        def handle_accuracy(acc_dict):
            acc_dict = {k: v if v is not None else 1 for k, v in acc_dict.items()}
            return acc_dict

        s_node_accuracy = handle_accuracy(s_node_accuracy)
        prop_rel_accuracy = handle_accuracy(prop_rel_accuracy)
        loc_rel_accuracy = handle_accuracy(loc_rel_accuracy)
        prop_ya_accuracy = handle_accuracy(prop_ya_accuracy)
        loc_ta_accuracy = handle_accuracy(loc_ta_accuracy)
        prop_ya_an_accuracy = handle_accuracy(prop_ya_an_accuracy)

        # Calculate the average accuracy for each class
        def calculate_average_accuracy(acc_dict):
            values = list(acc_dict.values())
            return sum(values) / len(values) if len(values) > 0 else 0

        s_node_accuracy = calculate_average_accuracy(s_node_accuracy)
        prop_rel_accuracy = calculate_average_accuracy(prop_rel_accuracy)
        loc_rel_accuracy = calculate_average_accuracy(loc_rel_accuracy)
        prop_ya_accuracy = calculate_average_accuracy(prop_ya_accuracy)
        loc_ta_accuracy = calculate_average_accuracy(loc_ta_accuracy)
        prop_ya_an_accuracy = calculate_average_accuracy(prop_ya_an_accuracy)

        score_list = [
            s_node_accuracy,
            prop_rel_accuracy,
            loc_rel_accuracy,
            prop_ya_accuracy,
            loc_ta_accuracy,
            prop_ya_an_accuracy,
        ]

        Accuracy = sum(score_list) / float(len(score_list))

        return Accuracy

    # U-Alpha range from 0 to 1
    @staticmethod
    def u_alpha_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    ):
        # u-alpha calculation
        s_node_u_alpha = all_s_a_cm.Alpha
        prop_rel_u_alpha = prop_rels_comp_cm.Alpha
        loc_rel_u_alpha = loc_ya_rels_comp_cm.Alpha
        prop_ya_u_alpha = prop_ya_comp_cm.Alpha
        loc_ta_u_alpha = loc_ta_cm.Alpha
        prop_ya_an_u_alpha = prop_ya_cm.Alpha

        if match.check_none(s_node_u_alpha):
            s_node_u_alpha = 1.0
        if match.check_none(prop_rel_u_alpha):
            prop_rel_u_alpha = 1.0
        if match.check_none(loc_rel_u_alpha):
            loc_rel_u_alpha = 1.0
        if match.check_none(prop_ya_u_alpha):
            prop_ya_u_alpha = 1.0
        if match.check_none(loc_ta_u_alpha):
            loc_ta_u_alpha = 1.0
        if match.check_none(prop_ya_an_u_alpha):
            prop_ya_an_u_alpha = 1.0

        score_list = [
            s_node_u_alpha,
            prop_rel_u_alpha,
            loc_rel_u_alpha,
            prop_ya_u_alpha,
            loc_ta_u_alpha,
            prop_ya_an_u_alpha,
        ]
        u_alpha = sum(score_list) / float(len(score_list))

        return u_alpha


def run_evaluation(predicted_nodeset_path: str, gold_nodeset_path: str):
    # Read the JSON data
    with open(predicted_nodeset_path) as f:
        predicted_data = json.load(f)
    with open(gold_nodeset_path) as f:
        gold_data = json.load(f)
    # Evaluate graphs
    eval = evaluation()
    # Kappa
    (
        all_s_a_cm,
        prop_rels_comp_cm,
        loc_ya_rels_comp_cm,
        prop_ya_comp_cm,
        loc_ta_cm,
        prop_ya_cm,
    ) = eval.matching_calculation(predicted_data, gold_data)

    kappa = eval.kappa_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("kappa", kappa)

    # works only for AIF data structure, in DialAM we do not have "text" field
    # Text Similarity and CASS
    # Text_similarity = eval.text_similarity(predicted_data, gold_data)
    # print("text similarity", Text_similarity)

    CASS = eval.CASS_calculation(1, kappa)
    # CASS = eval.CASS_calculation(Text_similarity, kappa)
    print("CASS", CASS)

    # F1
    F1 = eval.F1_Macro_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("F1", F1)
    # accuracy
    Acc = eval.accuracy_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("Accuracy", Acc)

    # U-Alpha
    U_Alpha = eval.u_alpha_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("U-Alpha", U_Alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predicted_nodeset_path",
        type=str,
        help="path to the predicted nodeset (stored as a graph in JSON format)",
    )
    parser.add_argument(
        "gold_nodeset_path",
        type=str,
        help="path to the gold nodeset (stored as a graph in JSON format)",
    )
    args = vars(parser.parse_args())
    run_evaluation(**args)
