from typing import Any, Dict, List, Tuple

import matching
from pycm import ConfusionMatrix


def matching_calculation(
    predicted_data: Dict[str, List[Dict[str, Any]]], gold_data: Dict[str, List[Dict[str, Any]]]
) -> Tuple[
    ConfusionMatrix,
    ConfusionMatrix,
    ConfusionMatrix,
    ConfusionMatrix,
    ConfusionMatrix,
    ConfusionMatrix,
]:
    """Build confusion matrices for different types of relations.
    Args:
        predicted_data: A dictionary with nodes, edges and locutions for the predicted nodeset.
        gold_data: A dictionary with nodes, edges and locutions for the gold nodeset.

    Returns:
        Confusion matrices for the following transitions:
            all_s_a_cm: YA > S (S-nodes: MA, RA, CA)
            prop_rels_comp_cm: I > S (S-nodes: MA, RA, CA)
            loc_ya_rels_comp_cm: L > YA
            prop_ya_comp_cm: YA > I
            loc_ta_cm: L > TA
            prop_ya_cm: L > (YA) > I

    """
    # Graph construction.
    graph1, graph2 = matching.get_graphs(predicted_data, gold_data)
    # Creating similarity matrix for propositional relations.
    prop_rels = matching.get_sim_matrix(graph1, graph2, "propositions")
    # Creating similarity matrix for locutional relations.
    loc_rels = matching.get_sim_matrix(graph1, graph2, "locutions")
    # (1) Do we have the same YA > S transitions? Do their annotations coincide?
    # Anchoring on S-nodes (RA/CA/MA) and combining them (checking how many YA-anchors,
    # predecessors of S-nodes, have the same/different text field annotations).
    ra_a = matching.s_rel_anchor("RA", graph1, graph2)
    ma_a = matching.s_rel_anchor("MA", graph1, graph2)
    ca_a = matching.s_rel_anchor("CA", graph1, graph2)
    all_a = matching.combine_s_node_matrix(ra_a, ca_a, ma_a)
    all_s_a_dict = matching.convert_to_dict(all_a)
    # (2) Do we have the same S-nodes?
    # Comparing propositional relations, building a confusion matrix for S-node (RA, MA, CA)
    # matches between the two graphs.
    prop_rels_comp_conf = matching.prop_rels_comp(prop_rels, graph1, graph2)
    prop_rels_comp_dict = matching.convert_to_dict(prop_rels_comp_conf)
    # (3) Do we have the same L > YA transitions? Do their annotations coincide?
    # Getting all YAs anchored in locutions, comparing the text field annotations of YA-nodes,
    # successors of L-nodes, anchored in locutions.
    loc_ya_rels_comp_conf = matching.loc_ya_rels_comp(loc_rels, graph1, graph2)
    loc_ya_rels_comp_dict = matching.convert_to_dict(loc_ya_rels_comp_conf)
    # (4) Do we have the same YA > I transitions? Do their annotations coincide?
    # Getting all YAs in propositions, comparing the text field annotations of YA-nodes,
    # predecessors of I-nodes, anchored in propositions.
    prop_ya_comp_conf = matching.prop_ya_comp(prop_rels, graph1, graph2)
    prop_ya_comp_dict = matching.convert_to_dict(prop_ya_comp_conf)
    # (5) Do we have the same (TA-node) transitions between the L-nodes (L > TA > L transitions)?
    # Getting all TAs anchored in locutions, checking whether each L-node pair (from graph1 and graph2)
    # has the same amount of outgoing TA-nodes.
    loc_ta_conf = matching.loc_ta_rels_comp(loc_rels, graph1, graph2)
    loc_ta_dict = matching.convert_to_dict(loc_ta_conf)
    # (6) Do we have I-nodes anchored in the same L-nodes (L > YA > I transitions)? Do their annotations coincide?
    # Getting all YAs anchored in propositions, comparing the text field annotations of L-/TA-nodes
    # that are predecessors of YA-nodes anchored in propositions (I-nodes).
    prop_ya_conf = matching.prop_ya_anchor_comp(prop_rels, graph1, graph2)
    prop_ya_dict = matching.convert_to_dict(prop_ya_conf)

    # Creating confusion matrix for S-nodes/YA/TA.
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


def kappa_calculation(
    all_s_a_cm: ConfusionMatrix,
    prop_rels_comp_cm: ConfusionMatrix,
    loc_ya_rels_comp_cm: ConfusionMatrix,
    prop_ya_comp_cm: ConfusionMatrix,
    loc_ta_cm: ConfusionMatrix,
    prop_ya_cm: ConfusionMatrix,
) -> float:
    """Calculate Kappa values (in range between -1 and 1).
    Args:
        Confusion matrices for the following transitions:
            all_s_a_cm: YA > S (S-nodes: MA, RA, CA)
            prop_rels_comp_cm: I > S (S-nodes: MA, RA, CA)
            loc_ya_rels_comp_cm: L > YA
            prop_ya_comp_cm: YA > I
            loc_ta_cm: L > TA
            prop_ya_cm: L > (YA) > I

    Returns:
        Kappa value based on graph matching.
    """

    s_node_kapp = all_s_a_cm.Kappa
    prop_rel_kapp = prop_rels_comp_cm.Kappa
    loc_rel_kapp = loc_ya_rels_comp_cm.Kappa
    prop_ya_kapp = prop_ya_comp_cm.Kappa
    loc_ta_kapp = loc_ta_cm.Kappa
    prop_ya_an_kapp = prop_ya_cm.Kappa

    if s_node_kapp == "None":
        s_node_kapp = all_s_a_cm.KappaNoPrevalence
    if prop_rel_kapp == "None":
        prop_rel_kapp = prop_rels_comp_cm.KappaNoPrevalence
    if loc_rel_kapp == "None":
        loc_rel_kapp = loc_ya_rels_comp_cm.KappaNoPrevalence
    if prop_ya_kapp == "None":
        prop_ya_kapp = prop_ya_comp_cm.KappaNoPrevalence
    if loc_ta_kapp == "None":
        loc_ta_kapp = loc_ta_cm.KappaNoPrevalence
    if prop_ya_an_kapp == "None":
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


def text_similarity(
    predicted_data: Dict[str, List[Dict[str, Any]]], gold_data: Dict[str, List[Dict[str, Any]]]
) -> float:
    """Compute similarity between the segemented texts.
    Args:
        predicted_data: A dictionary with nodes, edges and locutions for the predicted nodeset.
        gold_data: A dictionary with nodes, edges and locutions for the gold nodeset.

    Returns:
        Segmentation similarity for texts from two different graphs.
    """

    text1 = predicted_data["text"]
    text2 = gold_data["text"]
    # Check if text1 is a dictionary with 'txt' key
    if isinstance(text1, dict) and "txt" in text1:
        text1 = text1["txt"]

    # Check if text2 is a dictionary with 'txt' key
    if isinstance(text2, dict) and "txt" in text2:
        text2 = text2["txt"]
    # Similarity between two texts
    ss = matching.get_similarity(text1, text2)
    return ss


def CASS_calculation(text_sim_ss: float, k_graph: float) -> float:
    """Compute the CASS metric: https://aclanthology.org/W16-2805.pdf
    Args:
        text_sim_ss: Text segmentation similarity.
        k_graph: Graph matching similarity.

    Returns:
        CASS metric (between 0 and 1).
    """
    if text_sim_ss > 0:
        return (float(k_graph) + float(text_sim_ss)) / 2
    else:
        return 0.0


def F1_Macro_calculation(
    all_s_a_cm: ConfusionMatrix,
    prop_rels_comp_cm: ConfusionMatrix,
    loc_ya_rels_comp_cm: ConfusionMatrix,
    prop_ya_comp_cm: ConfusionMatrix,
    loc_ta_cm: ConfusionMatrix,
    prop_ya_cm: ConfusionMatrix,
) -> float:
    """Compute macro F1 score based on the confusion matrices for each type of relations.
    Args:
        Confusion matrices for the following transitions:
            all_s_a_cm: YA > S (S-nodes: MA, RA, CA)
            prop_rels_comp_cm: I > S (S-nodes: MA, RA, CA)
            loc_ya_rels_comp_cm: L > YA
            prop_ya_comp_cm: YA > I
            loc_ta_cm: L > TA
            prop_ya_cm: L > (YA) > I

    Returns:
        Macro F1 score.
    """
    s_node_F1_macro = all_s_a_cm.F1_Macro
    prop_rel_F1_macro = prop_rels_comp_cm.F1_Macro
    loc_rel_F1_macro = loc_ya_rels_comp_cm.F1_Macro
    prop_ya_F1_macro = prop_ya_comp_cm.F1_Macro
    loc_ta_F1_macro = loc_ta_cm.F1_Macro
    prop_ya_an_F1_macro = prop_ya_cm.F1_Macro

    if s_node_F1_macro == "None":
        s_node_F1_macro = 1.0
    if prop_rel_F1_macro == "None":
        prop_rel_F1_macro = 1.0
    if loc_rel_F1_macro == "None":
        loc_rel_F1_macro = 1.0
    if prop_ya_F1_macro == "None":
        prop_ya_F1_macro = 1.0
    if loc_ta_F1_macro == "None":
        loc_ta_F1_macro = 1.0
    if prop_ya_an_F1_macro == "None":
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


def accuracy_calculation(
    all_s_a_cm: ConfusionMatrix,
    prop_rels_comp_cm: ConfusionMatrix,
    loc_ya_rels_comp_cm: ConfusionMatrix,
    prop_ya_comp_cm: ConfusionMatrix,
    loc_ta_cm: ConfusionMatrix,
    prop_ya_cm: ConfusionMatrix,
) -> float:
    """Compute accuracy score based on the confusion matrices for each type of relations.
    Args:
        Confusion matrices for the following transitions:
            all_s_a_cm: YA > S (S-nodes: MA, RA, CA)
            prop_rels_comp_cm: I > S (S-nodes: MA, RA, CA)
            loc_ya_rels_comp_cm: L > YA
            prop_ya_comp_cm: YA > I
            loc_ta_cm: L > TA
            prop_ya_cm: L > (YA) > I

    Returns:
        Average accuracy.
    """
    # Get accuracy scores from confusion matrices for each category/class.
    s_node_accuracy = all_s_a_cm.ACC
    prop_rel_accuracy = prop_rels_comp_cm.ACC
    loc_rel_accuracy = loc_ya_rels_comp_cm.ACC
    prop_ya_accuracy = prop_ya_comp_cm.ACC
    loc_ta_accuracy = loc_ta_cm.ACC
    prop_ya_an_accuracy = prop_ya_cm.ACC

    # Handle cases where accuracy is None.
    def handle_accuracy(acc_dict):
        acc_dict = {k: v if v is not None else 1 for k, v in acc_dict.items()}
        return acc_dict

    s_node_accuracy = handle_accuracy(s_node_accuracy)
    prop_rel_accuracy = handle_accuracy(prop_rel_accuracy)
    loc_rel_accuracy = handle_accuracy(loc_rel_accuracy)
    prop_ya_accuracy = handle_accuracy(prop_ya_accuracy)
    loc_ta_accuracy = handle_accuracy(loc_ta_accuracy)
    prop_ya_an_accuracy = handle_accuracy(prop_ya_an_accuracy)

    # Calculate the average accuracy for each class.
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


def u_alpha_calculation(
    all_s_a_cm: ConfusionMatrix,
    prop_rels_comp_cm: ConfusionMatrix,
    loc_ya_rels_comp_cm: ConfusionMatrix,
    prop_ya_comp_cm: ConfusionMatrix,
    loc_ta_cm: ConfusionMatrix,
    prop_ya_cm: ConfusionMatrix,
) -> float:
    """Compute U-Alpha score based on the confusion matrices for each type of relations.
    Args:
        Confusion matrices for the following transitions:
            all_s_a_cm: YA > S (S-nodes: MA, RA, CA)
            prop_rels_comp_cm: I > S (S-nodes: MA, RA, CA)
            loc_ya_rels_comp_cm: L > YA
            prop_ya_comp_cm: YA > I
            loc_ta_cm: L > TA
            prop_ya_cm: L > (YA) > I

    Returns:
        U-Alpha score (between 0 and 1).
    """
    # u-alpha calculation
    s_node_u_alpha = all_s_a_cm.Alpha
    prop_rel_u_alpha = prop_rels_comp_cm.Alpha
    loc_rel_u_alpha = loc_ya_rels_comp_cm.Alpha
    prop_ya_u_alpha = prop_ya_comp_cm.Alpha
    loc_ta_u_alpha = loc_ta_cm.Alpha
    prop_ya_an_u_alpha = prop_ya_cm.Alpha

    if s_node_u_alpha == "None":
        s_node_u_alpha = 1.0
    if prop_rel_u_alpha == "None":
        prop_rel_u_alpha = 1.0
    if loc_rel_u_alpha == "None":
        loc_rel_u_alpha = 1.0
    if prop_ya_u_alpha == "None":
        prop_ya_u_alpha = 1.0
    if loc_ta_u_alpha == "None":
        loc_ta_u_alpha = 1.0
    if prop_ya_an_u_alpha == "None":
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
