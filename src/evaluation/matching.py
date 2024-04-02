import copy
import logging
from typing import Any, Dict, List, Tuple

import centrality
import load_map
import numpy as np
import segeval
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from networkx.classes.digraph import DiGraph
from numpy import unravel_index

logger = logging.getLogger(__name__)


def get_graphs(
    dt1: Dict[str, List[Dict[str, str]]], dt2: Dict[str, List[Dict[str, str]]]
) -> Tuple[DiGraph, DiGraph]:
    """Load the graphs from the corpus, remove isolated nodes."""

    # load graphs
    graph1 = load_map.parse_json(dt1)
    graph2 = load_map.parse_json(dt2)
    # remove isolated nodes
    graph1 = centrality.remove_iso_analyst_nodes(graph1)
    graph2 = centrality.remove_iso_analyst_nodes(graph2)
    return graph1, graph2


def get_similarity(text_1: str, text_2: str) -> float:
    """
    Compute segmentation similarity between two input texts.
    Args:
        text_1: Input text 1.
        text_2: Input text 2.

    Both text_1 and text_2 are xml data that use spans to separate boundaries
    e.g. BOSTON, MA ... <span class="highlighted" id="634541">Steven L.
    Davis pled guilty yesterday to federal charges that he stole and disclosed trade secrets of The Gillette Company</span>.

    Returns: segmentation similarity score between 0 and 1.

    """

    ss = 0.0  # segmentation similarity
    if text_1 == "" or text_2 == "":
        raise Exception("Text Input Is Empty")
    else:
        # Normalize text
        chars_to_remove = [
            "`",
            "’",
            "'",
        ]
        chars_to_replace_with_space = ["  ", "...", "…", ".", ",", "!", "?", "  "]

        for ch in chars_to_remove:
            text_1 = text_1.replace(ch, "")
            text_2 = text_2.replace(ch, "")

        text_1 = text_1.strip()
        text_2 = text_2.strip()

        text_1 = text_1.replace("[", " [")
        text_2 = text_2.replace("[", " [")
        text_1 = text_1.replace("]", "] ")
        text_2 = text_2.replace("]", "] ")

        for ch in chars_to_replace_with_space:
            text_1 = text_1.replace(ch, " ")
            text_2 = text_2.replace(ch, " ")

        # Parse text using BeautifulSoup
        xml_soup_1 = BeautifulSoup(text_1, features="lxml")
        xml_soup_2 = BeautifulSoup(text_2, features="lxml")

        # Remove unwanted HTML tags
        xml_soup_1 = remove_html_tags(xml_soup_1)
        # xml_soup_1 = BeautifulSoup(str(xml_soup_1), features="lxml").text

        xml_soup_2 = remove_html_tags(xml_soup_2)
        # xml_soup_2 = BeautifulSoup(str(xml_soup_2), features="lxml").text

        # Get segments
        segments_1, words_1 = get_segements(xml_soup_1)
        segments_2, words_2 = get_segements(xml_soup_2)

        # Check segment length
        seg_check, seg1, seg2 = check_segment_length(segments_1, words_1, segments_2, words_2)

        if not seg_check:
            raise Exception("Source text was different as segmentations differ in length.")
        else:
            if seg1 == seg2:
                ss = 1.0  # if segmentation sequences are identical, we set similarity to maximum (1.0)
            else:
                # Convert segments to masses
                masses_1 = segeval.convert_positions_to_masses(seg1)
                masses_2 = segeval.convert_positions_to_masses(seg2)

                # Calculate segmentation similarity
                ss = segeval.segmentation_similarity(masses_1, masses_2)
    return ss


def remove_html_tags(xml_soup: BeautifulSoup) -> BeautifulSoup:
    """Remove HTML tags from the XML-formatted document."""
    for match in xml_soup.findAll("div"):
        match.replaceWithChildren()
    for match in xml_soup.findAll("p"):
        match.replaceWithChildren()
    for match in xml_soup.findAll("br"):
        match.replaceWithChildren()
    return xml_soup


def get_segements(xml_soup: BeautifulSoup) -> Tuple[List[int], List[str]]:
    """Retrieve segments and words from XML-formatted document."""
    segment_list = []
    word_list = []
    if xml_soup.body:
        xml_soup_body = xml_soup.body
    else:
        xml_soup_body = xml_soup
    for i, tag in enumerate(xml_soup_body):
        boundary_counter = i + 1
        tag_text = ""
        if "span" in str(tag):
            tag_text = tag.text
        else:
            tag_text = str(tag)
        words = tag_text.split()
        seg_len = len(words)
        segment_list += seg_len * [boundary_counter]
        word_list += words
    return segment_list, word_list


def check_segment_length(
    seg_1: List[int], word_1: List[str], seg_2: List[int], word_2: List[str]
) -> Tuple[bool, Any, Any]:
    """Check whether both segments have the same length and return the segmentation."""
    seg_1_len = len(seg_1)
    seg_2_len = len(seg_2)
    if seg_1_len == seg_2_len:
        return True, seg_1, seg_2
    else:
        if seg_1_len > seg_2_len:
            for i in range(len(word_1) - 1):
                if word_1[i] + word_1[i + 1] in word_2:
                    word_1[i : i + 2] = [word_1[i] + word_1[i + 1]]

                    seg_1 = [seg_1[0]] * len(word_1)
                    if len(seg_1) == len(seg_2):
                        return True, seg_1, seg_2
            return False, None, None
        else:
            for i in range(len(word_2) - 1):
                if word_2[i] + word_2[i + 1] in word_1:
                    word_2[i : i + 2] = [word_2[i] + word_2[i + 1]]

                    seg_2 = [seg_2[0]] * len(word_2)
                    if len(seg_1) == len(seg_2):
                        return True, seg_1, seg_2
            return False, None, None


def rels_to_dict(
    rels: List[Tuple[Tuple[int, str], Tuple[int, str]]], switched: bool
) -> List[Dict[str, Any]]:
    """Convert relations to a list of dictionaries where each dictionary includes both node IDs and
    node text annotations for each pair of nodes in relation."""
    new_list = []
    for rel in rels:
        id_1 = rel[0][0]
        id_2 = rel[1][0]
        text_1 = rel[0][1]
        text_2 = rel[1][1]
        # If switched is True the graphs were reversed.
        if switched:
            mat_dict = {"ID1": id_2, "ID2": id_1, "text1": text_2, "text2": text_1}
        else:
            mat_dict = {"ID1": id_1, "ID2": id_2, "text1": text_1, "text2": text_2}
        new_list.append(mat_dict)
    return new_list


def get_sim_matrix(graph1: DiGraph, graph2: DiGraph, rel_type: str) -> List[Dict[str, Any]]:
    """Create similarity matrix for propositional or locutional relations (w/o similarity
    values)."""

    g1_copy = graph1.copy()
    g2_copy = graph2.copy()
    g1_inodes = []
    g2_inodes = []

    if rel_type == "propositions":
        g1_inodes = centrality.get_i_node_list(g1_copy)
        g2_inodes = centrality.get_i_node_list(g2_copy)
    elif rel_type == "locutions":
        g1_inodes = centrality.get_l_node_list(g1_copy)
        g2_inodes = centrality.get_l_node_list(g2_copy)
    else:
        raise Exception(
            f"Unknown relation type: {rel_type}. Must be either propositions or locutions."
        )

    relsi, valsi, switched = text_sim_matrix(g1_inodes, g2_inodes)
    # If switched is True, the relations have been in a switched order, so they need to be reversed when creating the dictionary.
    rels_dict = rels_to_dict(relsi, switched)
    return rels_dict


def text_sim_matrix(
    g1_list: List[Tuple[int, str]], g2_list: List[Tuple[int, str]]
) -> Tuple[
    List[Tuple[Tuple[int, str], Tuple[int, str]]],
    List[Tuple[Tuple[int, str], Tuple[int, str]]],
    bool,
]:
    """Compute text similarity based on two lists of nodes coming from two different graphs."""
    g1_size = len(g1_list)
    g2_size = len(g2_list)

    switch_flag = False  # check whether relation is in a reversed order

    if g1_size >= g2_size:
        mat = loop_nodes(g1_list, g2_list)
        rels, vals = select_max_vals(mat, g2_size, g1_list, g2_list)
    else:
        switch_flag = True
        mat = loop_nodes(g2_list, g1_list)
        rels, vals = select_max_vals(mat, g1_size, g2_list, g1_list)
    return rels, vals, switch_flag


def loop_nodes(g1_list: List[Tuple[int, str]], g2_list: List[Tuple[int, str]]):
    """Compute paiwise similarity (for nodes texts) based on fuzzy string matching."""
    matrix = np.zeros((len(g1_list), len(g2_list)))
    for i1, node1 in enumerate(g1_list):
        text1 = node1[1].lower()
        for i2, node2 in enumerate(g2_list):
            text2 = node2[1].lower()
            # lev_val = normalized_levenshtein.distance(text1, text2)
            lev_val = (fuzz.ratio(text1, text2)) / 100
            matrix[i1][i2] = lev_val
    return matrix


def select_max_vals(
    matrix: Any,
    smallest_value: int,
    g1_list: List[Tuple[int, str]],
    g2_list: List[Tuple[int, str]],
):
    """Find maximum values and corresponding relations in the similarity matrix (g2_list is the
    shortest list)."""
    counter = 0
    lev_vals = []
    lev_rels = []
    index_list = list(range(len(g1_list)))
    m_copy = copy.deepcopy(matrix)
    while counter <= smallest_value - 1:
        index_tup = unravel_index(m_copy.argmax(), m_copy.shape)
        row_i = index_tup[0]
        col_i = index_tup[1]
        m_copy[row_i] = 0  # zeroes out row i
        m_copy[:, col_i] = 0  # zeroes out column i
        lev_rels.append((g1_list[row_i], g2_list[col_i]))
        lev_vals.append(matrix[row_i][col_i])
        index_list.remove(row_i)
        counter += 1
    # Fill the rest of the (non-aligned) positions with 0s, note that g1_list is always longer than g2_list.
    for vals in index_list:
        lev_rels.append((g1_list[vals], (0, "")))
        lev_vals.append(0)
    return lev_rels, lev_vals


def convert_to_dict(conf_matrix: List[List[int]]) -> Dict[int, Dict[int, int]]:
    """Convert confusion matrix to a dictionary."""
    dicts: Dict[int, Dict[int, int]] = dict()
    for i, col in enumerate(conf_matrix):
        dicts[i] = dict()
        for j, row in enumerate(col):
            dicts[i][j] = row
    return dicts


def s_rel_anchor(rel_type: str, graph1: DiGraph, graph2: DiGraph) -> List[List[int]]:
    """Create a confusion matrix for S-nodes of type CA, MA or RA.

    Each S-node is anchored in the corresponding YA-node. Confusion matrix shows how many YA-
    anchors are the same (or different) for all S-node pairs coming from two different graphs.
    """
    conf_matrix = [[0, 0], [0, 0]]
    rel1 = centrality.get_rels(rel_type, graph1)
    rel2 = centrality.get_rels(rel_type, graph2)

    rel1_len = len(rel1)
    rel2_len = len(rel2)

    if rel1_len > 0 and rel2_len > 0:
        if rel1_len > rel2_len:
            for rel_i, rel in enumerate(rel1):
                rel2_id = ""
                yas1 = get_ya_node_text_from_prop(rel, graph1)
                try:
                    rel2_id = rel2[rel_i]
                except Exception as e:
                    rel2_id = ""
                    logger.error(
                        f"Failed to find index {rel_i} in {rel2} (relation type {rel_type}): {e}"
                    )
                if rel2_id == "":
                    conf_matrix[1][0] += 1
                else:
                    yas2 = get_ya_node_text_from_id(int(rel2_id), graph2)
                    if yas1 == yas2:
                        conf_matrix[0][0] += 1
                    else:
                        conf_matrix[1][0] += 1
        elif rel2_len > rel1_len:
            for rel_i, rel in enumerate(rel2):
                rel1_id = ""
                yas2 = get_ya_node_text_from_prop(rel, graph2)
                try:
                    rel1_id = rel1[rel_i]
                except Exception as e:
                    rel1_id = ""
                    logger.error(
                        f"Failed to find index {rel_i} in {rel1} (relation type {rel_type}): {e}"
                    )
                if rel1_id == "":
                    conf_matrix[0][1] = conf_matrix[0][1] + 1
                else:
                    yas1 = get_ya_node_text_from_id(int(rel1_id), graph1)
                    if yas1 == yas2:
                        conf_matrix[0][0] += 1
                    else:
                        conf_matrix[0][1] += 1
        else:
            for rel_i, rel in enumerate(rel1):
                ya1 = get_ya_node_text_from_prop(rel, graph1)
                try:
                    rel2_id = rel2[rel_i]
                except Exception as e:
                    rel2_id = ""
                    logger.error(
                        f"Failed to find index {rel_i} in {rel2} (relation type {rel_type}): {e}"
                    )
                if rel2_id == "":
                    conf_matrix[1][0] += 1
                else:
                    ya2 = get_ya_node_text_from_prop(int(rel2_id), graph2)
                    if ya1 == ya2:
                        conf_matrix[0][0] += 1
                    else:
                        conf_matrix[1][0] += 1

    elif rel1_len == 0 and rel2_len == 0:
        conf_matrix[1][1] += 1
    elif rel1_len == 0:
        conf_matrix[0][1] += rel2_len
    elif rel2_len == 0:
        conf_matrix[1][0] += rel1_len
    return conf_matrix


def combine_s_node_matrix(
    ra: List[List[int]], ca: List[List[int]], ma: List[List[int]]
) -> List[List[int]]:
    """Combine the confusion matrices for all S-nodes (RA, CA, MA)."""
    result = [[ra[i][j] + ca[i][j] for j in range(len(ra[0]))] for i in range(len(ra))]

    all_result = [
        [ma[i][j] + result[i][j] for j in range(len(result[0]))] for i in range(len(result))
    ]
    return all_result


def count_s_nodes(node_id: int, graph: DiGraph) -> Tuple[int, int, int]:
    """Count how many S-nodes of each type (RA, CA, MA) we have in the graph."""
    RA_count = 0
    MA_count = 0
    CA_count = 0
    try:
        s_nodes = list(
            graph.predecessors(node_id)
        )  # TODO: Why do we consider *only* predecessors? In principle, RA-nodes can also point down, i.e., RA-node could be in successors of the node with the current node_id!
    except Exception as e:
        logger.error(f"Failed to get predecessors for node with ID {node_id}")
        s_nodes = []
    for s in s_nodes:
        n_type = graph.nodes[s]["type"]
        if n_type == "RA":
            RA_count += 1
        elif n_type == "CA":
            CA_count += 1
        elif n_type == "MA":
            MA_count += 1
    return RA_count, CA_count, MA_count


def count_ta_nodes(node_id: int, graph: DiGraph) -> int:
    """Count how many successor TA-nodes we have in the graph."""
    TA_count = 0
    try:
        ta_nodes = list(graph.successors(node_id))
        for ta in ta_nodes:
            n_type = graph.nodes[ta]["type"]
            if n_type == "TA":
                TA_count = TA_count + 1
    except Exception as e:
        logger.error(f"Failed to get successors for node with ID {node_id}")
    return TA_count


def prop_rels_comp(
    prop_matrix: List[Dict[str, Any]], graph1: DiGraph, graph2: DiGraph
) -> List[List[int]]:
    """Create a confusion matrix for propositional relations."""
    conf_matrix = [[0, 0], [0, 0]]

    # Check the matching S-nodes (MA, RA, CA) between the two graphs.
    for rel_dict in prop_matrix:
        ID1 = rel_dict["ID1"]
        ID2 = rel_dict["ID2"]
        text1 = rel_dict["text1"]
        text2 = rel_dict["text2"]

        if ID1 != 0 and ID2 != 0:
            ras1, cas1, mas1 = count_s_nodes(ID1, graph1)
            ras2, cas2, mas2 = count_s_nodes(ID2, graph2)
            for s_rel1, s_rel2 in zip([ras1, cas1, mas1], [ras2, cas2, mas2]):
                if s_rel1 == s_rel2:
                    conf_matrix[0][0] += 1
                elif s_rel1 > s_rel2:
                    conf_matrix[1][0] += 1
                elif s_rel2 > s_rel1:
                    conf_matrix[0][1] += 1

        elif ID1 == 0 and ID2 == 0:
            conf_matrix[1][1] += 1
        elif ID1 == 0:
            conf_matrix[0][1] += 1
        elif ID2 == 0:
            conf_matrix[1][0] += 1

    overallRelations = len(prop_matrix) * len(prop_matrix)

    total_agreed_none = (
        overallRelations - conf_matrix[0][0] - conf_matrix[0][1] - conf_matrix[1][0]
    )
    if total_agreed_none < 0:
        total_agreed_none = 0
    conf_matrix[1][1] = total_agreed_none
    return conf_matrix


def loc_ya_rels_comp(
    loc_matrix: List[Dict[str, Any]], graph1: DiGraph, graph2: DiGraph
) -> List[List[int]]:
    """Create a confusion matrix for locutional relations."""
    all_ya_text = get_ya_node_texts(graph1, graph2)
    conf_matrix = [[0 for x in range(len(all_ya_text) + 1)] for y in range(len(all_ya_text) + 1)]
    all_ya_text.append("")

    # Get all YAs anchored in locutions (L-nodes).
    for rel_dict in loc_matrix:
        ID1 = rel_dict["ID1"]
        ID2 = rel_dict["ID2"]
        text1 = rel_dict["text1"]
        text2 = rel_dict["text2"]

        # Check the YA-node annotations (e.g., "Asserting", "Questioning" etc.)
        # ID equals 0 when L-node from one graph could not be aligned to any L-node from the other graph.
        if ID1 != 0 and ID2 != 0:
            yas1 = get_ya_node_text_from_id(ID1, graph1)
            yas2 = get_ya_node_text_from_id(ID2, graph2)
            assert yas1 in all_ya_text, f"YA-node {yas1} must be in all_ya_text: {all_ya_text}"
            assert yas2 in all_ya_text, f"YA-node {yas2} must be in all_ya_text: {all_ya_text}"
            if yas1 == yas2:
                index = all_ya_text.index(yas1)
                conf_matrix[index][index] += 1
            else:
                index1 = all_ya_text.index(yas1)
                index2 = all_ya_text.index(yas2)
                conf_matrix[index2][index1] += 1
        elif ID1 == 0 and ID2 == 0:
            conf_matrix[len(all_ya_text) - 1][len(all_ya_text) - 1] += 1
        elif ID1 == 0:
            yas2 = get_ya_node_text_from_id(ID2, graph2)
            index = all_ya_text.index(yas2)
            conf_matrix[len(all_ya_text) - 1][index] += 1
        elif ID2 == 0:
            yas1 = get_ya_node_text_from_id(ID1, graph1)
            index = all_ya_text.index(yas1)
            conf_matrix[index][len(all_ya_text) - 1] += 1

        # Get all YA-nodes anchored in transitions (TA-nodes) via locutions (L-nodes) - we only want to loop the matrix once.
        conf_matrix = get_ta_locs(ID1, ID2, graph1, graph2, conf_matrix, all_ya_text)
    return conf_matrix


def update_conf_matrix_tas_in_one_graph(
    tas: List[int],
    graph: DiGraph,
    conf_matrix: List[List[int]],
    all_ya_text_ext: List[str],
    reverse_idx: bool = False,
):
    for ta in tas:
        yas = get_ya_node_text_from_id(ta, graph)
        if yas == "":
            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] += 1
        elif yas in all_ya_text_ext:
            index = all_ya_text_ext.index(yas)
            if reverse_idx:
                conf_matrix[index][len(all_ya_text_ext) - 1] += 1
            else:
                conf_matrix[len(all_ya_text_ext) - 1][index] += 1


def update_conf_matrix_tas_in_both_graphs(
    tas1: List[int],
    graph1: DiGraph,
    tas2: List[int],
    graph2: DiGraph,
    conf_matrix: List[List[int]],
    all_ya_text_ext: List[str],
    reverse_idx: bool = False,
):
    for tai, ta in enumerate(tas1):
        tas2_id = -1
        yas1 = get_ya_node_text_from_id(ta, graph1)
        try:
            tas2_id = tas2[tai]
        except Exception as e:
            tas2_id = -1
            logger.error(f"Failed to find index tai {tai} in tas2 {tas2}: {e}")

        if tas2_id == -1 and yas1 in all_ya_text_ext:
            index = all_ya_text_ext.index(yas1)
            if reverse_idx:
                conf_matrix[len(all_ya_text_ext) - 1][index] += 1
            else:
                conf_matrix[index][len(all_ya_text_ext) - 1] += 1
        elif tas2_id != -1:
            # Check YA-node text (annotation): "Arguing" etc.
            yas2 = get_ya_node_text_from_id(tas2_id, graph2)
            if yas1 == yas2 and yas1 in all_ya_text_ext:
                index = all_ya_text_ext.index(yas1)
                conf_matrix[index][index] += 1
            elif yas1 in all_ya_text_ext and yas2 in all_ya_text_ext:
                index1 = all_ya_text_ext.index(yas1)
                index2 = all_ya_text_ext.index(yas2)
                if reverse_idx:
                    conf_matrix[index1][index2] += 1
                else:
                    conf_matrix[index2][index1] += 1


def get_ta_locs(
    ID1: int,
    ID2: int,
    graph1: DiGraph,
    graph2: DiGraph,
    conf_matrix: List[List[int]],
    all_ya_text: List[str],
) -> List[List[int]]:
    """Create confusion matrix for transitions between the locutions (TA-nodes)."""
    all_ya_text_ext = copy.deepcopy(all_ya_text)
    if ID1 != 0 and ID2 != 0:
        tas1 = get_ta_nodes_from_id(ID1, graph1)
        tas2 = get_ta_nodes_from_id(ID2, graph2)

        if len(tas1) > 0 and len(tas2) > 0:
            if len(tas1) > len(tas2):
                update_conf_matrix_tas_in_both_graphs(
                    tas1,
                    graph1,
                    tas2,
                    graph2,
                    conf_matrix,
                    all_ya_text_ext,
                    reverse_idx=False,
                )
            elif len(tas2) > len(tas1):
                update_conf_matrix_tas_in_both_graphs(
                    tas2,
                    graph2,
                    tas1,
                    graph1,
                    conf_matrix,
                    all_ya_text_ext,
                    reverse_idx=True,
                )
            else:
                for tai, ta in enumerate(tas1):
                    yas1 = get_ya_node_text_from_id(ta, graph1)
                    yas2 = get_ya_node_text_from_id(tas2[tai], graph2)
                    if yas1 == yas2 and yas1 in all_ya_text_ext:
                        index = all_ya_text_ext.index(yas1)
                        conf_matrix[index][index] += 1
                    elif yas1 in all_ya_text_ext and yas2 in all_ya_text_ext:
                        index1 = all_ya_text_ext.index(yas1)
                        index2 = all_ya_text_ext.index(yas2)
                        conf_matrix[index2][index1] += 1

        elif len(tas1) > 0 and len(tas2) < 1:
            update_conf_matrix_tas_in_one_graph(
                tas1, graph1, conf_matrix, all_ya_text_ext, reverse_idx=True
            )

        elif len(tas2) > 0 and len(tas1) < 1:
            update_conf_matrix_tas_in_one_graph(
                tas2, graph2, conf_matrix, all_ya_text_ext, reverse_idx=False
            )

        elif len(tas1) < 1 and len(tas2) < 1:
            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] += 1

    elif ID1 == 0:
        tas2 = get_ta_nodes_from_id(ID2, graph2)
        if len(tas2) > 0:
            update_conf_matrix_tas_in_one_graph(
                tas2, graph2, conf_matrix, all_ya_text_ext, reverse_idx=False
            )
        elif len(tas2) < 1:
            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] += 1

    elif ID2 == 0:
        tas1 = get_ta_nodes_from_id(ID1, graph1)
        if len(tas1) > 0:
            update_conf_matrix_tas_in_one_graph(
                tas1, graph1, conf_matrix, all_ya_text_ext, reverse_idx=True
            )
        elif len(tas1) < 1:
            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] += 1
    return conf_matrix


def prop_ya_comp(prop_matrix: List[Dict[str, Any]], graph1: DiGraph, graph2: DiGraph):
    """Create confusion matrix for YA-nodes: check their text annotations that can be "Restating",
    "Asserting" etc."""
    all_ya_text = get_ya_node_texts(graph1, graph2)
    conf_matrix = [[0 for x in range(len(all_ya_text) + 1)] for y in range(len(all_ya_text) + 1)]
    all_ya_text.append("")
    for rel_dict in prop_matrix:
        ID1 = rel_dict["ID1"]
        ID2 = rel_dict["ID2"]
        text1 = rel_dict["text1"]
        text2 = rel_dict["text2"]

        if ID1 != 0 and ID2 != 0:
            yas1 = get_ya_node_text_from_prop(ID1, graph1)
            yas2 = get_ya_node_text_from_prop(ID2, graph2)

            if yas1 == yas2:
                index = all_ya_text.index(yas1)
                conf_matrix[index][index] += 1
            else:
                if yas1 != "" and yas2 != "":
                    index1 = all_ya_text.index(yas1)
                    index2 = all_ya_text.index(yas2)
                    conf_matrix[index2][index1] += 1

        elif ID1 == 0 and ID2 == 0:
            conf_matrix[len(all_ya_text) - 1][len(all_ya_text) - 1] += 1

        elif ID1 == 0:
            yas2 = get_ya_node_text_from_prop(ID2, graph2)
            index = all_ya_text.index(yas2)
            conf_matrix[len(all_ya_text) - 1][index] += 1

        elif ID2 == 0:
            yas2 = get_ya_node_text_from_prop(ID1, graph1)
            index = all_ya_text.index(yas1)
            conf_matrix[index][len(all_ya_text) - 1] += 1

    return conf_matrix


def loc_ta_rels_comp(
    loc_matrix: List[Dict[str, Any]], graph1: DiGraph, graph2: DiGraph
) -> List[List[int]]:
    """Create confusion matrix for TA-nodes anchored in L-nodes.

    We check whether for each L-node in the relation we have the same amount of outgoing TA- nodes.
    """
    conf_matrix = [[0, 0], [0, 0]]

    for rel_dict in loc_matrix:
        ID1 = rel_dict["ID1"]
        ID2 = rel_dict["ID2"]
        text1 = rel_dict["text1"]
        text2 = rel_dict["text2"]

        if ID1 != 0 and ID2 != 0:
            tas1 = count_ta_nodes(ID1, graph1)
            tas2 = count_ta_nodes(ID2, graph2)

            if tas1 == tas2:
                conf_matrix[0][0] += 1
            elif tas1 > tas2:
                conf_matrix[1][0] += 1
            elif tas2 > tas1:
                conf_matrix[0][1] += 1

        elif ID1 == 0:
            conf_matrix[0][1] += 1

        elif ID2 == 0:
            conf_matrix[1][0] += 1

    overallRelations = len(loc_matrix) * len(loc_matrix)

    total_agreed_none = (
        overallRelations - conf_matrix[0][0] - conf_matrix[0][1] - conf_matrix[1][0]
    )

    conf_matrix[1][1] = total_agreed_none
    return conf_matrix


def prop_ya_anchor_comp(prop_matrix: List[Dict[str, Any]], graph1: DiGraph, graph2: DiGraph):
    """Create confusion matrix for YA-nodes anchoring the propositions: check for matching text
    field annotations."""
    conf_matrix = [[0, 0], [0, 0]]

    for rel_dict in prop_matrix:
        ID1 = rel_dict["ID1"]
        ID2 = rel_dict["ID2"]
        text1 = rel_dict["text1"]
        text2 = rel_dict["text2"]

        if ID1 != 0 and ID2 != 0:
            yas1 = get_ya_node_from_prop_id(ID1, graph1)
            yas2 = get_ya_node_from_prop_id(ID2, graph2)
            n_anch_1 = None
            n_anch_2 = None
            if yas1 != -1:
                n_anch_1 = get_node_anchor(yas1, graph1)
            if yas2 != -1:
                n_anch_2 = get_node_anchor(yas2, graph2)

            if not (n_anch_1 is None):
                if n_anch_1 == n_anch_2:
                    conf_matrix[0][0] = conf_matrix[0][0] + 1
                else:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1

        elif ID1 == 0 and ID2 == 0:
            conf_matrix[1][1] += 1
        elif ID1 == 0:
            conf_matrix[0][1] += 1
        elif ID2 == 0:
            conf_matrix[1][0] += 1

    return conf_matrix


def get_ta_nodes_from_id(node_id: int, graph: DiGraph) -> List[int]:
    """Collect all TA-nodes that are successors of a given node."""
    try:
        successor_nodes = list(graph.successors(node_id))
        ta_list = []
        for n in successor_nodes:
            n_type = graph.nodes[n]["type"]
            if n_type == "TA":
                n_id = n
                ta_list.append(n_id)
        return ta_list

    except Exception as e:
        logger.error(f"Failed to get successors for node with ID {node_id}")
        return []


def get_ya_node_from_prop_id(node_id: int, graph: DiGraph) -> int:
    """Get the predecessor YA-node (returns the first match or -1 if not found)."""
    try:
        predecessor_nodes = list(graph.predecessors(node_id))
        for n in predecessor_nodes:
            n_type = graph.nodes[n]["type"]
            if n_type == "YA":
                n_text = graph.nodes[n]["text"]
                return n
        return -1
    except Exception as e:
        logger.error(f"Failed to get predecessors for node with ID {node_id}")
        return -1


def get_ya_node_texts(graph1: DiGraph, graph2: DiGraph) -> List[str]:
    """Collect all possible text annotations for YA-nodes (w/o any duplicates)."""
    ya_node_list1 = []
    ya_node_list2 = []
    ya_text_list = []

    ya_node_list1 = centrality.get_rels("YA", graph1)
    ya_node_list2 = centrality.get_rels("YA", graph2)

    for ya in ya_node_list1:
        n_text = graph1.nodes[ya]["text"]
        ya_text_list.append(n_text)
    for ya in ya_node_list2:
        n_text = graph2.nodes[ya]["text"]
        ya_text_list.append(n_text)

    ya_text_list = list(set(ya_text_list))
    return ya_text_list


def get_ya_node_text_from_id(node_id: int, graph: DiGraph) -> str:
    """Get text field annotation for a YA-node that is a successor of the given node (returns the
    first match or empty string if not found)."""
    try:
        successor_nodes = list(graph.successors(node_id))
        for n in successor_nodes:
            n_type = graph.nodes[n]["type"]
            if n_type == "YA":
                n_text = graph.nodes[n]["text"]
                return n_text
        return ""
    except Exception as e:
        logger.error(f"Failed to get successors for node with ID {node_id}")
        return ""


def get_ya_node_text_from_prop(node_id: int, graph: DiGraph) -> str:
    """Get text field annotation for a YA-node that is a predecessor of the given node (returns the
    first match or empty string if not found)."""
    try:
        predecessor_nodes = list(graph.predecessors(node_id))
        for n in predecessor_nodes:
            n_type = graph.nodes[n]["type"]
            if n_type == "YA":
                n_text = graph.nodes[n]["text"]
                return n_text
        return ""
    except Exception as e:
        logger.error(f"Failed to get predecessors for node with ID {node_id}")
        return ""


def get_node_anchor(node_id: int, graph: DiGraph) -> str:
    """Get text field annotation for L-node or TA-node of the given node (returns the first match
    or empty string if not found)."""
    try:
        nodes = list(graph.predecessors(node_id))
        for n in nodes:
            n_type = graph.nodes[n]["type"]
            if n_type == "L" or n_type == "TA":
                n_text = graph.nodes[n]["text"]
                return n_text
        return ""
    except Exception as e:
        logger.error(f"Failed to get predecessors for node with ID {node_id}")
        return ""


def calculate_matching(
    predicted_data: Dict[str, List[Dict[str, Any]]], gold_data: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[int, Dict[int, int]]]:
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
    graph1, graph2 = get_graphs(predicted_data, gold_data)
    # Creating similarity matrix for propositional relations.
    prop_rels = get_sim_matrix(graph1, graph2, "propositions")
    # Creating similarity matrix for locutional relations.
    loc_rels = get_sim_matrix(graph1, graph2, "locutions")
    # (1) Do we have the same YA > S transitions? Do their annotations coincide?
    # Anchoring on S-nodes (RA/CA/MA) and combining them (checking how many YA-anchors,
    # predecessors of S-nodes, have the same/different text field annotations).
    ra_a = s_rel_anchor("RA", graph1, graph2)
    ma_a = s_rel_anchor("MA", graph1, graph2)
    ca_a = s_rel_anchor("CA", graph1, graph2)
    all_s = combine_s_node_matrix(ra_a, ca_a, ma_a)

    # (2) Do we have the same S-nodes?
    # Comparing propositional relations, building a confusion matrix for S-node (RA, MA, CA)
    # matches between the two graphs.
    prop_rels_comp_conf = prop_rels_comp(prop_rels, graph1, graph2)

    # (3) Do we have the same L > YA transitions? Do their annotations coincide?
    # Getting all YAs anchored in locutions, comparing the text field annotations of YA-nodes,
    # successors of L-nodes, anchored in locutions.
    loc_ya_rels_comp_conf = loc_ya_rels_comp(loc_rels, graph1, graph2)

    # (4) Do we have the same YA > I transitions? Do their annotations coincide?
    # Getting all YAs in propositions, comparing the text field annotations of YA-nodes,
    # predecessors of I-nodes, anchored in propositions.
    prop_ya_comp_conf = prop_ya_comp(prop_rels, graph1, graph2)

    # (5) Do we have the same (TA-node) transitions between the L-nodes (L > TA > L transitions)?
    # Getting all TAs anchored in locutions, checking whether each L-node pair (from graph1 and graph2)
    # has the same amount of outgoing TA-nodes.
    loc_ta_conf = loc_ta_rels_comp(loc_rels, graph1, graph2)

    # (6) Do we have I-nodes anchored in the same L-nodes (L > YA > I transitions)? Do their annotations coincide?
    # Getting all YAs anchored in propositions, comparing the text field annotations of L-/TA-nodes
    # that are predecessors of YA-nodes anchored in propositions (I-nodes).
    prop_ya_conf = prop_ya_anchor_comp(prop_rels, graph1, graph2)

    conf_matrices = [
        all_s,
        prop_rels_comp_conf,
        loc_ya_rels_comp_conf,
        prop_ya_comp_conf,
        loc_ta_conf,
        prop_ya_conf,
    ]
    as_dicts = [convert_to_dict(cm) for cm in conf_matrices]
    return as_dicts
