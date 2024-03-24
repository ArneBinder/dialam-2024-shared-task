import copy
import logging

import gmatch4py as gm
import networkx as nx
import numpy as np
import segeval
from bs4 import BeautifulSoup
from centrality import Centrality
from fuzzywuzzy import fuzz
from load_map import CorpusLoader
from numpy import unravel_index

logger = logging.getLogger(__name__)


class match:
    @staticmethod
    def get_graphs(dt1, dt2):
        centra = Centrality()

        corpus_loader = CorpusLoader()
        graph1 = corpus_loader.parse_json(dt1)
        graph2 = corpus_loader.parse_json(dt2)
        graph1 = centra.remove_iso_analyst_nodes(graph1)
        graph2 = centra.remove_iso_analyst_nodes(graph2)

        return graph1, graph2

    @staticmethod
    def get_similarity(text_1, text_2):
        aifsim = match()
        # text_1 and text_2 are xml data that uses spans to separate boundaries
        # e.g. BOSTON, MA ... <span class="highlighted" id="634541">Steven L.
        # Davis pled guilty yesterday to federal charges that he stole and disclosed trade secrets of The Gillette Company</span>.

        if text_1 == "" or text_2 == "":
            return "Error Text Input Is Empty"
        else:

            text_1 = text_1.replace("`", "")
            text_1 = text_1.replace("’", "")
            text_1 = text_1.replace("'", "")
            text_2 = text_2.replace("`", "")
            text_2 = text_2.replace("’", "")
            text_2 = text_2.replace("'", "")
            text_1 = text_1.strip()
            text_2 = text_2.strip()
            text_1 = text_1.replace("[", " [")
            text_2 = text_2.replace("[", " [")
            text_1 = text_1.replace("]", "] ")
            text_2 = text_2.replace("]", "] ")
            text_1 = text_1.replace("  ", " ")
            text_2 = text_2.replace("  ", " ")
            text_1 = text_1.replace("...", " ")
            text_2 = text_2.replace("...", " ")
            text_1 = text_1.replace("…", " ")
            text_2 = text_2.replace("…", " ")
            text_1 = text_1.replace(".", " ")
            text_2 = text_2.replace(".", " ")
            text_1 = text_1.replace(",", " ")
            text_2 = text_2.replace(",", " ")
            text_1 = text_1.replace("!", " ")
            text_2 = text_2.replace("!", " ")
            text_1 = text_1.replace("?", " ")
            text_2 = text_2.replace("?", " ")
            text_1 = text_1.replace("  ", " ")
            text_2 = text_2.replace("  ", " ")

            # Parse text using BeautifulSoup
            xml_soup_1 = BeautifulSoup(text_1, features="lxml")
            xml_soup_2 = BeautifulSoup(text_2, features="lxml")

            # Remove unwanted HTML tags
            xml_soup_1 = aifsim.remove_html_tags(xml_soup_1)
            # xml_soup_1 = BeautifulSoup(str(xml_soup_1), features="lxml").text

            xml_soup_2 = aifsim.remove_html_tags(xml_soup_2)
            # xml_soup_2 = BeautifulSoup(str(xml_soup_2), features="lxml").text
            # Get segments
            segments_1, words_1 = aifsim.get_segements(xml_soup_1)

            segments_2, words_2 = aifsim.get_segements(xml_soup_2)

            # Check segment length
            seg_check, seg1, seg2 = aifsim.check_segment_length(
                segments_1, words_1, segments_2, words_2
            )

            if not seg_check:
                error_text = "Error: Source Text Was Different as Segmentations differ in length"
                return error_text
            else:
                if seg1 == seg2:
                    ss = 1.0  # If segmentation sequences are identical, set similarity to maximum (1.0)
                else:
                    # Convert segments to masses
                    masses_1 = segeval.convert_positions_to_masses(seg1)

                    masses_2 = segeval.convert_positions_to_masses(seg2)

                    # Calculate segmentation similarity
                    ss = segeval.segmentation_similarity(masses_1, masses_2)

                return ss

    @staticmethod
    def is_iat(g, g1, centra):
        l_nodes = centra.get_l_node_list(g)
        l1_nodes = centra.get_l_node_list(g1)

        if len(l_nodes) < 1 and len(l1_nodes) < 1:
            return "aif"
        elif len(l_nodes) > 1 and len(l1_nodes) > 1:
            return "iat"
        else:
            return "diff"

    @staticmethod
    def remove_html_tags(xml_soup):
        for match in xml_soup.findAll("div"):
            match.replaceWithChildren()
        for match in xml_soup.findAll("p"):
            match.replaceWithChildren()
        for match in xml_soup.findAll("br"):
            match.replaceWithChildren()
        # new added
        # for match in xml_soup.findAll('span'):
        #     match.replaceWithChildren()
        #

        return xml_soup

    @staticmethod
    def get_segements(xml_soup):
        segment_list = []
        word_list = []
        if xml_soup.body:
            for i, tag in enumerate(xml_soup.body):
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
        else:
            for i, tag in enumerate(xml_soup):
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

    @staticmethod
    def check_segment_length(seg_1, word_1, seg_2, word_2):
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
                        # else:
                        #     return False, None, None
                return False, None, None

            else:
                for i in range(len(word_2) - 1):
                    if word_2[i] + word_2[i + 1] in word_1:
                        word_2[i : i + 2] = [word_2[i] + word_2[i + 1]]
                        seg_2 = [seg_2[0]] * len(word_2)

                        if len(seg_1) == len(seg_2):
                            return True, seg_1, seg_2
                        # else:
                        #     return False, None, None
                return False, None, None

    # @staticmethod
    # def check_segment_length(seg_1, seg_2):
    #     seg_1_len = len(seg_1)
    #     seg_2_len = len(seg_2)
    #
    #     if seg_1_len == seg_2_len:
    #         return True
    #     else:
    #         return False

    @staticmethod
    def get_normalized_edit_distance(g1, g2, label_equal, attr_name):
        if label_equal:
            dist = nx.algorithms.similarity.optimize_graph_edit_distance(
                g1, g2, node_match=lambda a, b: a[attr_name] == b[attr_name]
            )
        else:
            dist = nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2)

        max_g_len = max(len(g1.nodes), len(g2.nodes))
        ed_dist = min(list(dist))

        norm_ed_dist = (max_g_len - ed_dist) / max_g_len

        return norm_ed_dist

    @staticmethod
    def get_normalized_path_edit_distance(g1, g2, label_equal, attr_name):
        if label_equal:
            dist = nx.algorithms.similarity.optimize_edit_paths(
                g1, g2, node_match=lambda a, b: a[attr_name] == b[attr_name]
            )
        else:
            dist = nx.algorithms.similarity.optimize_edit_paths(g1, g2)

        max_g_len = max(len(g1.nodes), len(g2.nodes))
        ed_dist = min(list(dist))

        norm_ed_dist = (max_g_len - ed_dist) / max_g_len

        return norm_ed_dist

    @staticmethod
    def get_s_nodes(g):
        s_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] == "RA" or y["type"] == "CA" or y["type"] == "MA" or y["type"] == "PA"
        ]
        not_s_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] != "RA" and y["type"] != "CA" and y["type"] != "MA" and y["type"] != "PA"
        ]
        return s_nodes, not_s_nodes

    # Function to get I nodes and S nodes
    @staticmethod
    def get_i_s_nodes(g):
        i_s_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] == "I"
            or y["type"] == "RA"
            or y["type"] == "CA"
            or y["type"] == "MA"
            or y["type"] == "PA"
        ]
        not_i_s_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] != "I"
            and y["type"] != "RA"
            and y["type"] != "CA"
            and y["type"] != "MA"
            and y["type"] != "PA"
        ]
        return i_s_nodes, not_i_s_nodes

    @staticmethod
    def get_l_nodes(g):
        l_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "L"]
        not_l_nodes = [x for x, y in g.nodes(data=True) if y["type"] != "L"]
        return l_nodes, not_l_nodes

    @staticmethod
    def get_l_ta_nodes(g):
        l_ta_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "L" or y["type"] == "TA"]
        not_l_ta_nodes = [
            x for x, y in g.nodes(data=True) if y["type"] != "L" and y["type"] != "TA"
        ]
        return l_ta_nodes, not_l_ta_nodes

    @staticmethod
    def get_i_nodes(g):
        i_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "I"]
        not_i_nodes = [x for x, y in g.nodes(data=True) if y["type"] != "I"]
        return i_nodes, not_i_nodes

    @staticmethod
    def get_ya_nodes(g):
        ya_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "YA"]
        not_ya_nodes = [x for x, y in g.nodes(data=True) if y["type"] != "YA"]
        return ya_nodes, not_ya_nodes

    @staticmethod
    def get_ta_nodes(g):
        ta_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "TA"]
        not_ta_nodes = [x for x, y in g.nodes(data=True) if y["type"] != "TA"]
        return ta_nodes, not_ta_nodes

    @staticmethod
    def get_l_ya_nodes(g):
        l_ya_nodes = [x for x, y in g.nodes(data=True) if y["type"] == "L" or y["type"] == "YA"]
        not_l_ya_nodes = [
            x for x, y in g.nodes(data=True) if y["type"] != "L" and y["type"] != "YA"
        ]
        return l_ya_nodes, not_l_ya_nodes

    @staticmethod
    def get_l_i_ya_nodes(g):
        l_i_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] == "L" or y["type"] == "YA" or y["type"] == "I"
        ]
        not_l_i_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] != "L" and y["type"] != "YA" and y["type"] != "I"
        ]
        return l_i_ya_nodes, not_l_i_ya_nodes

    @staticmethod
    def get_l_ta_ya_nodes(g):
        l_ta_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] == "L" or y["type"] == "YA" or y["type"] == "TA"
        ]
        not_l_ta_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] != "L" and y["type"] != "YA" and y["type"] != "TA"
        ]
        return l_ta_ya_nodes, not_l_ta_ya_nodes

    @staticmethod
    def get_i_s_ya_nodes(g):
        i_s_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] == "I"
            or y["type"] == "RA"
            or y["type"] == "CA"
            or y["type"] == "MA"
            or y["type"] == "PA"
            or y["type"] == "YA"
        ]
        not_i_s_ya_nodes = [
            x
            for x, y in g.nodes(data=True)
            if y["type"] != "I"
            and y["type"] != "RA"
            and y["type"] != "CA"
            and y["type"] != "MA"
            and y["type"] != "PA"
            and y["type"] != "YA"
        ]
        return i_s_ya_nodes, not_i_s_ya_nodes

    @staticmethod
    def remove_nodes(graph, remove_list):
        graph.remove_nodes_from(remove_list)
        return graph

    @staticmethod
    def get_i_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()
        g1_i_nodes, g1_not_i_nodes = aifsim.get_i_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_i_nodes)
        g2_i_nodes, g2_not_i_nodes = aifsim.get_i_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_i_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, "")
        return ed

    @staticmethod
    def get_s_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()
        g1_nodes, g1_not_nodes = aifsim.get_s_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_s_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, "")
        return ed

    @staticmethod
    def get_i_s_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()
        g1_nodes, g1_not_nodes = aifsim.get_i_s_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_i_s_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "type")
        return ed

    @staticmethod
    def get_l_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()
        g1_nodes, g1_not_nodes = aifsim.get_l_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, "type")
        return ed

    @staticmethod
    def get_l_ta_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()
        g1_nodes, g1_not_nodes = aifsim.get_l_ta_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ta_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, "type")
        return ed

    @staticmethod
    def get_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "text")
        return ed

    @staticmethod
    def get_ya_l_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_l_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "text")
        return ed

    @staticmethod
    def get_ta_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_ta_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_ta_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "text")
        return ed

    @staticmethod
    def get_ya_l_i_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_l_i_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_i_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "text")
        return ed

    @staticmethod
    def get_l_ta_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_l_ta_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ta_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, "type")
        return ed

    @staticmethod
    def get_i_s_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = match()

        g1_nodes, g1_not_nodes = aifsim.get_i_s_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_i_s_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, "type")
        return ed

    @staticmethod
    def findMean(a, N):

        summ = 0

        # total sum calculation of matrix
        for i in range(N):
            for j in range(N):
                summ += a[i][j]

        return summ / (N * N)

    @staticmethod
    def get_normalized_gm_edit_distance(g1, g2, label_equal, attr_name):
        ged = gm.GraphEditDistance(1, 1, 1, 1)

        if label_equal:
            ged.set_attr_graph_used(attr_name, None)
            result = ged.compare([g1, g2], None)
        else:
            result = ged.compare([g1, g2], None)

        sim = ged.similarity(result)
        flat_sim = sim.flatten()
        flat_sim = flat_sim[flat_sim != 0]
        norm_ed_dist = min(flat_sim)
        # norm_ed_dist = findMean(sim, 2)
        # norm_ed_dist = (sim[0][1] + sim[1][0])/2
        return norm_ed_dist

    @staticmethod
    def call_diagram_parts_and_sum(g_copy, g1_copy, rep):
        aifsim = match()
        if rep == "aif":
            i_sim = aifsim.get_i_node_sim(g_copy, g1_copy)
            s_sim = aifsim.get_s_node_sim(g_copy, g1_copy)
            i_s_sim = aifsim.get_i_s_node_sim(g_copy, g1_copy)
            sum_list = [i_sim, s_sim, i_s_sim]
        else:
            i_sim = aifsim.get_i_node_sim(g_copy, g1_copy)
            s_sim = aifsim.get_s_node_sim(g_copy, g1_copy)
            i_s_sim = aifsim.get_i_s_node_sim(g_copy, g1_copy)

            i_s_ya_sim = aifsim.get_i_s_ya_node_sim(g_copy, g1_copy)
            l_sim = aifsim.get_l_node_sim(g_copy, g1_copy)
            l_ta_sim = aifsim.get_l_ta_node_sim(g_copy, g1_copy)
            ya_sim = aifsim.get_ya_node_sim(g_copy, g1_copy)
            ta_sim = aifsim.get_ta_node_sim(g_copy, g1_copy)
            l_i_ya_sim = aifsim.get_ya_l_i_node_sim(g_copy, g1_copy)
            l_ta_ya_sim = aifsim.get_l_ta_ya_node_sim(g_copy, g1_copy)
            l_ta_ya_sim = aifsim.get_ya_l_node_sim(g_copy, g1_copy)
            sum_list = [
                i_sim,
                s_sim,
                i_s_sim,
                i_s_ya_sim,
                l_sim,
                l_ta_sim,
                ya_sim,
                ta_sim,
                l_i_ya_sim,
                l_ta_ya_sim,
                l_ta_ya_sim,
            ]
        sum_tot = sum(sum_list)
        tot = sum_tot / len(sum_list)
        sum_list = np.asarray(sum_list)
        harm = len(sum_list) / np.sum(1.0 / sum_list)
        return tot

    @staticmethod
    def rels_to_dict(rels, switched):
        new_list = []
        for rel in rels:
            id_1 = rel[0][0]
            id_2 = rel[1][0]
            text_1 = rel[0][1]
            text_2 = rel[1][1]

            if switched:

                mat_dict = {"ID1": id_2, "ID2": id_1, "text1": text_2, "text2": text_1}
            else:
                mat_dict = {"ID1": id_1, "ID2": id_2, "text1": text_1, "text2": text_2}
            new_list.append(mat_dict)
        return new_list

    @staticmethod
    def get_prop_sim_matrix(graph, graph1):
        centra = Centrality()
        aifsim = match()

        g_copy = graph.copy()
        g1_copy = graph1.copy()

        g_inodes = centra.get_i_node_list(g_copy)
        g1_inodes = centra.get_i_node_list(g1_copy)
        relsi, valsi, switched = aifsim.text_sim_matrix(g_inodes, g1_inodes)
        # if switched the relations have been switched order so they need reversed when creating the dictionary

        rels_dict = aifsim.rels_to_dict(relsi, switched)

        return rels_dict

    @staticmethod
    def get_loc_sim_matrix(graph, graph1):
        centra = Centrality()
        aifsim = match()

        g_copy = graph.copy()
        g1_copy = graph1.copy()

        g_lnodes = centra.get_l_node_list(g_copy)
        g1_lnodes = centra.get_l_node_list(g1_copy)
        relsl, valsl, switched = aifsim.text_sim_matrix(g_lnodes, g1_lnodes)

        rels_dict = aifsim.rels_to_dict(relsl, switched)

        return rels_dict

    @staticmethod
    def text_sim_matrix(g_list, g1_list):
        aifsim = match()
        g_size = len(g_list)
        g1_size = len(g1_list)

        switch_flag = False

        if g_size >= g1_size:
            mat = aifsim.loop_nodes(g_list, g1_list)
            rels, vals = aifsim.select_max_vals(mat, g1_size, g_list, g1_list)
        else:
            switch_flag = True
            mat = aifsim.loop_nodes(g1_list, g_list)
            rels, vals = aifsim.select_max_vals(mat, g_size, g1_list, g_list)

        return rels, vals, switch_flag

    @staticmethod
    def loop_nodes(g_list, g1_list):
        matrix = np.zeros((len(g_list), len(g1_list)))
        for i, node in enumerate(g_list):
            text = node[1]
            text = text.lower()
            for i1, node1 in enumerate(g1_list):

                text1 = node1[1]
                text1 = text1.lower()
                # lev_val = normalized_levenshtein.distance(text, text1)
                lev_val = (fuzz.ratio(text, text1)) / 100
                matrix[i][i1] = lev_val

        return matrix

    @staticmethod
    def select_max_vals(matrix, smallest_value, g_list, g1_list):
        counter = 0
        lev_vals = []
        lev_rels = []
        index_list = list(range(len(g_list)))
        m_copy = copy.deepcopy(matrix)
        while counter <= smallest_value - 1:
            index_tup = unravel_index(m_copy.argmax(), m_copy.shape)
            # matrix[index_tup[0]][index_tup[1]] = -9999999
            m_copy[index_tup[0]] = 0  # zeroes out row i
            m_copy[:, index_tup[1]] = 0  # zeroes out column i
            lev_rels.append((g_list[index_tup[0]], g1_list[index_tup[1]]))
            lev_vals.append(matrix[index_tup[0]][index_tup[1]])
            index_list.remove(index_tup[0])
            counter = counter + 1
        for vals in index_list:
            lev_rels.append((g_list[vals], (0, "")))
            lev_vals.append(0)
        return lev_rels, lev_vals

    @staticmethod
    def convert_to_dict(conf_matrix):
        values = []
        dicts = {}
        for i, col in enumerate(conf_matrix):
            dicts[i] = {}
            for j, row in enumerate(col):
                dicts[i][j] = row
        return dicts

    @staticmethod
    def get_mean_of_list(a):
        val_tot = sum(a)
        tot = val_tot / len(a)
        return tot

    @staticmethod
    def get_l_i_mean(lnode, inode):
        return (lnode + inode) / 2

    @staticmethod
    def get_graph_sim(aif_id1, aif_id2):
        centra = Centrality()
        aifsim = match()
        graph, json = aifsim.get_graph(aif_id1, centra)
        graph1, json1 = aifsim.get_graph(aif_id2, centra)
        graph = centra.remove_iso_analyst_nodes(graph)
        graph1 = centra.remove_iso_analyst_nodes(graph1)
        rep_form = aifsim.is_iat(graph, graph1, centra)
        g_copy = graph.copy()
        g1_copy = graph1.copy()
        graph_mean = 0
        text_mean = 0
        overall_mean = 0
        if rep_form == "diff":
            return "Error"
        else:
            graph_mean = aifsim.call_diagram_parts_and_sum(g_copy, g1_copy, rep_form)
        if rep_form == "aif":
            g_inodes = centra.get_i_node_list(g_copy)
            g1_inodes = centra.get_i_node_list(g1_copy)
            relsi, valsi = aifsim.text_sim_matrix(g_inodes, g1_inodes)
            i_mean = aifsim.get_mean_of_list(valsi)
            text_mean = i_mean
        else:
            g_inodes = centra.get_i_node_list(g_copy)
            g1_inodes = centra.get_i_node_list(g1_copy)
            g_lnodes = centra.get_l_node_list(g_copy)
            g1_lnodes = centra.get_l_node_list(g1_copy)
            relsi, valsi = aifsim.text_sim_matrix(g_inodes, g1_inodes)
            relsl, valsl = aifsim.text_sim_matrix(g_lnodes, g1_lnodes)
            i_mean = aifsim.get_mean_of_list(valsi)
            l_mean = aifsim.get_mean_of_list(valsl)
            text_mean = aifsim.get_l_i_mean(l_mean, i_mean)

        overall_score = aifsim.get_l_i_mean(text_mean, graph_mean)
        return overall_score, text_mean, graph_mean

    # Get TA Anchor RA, CA, MA - Requires CA_anchor ma_anchor and ra_anchor then combination of confusion matrices

    @staticmethod
    def ra_anchor(graph1, graph2):

        conf_matrix = [[0, 0], [0, 0]]
        aifsim = match()
        cent = Centrality()
        ras1 = cent.get_ras(graph1)
        ras2 = cent.get_ras(graph2)

        ra1_len = len(ras1)
        ra2_len = len(ras2)

        if ra1_len > 0 and ra2_len > 0:
            if ra1_len > ra2_len:
                for ra_i, ra in enumerate(ras1):
                    ras2_id = ""
                    yas1 = aifsim.get_ya_nodes_from_prop(ra, graph1)
                    try:
                        ras2_id = ras2[ra_i]
                    except Exception as e:
                        ras2_id = ""
                        logger.error(f"Failed to find index {ra_i} in {ras2}: {e}")

                    if ras2_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[1][0] = conf_matrix[1][0] + 1
                    else:
                        yas2 = aifsim.get_ya_nodes_by_node_id(ras2_id, graph2)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[1][0] = conf_matrix[1][0] + 1

            elif ra2_len > ra1_len:
                for ra_i, ra in enumerate(ras2):
                    ras1_id = ""
                    yas2 = aifsim.get_ya_nodes_from_prop(ra, graph2)
                    try:
                        ras1_id = ras1[ra_i]
                    except Exception as e:
                        ras1_id = ""
                        logger.error(f"Failed to find index {ra_i} in {ras1}: {e}")

                    if ras1_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[0][1] = conf_matrix[0][1] + 1
                    else:
                        yas1 = aifsim.get_ya_nodes_by_node_id(ras1_id, graph1)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[0][1] = conf_matrix[0][1] + 1

            else:
                for ra_i, ra in enumerate(ras1):
                    ya1 = aifsim.get_ya_nodes_from_prop(ra, graph1)
                    ya2 = aifsim.get_ya_nodes_from_prop(ras2[ra_i], graph2)

                    if ya1 == ya2 and ya1 != "":
                        conf_matrix[0][0] = conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] = conf_matrix[1][0] + 1

        elif ra1_len == 0 and ra2_len == 0:
            conf_matrix[1][1] = conf_matrix[1][1] + 1

        elif ra1_len == 0:
            conf_matrix[0][1] = conf_matrix[0][1] + ra2_len
        elif ra2_len == 0:
            conf_matrix[1][0] = conf_matrix[1][0] + ra1_len

        return conf_matrix

    @staticmethod
    def ma_anchor(graph1, graph2):

        conf_matrix = [[0, 0], [0, 0]]
        aifsim = match()
        cent = Centrality()
        cas1 = cent.get_mas(graph1)
        cas2 = cent.get_mas(graph2)

        ca1_len = len(cas1)
        ca2_len = len(cas2)

        if ca1_len > 0 and ca2_len > 0:
            if ca1_len > ca2_len:
                for ca_i, ca in enumerate(cas1):
                    cas2_id = ""
                    yas1 = aifsim.get_ya_nodes_from_prop(ca, graph1)
                    try:
                        cas2_id = cas2[ca_i]
                    except Exception as e:
                        cas2_id = ""
                        logger.error(f"Failed to find index {ca_i} in {cas2}: {e}")

                    if cas2_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[1][0] = conf_matrix[1][0] + 1
                    else:
                        yas2 = aifsim.get_ya_nodes_by_node_id(cas2_id, graph2)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[1][0] = conf_matrix[1][0] + 1

            elif ca2_len > ca1_len:
                for ca_i, ca in enumerate(cas2):
                    cas1_id = ""
                    yas2 = aifsim.get_ya_nodes_from_prop(ca, graph2)
                    try:
                        cas1_id = cas1[ca_i]
                    except Exception as e:
                        cas1_id = ""
                        logger.error(f"Failed to find index {ca_i} in {cas1}: {e}")

                    if cas1_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[0][1] = conf_matrix[0][1] + 1
                    else:
                        yas1 = aifsim.get_ya_nodes_by_node_id(cas1_id, graph1)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[0][1] = conf_matrix[0][1] + 1

            else:
                for ca_i, ca in enumerate(cas1):
                    ya1 = aifsim.get_ya_nodes_from_prop(ca, graph1)
                    ya2 = aifsim.get_ya_nodes_from_prop(cas2[ca_i], graph2)

                    if ya1 == ya2 and ya1 != "":
                        conf_matrix[0][0] = conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] = conf_matrix[1][0] + 1

        elif ca1_len == 0 and ca2_len == 0:
            conf_matrix[1][1] = conf_matrix[1][1] + 1

        elif ca1_len == 0:
            conf_matrix[0][1] = conf_matrix[0][1] + ca2_len
        elif ca2_len == 0:
            conf_matrix[1][0] = conf_matrix[1][0] + ca1_len

        return conf_matrix

    @staticmethod
    def ca_anchor(graph1, graph2):

        conf_matrix = [[0, 0], [0, 0]]
        aifsim = match()
        cent = Centrality()
        cas1 = cent.get_cas(graph1)
        cas2 = cent.get_cas(graph2)

        ca1_len = len(cas1)
        ca2_len = len(cas2)

        if ca1_len > 0 and ca2_len > 0:
            if ca1_len > ca2_len:
                for ca_i, ca in enumerate(cas1):
                    cas2_id = ""
                    yas1 = aifsim.get_ya_nodes_from_prop(ca, graph1)
                    try:
                        cas2_id = cas2[ca_i]
                    except Exception as e:
                        cas2_id = ""
                        logger.error(f"Failed to find index {ca_i} in {cas2}: {e}")

                    if cas2_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[1][0] = conf_matrix[1][0] + 1
                    else:
                        yas2 = aifsim.get_ya_nodes_by_node_id(cas2_id, graph2)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[1][0] = conf_matrix[1][0] + 1

            elif ca2_len > ca1_len:
                for ca_i, ca in enumerate(cas2):
                    cas1_id = ""
                    yas2 = aifsim.get_ya_nodes_from_prop(ca, graph2)
                    try:
                        cas1_id = cas1[ca_i]
                    except Exception as e:
                        cas1_id = ""
                        logger.error(f"Failed to find index {ca_i} in {cas1}: {e}")

                    if cas1_id == "":
                        # conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                        conf_matrix[0][1] = conf_matrix[0][1] + 1
                    else:
                        yas1 = aifsim.get_ya_nodes_by_node_id(cas1_id, graph1)
                        if yas1 == yas2 and yas1 != "":

                            conf_matrix[0][0] = conf_matrix[0][0] + 1
                        else:
                            conf_matrix[0][1] = conf_matrix[0][1] + 1

            else:
                for ca_i, ca in enumerate(cas1):
                    ya1 = aifsim.get_ya_nodes_from_prop(ca, graph1)
                    ya2 = aifsim.get_ya_nodes_from_prop(cas2[ca_i], graph2)

                    if ya1 == ya2 and ya1 != "":
                        conf_matrix[0][0] = conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] = conf_matrix[1][0] + 1

        elif ca1_len == 0 and ca2_len == 0:
            conf_matrix[1][1] = conf_matrix[1][1] + 1

        elif ca1_len == 0:
            conf_matrix[0][1] = conf_matrix[0][1] + ca2_len
        elif ca2_len == 0:
            conf_matrix[1][0] = conf_matrix[1][0] + ca1_len

        return conf_matrix

    @staticmethod
    def combine_s_node_matrix(ra, ca, ma):
        # Combines the ra_anchor, ca_anchor and ma_anchor, matrices
        result = [[ra[i][j] + ca[i][j] for j in range(len(ra[0]))] for i in range(len(ra))]

        all_result = [
            [ma[i][j] + result[i][j] for j in range(len(result[0]))] for i in range(len(result))
        ]

        return all_result

    @staticmethod
    def get_ya_nodes_by_node_id(node_id, graph):

        ya_nodes = list(graph.successors(node_id))
        for ya in ya_nodes:
            n_type = graph.nodes[ya]["type"]
            if n_type == "YA":
                n_text = graph.nodes[ya]["text"]
                return n_text
        return ""

    @staticmethod
    def prop_rels_comp(prop_matrix, graph1, graph2):
        aifsim = match()
        conf_matrix = [[0, 0], [0, 0]]

        for rel_dict in prop_matrix:
            ID1 = rel_dict["ID1"]
            ID2 = rel_dict["ID2"]
            text1 = rel_dict["text1"]
            text2 = rel_dict["text2"]

            if ID1 != 0 and ID2 != 0:

                ras1, cas1, mas1 = aifsim.count_s_nodes(ID1, graph1)

                ras2, cas2, mas2 = aifsim.count_s_nodes(ID2, graph2)

                if ras1 == ras2 and ras1 != "":
                    conf_matrix[0][0] = conf_matrix[0][0] + 1
                elif ras1 > ras2:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1
                elif ras2 > ras1:
                    conf_matrix[0][1] = conf_matrix[0][1] + 1

                if cas1 == cas2 and cas1 != "":
                    conf_matrix[0][0] = conf_matrix[0][0] + 1
                elif cas1 > cas2:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1
                elif cas2 > cas1:
                    conf_matrix[0][1] = conf_matrix[0][1] + 1

                if mas1 == mas2 and mas1 != "":
                    conf_matrix[0][0] = conf_matrix[0][0] + 1
                elif mas1 > mas2:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1
                elif mas2 > mas1:
                    conf_matrix[0][1] = conf_matrix[0][1] + 1
            elif ID1 == 0 and ID2 == 0:
                conf_matrix[1][1] = conf_matrix[1][1] + 1
            elif ID1 == 0:
                conf_matrix[0][1] = conf_matrix[0][1] + 1
            elif ID2 == 0:
                conf_matrix[1][0] = conf_matrix[1][0] + 1

        overallRelations = len(prop_matrix) * len(prop_matrix)

        total_agreed_none = (
            overallRelations - conf_matrix[0][0] - conf_matrix[0][1] - conf_matrix[1][0]
        )

        # update
        if total_agreed_none < 0:
            total_agreed_none = 0
        conf_matrix[1][1] = total_agreed_none

        #

        # conf_matrix[1][1] = total_agreed_none

        return conf_matrix

    @staticmethod
    def count_s_nodes(node_id, graph):
        RA_count = 0
        MA_count = 0
        CA_count = 0
        try:
            s_nodes = list(graph.predecessors(node_id))
        except Exception as e:
            logger.error(f"Failed to get predecessors for node with ID {node_id}")
            s_nodes = []
        for s in s_nodes:
            n_type = graph.nodes[s]["type"]
            if n_type == "RA":
                RA_count = RA_count + 1
            elif n_type == "CA":
                CA_count = CA_count + 1
            elif n_type == "MA":
                MA_count = MA_count + 1
        return RA_count, CA_count, MA_count

    @staticmethod
    def loc_ya_rels_comp(loc_matrix, graph1, graph2):

        aifsim = match()

        all_ya_text = aifsim.get_ya_node_text(graph1, graph2)
        conf_matrix = [
            [0 for x in range(len(all_ya_text) + 1)] for y in range(len(all_ya_text) + 1)
        ]
        all_ya_text.append("")

        # Gets all YAs anchored in Locutions

        for rel_dict in loc_matrix:
            ID1 = rel_dict["ID1"]
            ID2 = rel_dict["ID2"]
            text1 = rel_dict["text1"]
            text2 = rel_dict["text2"]

            if ID1 != 0 and ID2 != 0:

                yas1 = aifsim.get_ya_nodes_from_id(ID1, graph1)
                yas2 = aifsim.get_ya_nodes_from_id(ID2, graph2)

                if yas1 == yas2 and yas1 in all_ya_text:
                    index = all_ya_text.index(yas1)
                    conf_matrix[index][index] = conf_matrix[index][index] + 1

                elif yas1 in all_ya_text and yas2 in all_ya_text:
                    index1 = all_ya_text.index(yas1)
                    index2 = all_ya_text.index(yas2)
                    conf_matrix[index2][index1] = conf_matrix[index2][index1] + 1

            elif ID1 == 0 and ID2 == 0:
                conf_matrix[len(all_ya_text) - 1][len(all_ya_text) - 1] = (
                    conf_matrix[len(all_ya_text) - 1][len(all_ya_text) - 1] + 1
                )

            elif ID1 == 0:
                yas2 = aifsim.get_ya_nodes_from_id(ID2, graph2)
                index = all_ya_text.index(yas2)

                conf_matrix[len(all_ya_text) - 1][index] = (
                    conf_matrix[len(all_ya_text) - 1][index] + 1
                )
            elif ID2 == 0:
                yas1 = aifsim.get_ya_nodes_from_id(ID1, graph1)
                index = all_ya_text.index(yas1)

                conf_matrix[index][len(all_ya_text) - 1] = (
                    conf_matrix[index][len(all_ya_text) - 1] + 1
                )

            # Gets all YAs anchored in Transitions via locutions - we only want to loop the matrix once

            conf_matrix = aifsim.get_ta_locs(ID1, ID2, graph1, graph2, conf_matrix, all_ya_text)

        return conf_matrix

    @staticmethod
    def check_none(val):
        if val == "None":
            return True
        else:
            return False

    @staticmethod
    def get_ta_locs(ID1, ID2, graph1, graph2, conf_matrix, all_ya_text):
        all_ya_text_ext = copy.deepcopy(all_ya_text)
        aifsim = match()
        if ID1 != 0 and ID2 != 0:

            tas1 = aifsim.get_ta_node_from_id(ID1, graph1)
            tas2 = aifsim.get_ta_node_from_id(ID2, graph2)

            if len(tas1) > 0 and len(tas2) > 0:
                # compare the ta lists
                if len(tas1) > len(tas2):
                    for tai, ta in enumerate(tas1):
                        tas2_id = ""
                        yas1 = aifsim.get_ya_nodes_from_id(ta, graph1)
                        try:
                            tas2_id = tas2[tai]
                        except Exception as e:
                            tas2_id = ""
                            logger.error(f"Failed to find index {tai} in {tas2}: {e}")

                        if tas2_id == "":
                            index = all_ya_text_ext.index(yas1)

                            conf_matrix[index][len(all_ya_text_ext) - 1] = (
                                conf_matrix[index][len(all_ya_text_ext) - 1] + 1
                            )

                        else:
                            yas2 = aifsim.get_ya_nodes_from_id(tas2_id, graph2)
                            if yas1 == yas2 and yas1 in all_ya_text_ext:
                                index = all_ya_text_ext.index(yas1)
                                conf_matrix[index][index] = conf_matrix[index][index] + 1

                            elif yas1 in all_ya_text_ext and yas2 in all_ya_text_ext:
                                index1 = all_ya_text_ext.index(yas1)
                                index2 = all_ya_text_ext.index(yas2)

                                conf_matrix[index2][index1] = conf_matrix[index2][index1] + 1
                elif len(tas2) > len(tas1):
                    for tai, ta in enumerate(tas2):
                        tas1_id = ""
                        yas2 = aifsim.get_ya_nodes_from_id(ta, graph2)
                        try:
                            tas1_id = tas1[tai]
                        except Exception as e:
                            tas1_id = ""
                            logger.error(f"Failed to find index {tai} in {tas1}: {e}")

                        if tas1_id == "":
                            index = all_ya_text_ext.index(yas2)

                            conf_matrix[len(all_ya_text_ext) - 1][index] = (
                                conf_matrix[len(all_ya_text_ext) - 1][index] + 1
                            )

                        else:
                            yas1 = aifsim.get_ya_nodes_from_id(tas1_id, graph1)
                            if yas1 == yas2 and yas2 in all_ya_text_ext:
                                index = all_ya_text_ext.index(yas2)

                                conf_matrix[index][index] = conf_matrix[index][index] + 1

                            elif yas1 in all_ya_text_ext and yas2 in all_ya_text_ext:
                                index1 = all_ya_text_ext.index(yas1)
                                index2 = all_ya_text_ext.index(yas2)

                                conf_matrix[index2][index1] = conf_matrix[index2][index1] + 1
                else:
                    for tai, ta in enumerate(tas1):
                        yas1 = aifsim.get_ya_nodes_from_id(ta, graph1)
                        yas2 = aifsim.get_ya_nodes_from_id(tas2[tai], graph2)
                        if yas1 == yas2 and yas1 in all_ya_text_ext:
                            index = all_ya_text_ext.index(yas1)

                            conf_matrix[index][index] = conf_matrix[index][index] + 1

                        elif yas1 in all_ya_text_ext and yas2 in all_ya_text_ext:
                            index1 = all_ya_text_ext.index(yas1)
                            index2 = all_ya_text_ext.index(yas2)

                            conf_matrix[index2][index1] = conf_matrix[index2][index1] + 1

            elif len(tas1) > 0 and len(tas2) < 1:

                for ta in tas1:
                    yas1 = aifsim.get_ya_nodes_from_id(ta, graph1)
                    if yas1 == "":
                        conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                        )
                    else:
                        index = all_ya_text_ext.index(yas1)

                        conf_matrix[index][len(all_ya_text_ext) - 1] = (
                            conf_matrix[index][len(all_ya_text_ext) - 1] + 1
                        )

            elif len(tas2) > 0 and len(tas1) < 1:

                for ta in tas2:
                    yas2 = aifsim.get_ya_nodes_from_id(ta, graph2)
                    if yas2 == "":
                        conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                        )
                    else:
                        index = all_ya_text_ext.index(yas2)

                        conf_matrix[len(all_ya_text_ext) - 1][index] = (
                            conf_matrix[len(all_ya_text_ext) - 1][index] + 1
                        )

            elif len(tas1) < 1 and len(tas2) < 1:

                conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                    conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                )

        elif ID1 == 0:
            tas2 = aifsim.get_ta_node_from_id(ID2, graph2)

            if len(tas2) > 0:
                for ta in tas2:
                    yas2 = aifsim.get_ya_nodes_from_id(ta, graph2)
                    if yas2 == "":
                        conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                        )
                    else:
                        index = all_ya_text.index(yas2)

                        conf_matrix[len(all_ya_text_ext) - 1][index] = (
                            conf_matrix[len(all_ya_text_ext) - 1][index] + 1
                        )
            elif len(tas2) < 1:
                conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                    conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                )

        elif ID2 == 0:
            tas1 = aifsim.get_ta_node_from_id(ID1, graph1)

            if len(tas1) > 0:
                for ta in tas1:
                    yas1 = aifsim.get_ya_nodes_from_id(ta, graph1)
                    if yas1 == "":
                        conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                            conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                        )
                    else:
                        index = all_ya_text_ext.index(yas1)

                        conf_matrix[index][len(all_ya_text_ext) - 1] = (
                            conf_matrix[index][len(all_ya_text_ext) - 1] + 1
                        )
            elif len(tas1) < 1:
                conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] = (
                    conf_matrix[len(all_ya_text_ext) - 1][len(all_ya_text_ext) - 1] + 1
                )

        return conf_matrix

    @staticmethod
    def get_ya_node_text(graph1, graph2):
        ya_node_list1 = []
        ya_node_list2 = []
        ya_text_list = []

        centra = Centrality()
        ya_node_list1 = centra.get_yas(graph1)
        ya_node_list2 = centra.get_yas(graph2)

        for ya in ya_node_list1:
            n_text = graph1.nodes[ya]["text"]
            ya_text_list.append(n_text)
        for ya in ya_node_list2:
            n_text = graph2.nodes[ya]["text"]
            ya_text_list.append(n_text)

        ya_text_list = list(set(ya_text_list))

        return ya_text_list

    @staticmethod
    def get_ya_nodes_from_id(node_id, graph):

        ya_nodes = list(graph.successors(node_id))
        for ya in ya_nodes:
            n_type = graph.nodes[ya]["type"]
            if n_type == "YA":
                n_text = graph.nodes[ya]["text"]
                return n_text
        return ""

    @staticmethod
    def get_ta_node_from_id(node_id, graph):
        ta_nodes = list(graph.successors(node_id))
        ta_list = []
        for ta in ta_nodes:
            n_type = graph.nodes[ta]["type"]
            if n_type == "TA":
                n_id = ta

                ta_list.append(n_id)
        return ta_list

    @staticmethod
    def prop_ya_comp(prop_matrix, graph1, graph2):
        aifsim = match()
        all_ya_text = aifsim.get_ya_node_text(graph1, graph2)
        conf_matrix = [
            [0 for x in range(len(all_ya_text) + 1)] for y in range(len(all_ya_text) + 1)
        ]
        all_ya_text.append("")
        for rel_dict in prop_matrix:
            ID1 = rel_dict["ID1"]
            ID2 = rel_dict["ID2"]
            text1 = rel_dict["text1"]
            text2 = rel_dict["text2"]

            if ID1 != 0 and ID2 != 0:

                yas1 = aifsim.get_ya_nodes_from_prop(ID1, graph1)
                yas2 = aifsim.get_ya_nodes_from_prop(ID2, graph2)

                if yas1 == yas2 and yas1 in all_ya_text:
                    index = all_ya_text.index(yas1)
                    conf_matrix[index][index] = conf_matrix[index][index] + 1

                else:
                    if yas1 is not None and yas2 is not None:
                        index1 = all_ya_text.index(yas1)
                        index2 = all_ya_text.index(yas2)
                        conf_matrix[index2][index1] = conf_matrix[index2][index1] + 1

            elif ID1 == 0 and ID2 == 0:
                conf_matrix[len(all_ya_text) - 1][len(all_ya_text) - 1] = conf_matrix[
                    len(all_ya_text) - 1
                ][len(all_ya_text) - 1]
            elif ID1 == 0:
                yas2 = aifsim.get_ya_nodes_from_prop(ID2, graph2)
                index = all_ya_text.index(yas2)

                conf_matrix[len(all_ya_text) - 1][index] = (
                    conf_matrix[len(all_ya_text) - 1][index] + 1
                )
            elif ID2 == 0:
                yas2 = aifsim.get_ya_nodes_from_prop(ID1, graph1)
                index = all_ya_text.index(yas1)

                conf_matrix[index][len(all_ya_text) - 1] = (
                    conf_matrix[index][len(all_ya_text) - 1] + 1
                )

        return conf_matrix

    @staticmethod
    def get_ya_nodes_from_prop(node_id, graph):
        try:
            ya_nodes = list(graph.predecessors(node_id))
            for ya in ya_nodes:
                n_type = graph.nodes[ya]["type"]
                if n_type == "YA":
                    n_text = graph.nodes[ya]["text"]
                    return n_text
        except Exception as e:
            logger.error(f"Failed to get predecessors for node with ID {node_id}")
            return ""

    @staticmethod
    def loc_ta_rels_comp(loc_matrix, graph1, graph2):
        aifsim = match()
        conf_matrix = [[0, 0], [0, 0]]

        for rel_dict in loc_matrix:
            ID1 = rel_dict["ID1"]
            ID2 = rel_dict["ID2"]
            text1 = rel_dict["text1"]
            text2 = rel_dict["text2"]

            if ID1 != 0 and ID2 != 0:

                tas1 = aifsim.count_ta_nodes(ID1, graph1)
                tas2 = aifsim.count_ta_nodes(ID2, graph2)

                if tas1 == tas2 and tas1 != "":
                    conf_matrix[0][0] = conf_matrix[0][0] + 1
                elif tas1 > tas2:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1
                elif tas2 > tas1:
                    conf_matrix[0][1] = conf_matrix[0][1] + 1

            elif ID1 == 0:
                conf_matrix[0][1] = conf_matrix[0][1] + 1
            elif ID2 == 0:
                conf_matrix[1][0] = conf_matrix[1][0] + 1

        overallRelations = len(loc_matrix) * len(loc_matrix)

        total_agreed_none = (
            overallRelations - conf_matrix[0][0] - conf_matrix[0][1] - conf_matrix[1][0]
        )

        conf_matrix[1][1] = total_agreed_none

        return conf_matrix

    @staticmethod
    def count_ta_nodes(node_id, graph):
        TA_count = 0
        ta_nodes = list(graph.successors(node_id))
        for ta in ta_nodes:
            n_type = graph.nodes[ta]["type"]
            if n_type == "TA":
                TA_count = TA_count + 1
        return TA_count

    @staticmethod
    def prop_ya_anchor_comp(prop_matrix, graph1, graph2):
        aifsim = match()
        conf_matrix = [[0, 0], [0, 0]]

        for rel_dict in prop_matrix:
            ID1 = rel_dict["ID1"]
            ID2 = rel_dict["ID2"]
            text1 = rel_dict["text1"]
            text2 = rel_dict["text2"]

            if ID1 != 0 and ID2 != 0:

                yas1 = aifsim.get_ya_nodes_from_prop_id(ID1, graph1)
                yas2 = aifsim.get_ya_nodes_from_prop_id(ID2, graph2)
                n_anch_1 = None
                n_anch_2 = None
                if yas1 != "":
                    n_anch_1 = aifsim.get_node_anchor(yas1, graph1)
                if yas2 != "":
                    n_anch_2 = aifsim.get_node_anchor(yas2, graph2)

                if n_anch_1 == n_anch_2 and not (n_anch_1 is None):
                    conf_matrix[0][0] = conf_matrix[0][0] + 1

                else:
                    conf_matrix[1][0] = conf_matrix[1][0] + 1

            elif ID1 == 0 and ID2 == 0:
                conf_matrix[1][1] = conf_matrix[1][1] + 1
            elif ID1 == 0:
                conf_matrix[0][1] = conf_matrix[0][1] + 1
            elif ID2 == 0:
                conf_matrix[1][0] = conf_matrix[1][0] + 1

        return conf_matrix

    @staticmethod
    def get_ya_nodes_from_prop_id(node_id, graph):
        try:
            ya_nodes = list(graph.predecessors(node_id))
            for ya in ya_nodes:
                n_type = graph.nodes[ya]["type"]
                if n_type == "YA":
                    n_text = graph.nodes[ya]["text"]
                    return ya
        except Exception as e:
            logger.error(f"Failed to get predecessors for node with ID {node_id}")
            return ""

    @staticmethod
    def get_node_anchor(node_id, graph):
        try:
            nodes = list(graph.predecessors(node_id))
            for n in nodes:
                n_type = graph.nodes[n]["type"]
                if n_type == "L" or n_type == "TA":
                    n_text = graph.nodes[n]["text"]
                    return n_text
        except Exception as e:
            logger.error(f"Failed to get predecessors for node with ID {node_id}")
            return ""
