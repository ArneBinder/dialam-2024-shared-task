import json
import logging

import networkx as nx
from load_map import CorpusLoader

logger = logging.getLogger(__name__)


class Centrality:
    @staticmethod
    def get_graph(node_path):
        corpus_loader = CorpusLoader()
        try:
            with open(node_path) as json_data:
                graph = corpus_loader.parse_json(json.load(json_data))
        except (IOError):
            print("File was not found:")
            print(node_path)

        return graph

    @staticmethod
    def remove_redundant_nodes(graph):

        node_types = nx.get_node_attributes(graph, "type")

        nodes_to_remove = [
            x
            for x, y in graph.nodes(data=True)
            if y["type"] == "TA" or y["type"] == "L" or y["type"] == "YA"
        ]

        graph.remove_nodes_from(nodes_to_remove)

        return graph

    @staticmethod
    def remove_iso_nodes(graph):
        graph.remove_nodes_from(list(nx.isolates(graph)))
        return graph

    @staticmethod
    def remove_iso_analyst_nodes(graph):
        analyst_nodes = []
        isolated_nodes = list(nx.isolates(graph))
        for node in isolated_nodes:
            if graph.nodes[node]["type"] == "L":
                analyst_nodes.append(node)
        graph.remove_nodes_from(analyst_nodes)
        return graph

    @staticmethod
    def get_eigen_centrality(graph):
        try:
            cent = nx.eigenvector_centrality_numpy(graph)
        except Exception as e:
            cent = nx.degree_centrality(graph)
            logger.error(f"Failed to run nx.eigenvector_centrality_numpy(graph): {e}")

        nx.set_node_attributes(graph, cent, "central")
        i_nodes = [
            (x, y["central"], y["text"]) for x, y in graph.nodes(data=True) if y["type"] == "I"
        ]
        return i_nodes

    @staticmethod
    def sort_by_centrality(i_nodes):
        sorted_by_second = sorted(i_nodes, key=lambda tup: tup[1])
        ordered_ids = [(i[0], i[2]) for i in sorted_by_second]

        return ordered_ids

    @staticmethod
    def list_nodes(graph):
        return list(graph)

    @staticmethod
    def get_s_node_list(graph):
        s_nodes = [
            x
            for x, y in graph.nodes(data=True)
            if y["type"] == "MA" or y["type"] == "RA" or y["type"] == "CA" or y["type"] == "PA"
        ]
        return s_nodes

    @staticmethod
    def get_l_node_list(graph):
        l_nodes = [(x, y["text"]) for x, y in graph.nodes(data=True) if y["type"] == "L"]
        return l_nodes

    @staticmethod
    def get_i_node_list(graph):
        i_nodes = [(x, y["text"]) for x, y in graph.nodes(data=True) if y["type"] == "I"]
        return i_nodes

    @staticmethod
    def get_divergent_nodes(graph):
        list_of_nodes = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(graph.successors(v))
            if len(node_pres) > 1:
                list_of_nodes.append(v)
        return list_of_nodes

    @staticmethod
    def get_loc_prop_pair(graph):
        i_node_ids = [x for x, y in graph.nodes(data=True) if y["type"] == "I"]
        locution_prop_pair = []
        for node_id in i_node_ids:
            preds = list(graph.predecessors(node_id))
            for pred in preds:
                node_type = graph.nodes[pred]["type"]
                node_text = graph.nodes[pred]["text"]

                if node_type == "YA" and node_text != "Agreeing":
                    ya_preds = list(graph.predecessors(pred))
                    for ya_pred in ya_preds:
                        pred_node_type = graph.nodes[ya_pred]["type"]
                        pred_node_text = graph.nodes[ya_pred]["text"]

                        if pred_node_type == "L":
                            locution_prop_pair.append((ya_pred, node_id))
        return locution_prop_pair

    @staticmethod
    def get_child_edges(graph):
        list_of_nodes = []
        list_of_edges = []

        for v in list(graph.nodes):
            node_pres = []
            node_pres = list(nx.ancestors(graph, v))
            list_of_nodes.append((v, node_pres))
            edges = []
            edges = list(nx.edge_dfs(graph, v, orientation="reverse"))
            res_list = []
            res_list = [(x[0], x[1]) for x in edges]
            list_of_edges.append((v, res_list))

        return list_of_nodes, list_of_edges

    @staticmethod
    def get_ras(graph):
        ra_nodes = [x for x, y in graph.nodes(data=True) if y["type"] == "RA"]
        return ra_nodes

    @staticmethod
    def get_yas(graph):
        ya_nodes = [x for x, y in graph.nodes(data=True) if y["type"] == "YA"]
        return ya_nodes

    @staticmethod
    def get_cas(graph):
        ca_nodes = [x for x, y in graph.nodes(data=True) if y["type"] == "CA"]
        return ca_nodes

    @staticmethod
    def get_mas(graph):
        ma_nodes = [x for x, y in graph.nodes(data=True) if y["type"] == "MA"]
        return ma_nodes

    @staticmethod
    def get_ra_i_nodes(graph, ras):
        ra_tups = []
        for ra in ras:
            node_succ = list(graph.successors(ra))
            i_1 = node_succ[0]
            i_1_text = graph.nodes[i_1]["text"]
            node_pres = list(graph.predecessors(ra))

            for n in node_pres:
                n_type = graph.nodes[n]["type"]
                if n_type == "I":
                    i_2 = n
                    i_2_text = graph.nodes[i_2]["text"]
                    break

            ra_tup = (ra, i_1_text, i_2_text)
            ra_tups.append(ra_tup)
        return ra_tups
