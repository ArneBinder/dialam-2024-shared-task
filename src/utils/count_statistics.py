import argparse
import io
import json
import os
from collections import Counter

import pandas as pd


def generate_stats_table(dir_name):
    node2node_stats = dict()
    for filename in os.listdir(dir_name):
        with open(os.path.join(dir_name, filename)) as f:
            data = json.load(f)
        # Store the node mapping
        node_id2node = dict()
        for n in data["nodes"]:
            node_id2node[n["nodeID"]] = n
        # Collect the edges and their labels (they come from the "scheme" annotation of the corresponding nodes if available)
        for e in data["edges"]:
            from_node = node_id2node[e["fromID"]]
            from_node_type = from_node["type"]
            from_node_label = from_node["scheme"] if "scheme" in from_node else "No Label"
            to_node = node_id2node[e["toID"]]
            to_node_type = to_node["type"]
            to_node_label = to_node["scheme"] if "scheme" in to_node else "No Label"
            transition_label = from_node_label if from_node_label != "No Label" else to_node_label
            if not (from_node_type in node2node_stats):
                node2node_stats[from_node_type] = dict()
            if not (to_node_type in node2node_stats[from_node_type]):
                node2node_stats[from_node_type][to_node_type] = dict()
            if not (transition_label in node2node_stats[from_node_type][to_node_type]):
                node2node_stats[from_node_type][to_node_type][transition_label] = 0
            node2node_stats[from_node_type][to_node_type][transition_label] += 1

    all_node_types = list(node2node_stats.keys())
    html_table = "<table>"
    html_table += "<tr><th>to_node &rarr;<br/>from_node &darr;</th>"
    # Table header
    for node in all_node_types:
        html_table += "<th>" + node + "</th>"
    html_table += "</tr>"
    # Each node type has a separate row with valid transitions
    for from_node in all_node_types:
        html_table += "<tr>"
        html_table += "<td><b>" + from_node + "</b></td>"
        from_node_stats = dict()
        for to_node in all_node_types:
            edge_labels = []
            if to_node in node2node_stats[from_node]:
                edge_labels = [(k, v) for k, v in node2node_stats[from_node][to_node].items()]
                edge_labels.sort(key=lambda tup: tup[1], reverse=True)
            if len(edge_labels) == 0:
                to_node_stats = "-"
            else:
                to_node_stats = " ".join(
                    [tpl[0].replace(" ", "") + "-" + str(tpl[1]) + "</br>" for tpl in edge_labels]
                )
            html_table += "<td>" + to_node_stats + "</td>"
        html_table += "</tr>"
    html_table += "</table>"
    # Conversion to pandas DataFrame is not necessary here but it makes the formatting a bit nicer
    df = pd.read_html(io.StringIO(html_table))[0]
    html_table = df.to_html(index=False)
    return html_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    args = parser.parse_args()
    html_table = generate_stats_table(args.nodeset_dir)
    print(html_table)
