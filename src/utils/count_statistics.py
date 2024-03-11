import argparse
import io
import json
import os
from collections import Counter

import pandas as pd


def count_transitions(
    from_input_node_type,
    to_input_node_type,
    transitions_dict,
    node_id2node,
    edges,
):
    for e in edges:
        from_node_id = e["fromID"]
        from_node = node_id2node[from_node_id]
        from_node_type = from_node["type"]

        to_node_id = e["toID"]
        to_node = node_id2node[to_node_id]
        to_node_type = to_node["type"]

        node_transition_types = ["YA", "MA", "RA", "CA", "TA"]

        if from_node_type == from_input_node_type:
            for e2 in edges:
                if e["edgeID"] == e2["edgeID"]:
                    continue
                from_e2_node_type = node_id2node[e2["fromID"]]["type"]
                to_e2_node_type = node_id2node[e2["toID"]]["type"]
                if (
                    e["toID"] == e2["fromID"]
                    and to_e2_node_type == to_input_node_type
                    and from_e2_node_type in node_transition_types
                ):
                    transition_type = (
                        from_node_type
                        + " &rarr; "
                        + to_node_type
                        + " &rarr; "
                        + to_input_node_type
                    )
                    transition_label = to_node["scheme"] if "scheme" in to_node else "No Label"
                    if not (transition_type in transitions_dict):
                        transitions_dict[transition_type] = dict()
                    if not (transition_label in transitions_dict[transition_type]):
                        transitions_dict[transition_type][transition_label] = 0
                    transitions_dict[transition_type][transition_label] += 1


def generate_transitions_per_input_table(dir_name):
    transitions_dict = dict()
    input_node_types = ["L", "I", "TA"]
    input_pairs = [(i, j) for i in input_node_types for j in input_node_types]

    for filename in os.listdir(dir_name):
        with open(os.path.join(dir_name, filename)) as f:
            data = json.load(f)
        # Store the node mapping
        node_id2node = dict()
        for n in data["nodes"]:
            node_id2node[n["nodeID"]] = n
        edges = data["edges"]
        # Count transitions for each pair of nodes
        for from_input_node_type, to_input_node_type in input_pairs:
            count_transitions(
                from_input_node_type, to_input_node_type, transitions_dict, node_id2node, edges
            )
    transitions = [(k, sum(transitions_dict[k].values())) for k in transitions_dict]
    transitions.sort(key=lambda tup: tup[1], reverse=True)
    transitions = [t[0] for t in transitions]

    # Create html table
    html_table = "<table>"
    html_table += "<tr><th>input nodes</th>"
    # Table header
    for node in input_node_types:
        html_table += "<th>" + node + "</th>"
    html_table += "</tr>"
    for from_node in input_node_types:
        html_table += "<tr>"
        html_table += "<td><b>" + from_node + "</b></td>"
        for to_node in input_node_types:
            transitions_counts = ""
            for transition in transitions:
                # Check that transition starts and ends with the correct nodes
                # and appears more than once in the training set
                # if it appears only once we consider it an annotation error/artefact
                if (
                    transition.startswith(from_node + " &rarr; ")
                    and transition.endswith(" &rarr; " + to_node)
                    and sum(transitions_dict[transition].values()) > 1
                ):
                    if len(transitions_counts) > 0:
                        transitions_counts += "###"
                    transitions_counts += transition + "###"
                    transitions_counts += " ".join(
                        [
                            k.replace(" ", "") + ": " + str(v) + "###"
                            for k, v in sorted(
                                transitions_dict[transition].items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                        ]
                    )
            if len(transitions_counts) == 0:
                transitions_counts = "-"
            html_table += "<td>" + transitions_counts + "</td>"

        html_table += "</tr>"
    html_table += "</table>"
    df = pd.read_html(io.StringIO(html_table))[0]
    markdown_table = df.to_markdown(index=False)
    markdown_table = markdown_table.replace(
        "###", "<br>"
    )  # We use "###" to indicate the line break,
    # unfortunately, if we insert <br> directly
    # it will be removed after the html-to-df conversion,
    # hence we have to insert it back
    return markdown_table


def generate_stats_table(dir_name):
    node2node_stats = dict()
    for filename in os.listdir(dir_name):
        with open(os.path.join(dir_name, filename)) as f:
            data = json.load(f)
        # Store the node mapping
        node_id2node = dict()
        for n in data["nodes"]:
            node_id2node[n["nodeID"]] = n
        # Collect the edges and their labels:
        # they come from the "scheme" annotation of the corresponding nodes if available
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
    df = pd.read_html(io.StringIO(html_table))[0]
    markdown_table = df.to_markdown(index=False)
    return markdown_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    args = parser.parse_args()
    # markdown_table = generate_stats_table(args.nodeset_dir)
    transitions_per_input_table = generate_transitions_per_input_table(args.nodeset_dir)
    print(transitions_per_input_table)
