from __future__ import annotations

import argparse
import io
import json
import logging
import os
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


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

        node_transition_types = ["YA", "MA", "RA", "CA", "TA", "S"]
        s_nodes = ["MA", "RA", "CA", "S"]

        if from_node_type in s_nodes:
            from_node_type = "S"
        if to_node_type in s_nodes:
            to_node_type = "S"

        if from_node_type == from_input_node_type:
            for e2 in edges:
                if e["edgeID"] == e2["edgeID"]:
                    continue
                from_e2_node_type = node_id2node[e2["fromID"]]["type"]
                to_e2_node_type = node_id2node[e2["toID"]]["type"]
                if from_e2_node_type in s_nodes:
                    from_e2_node_type = "S"
                if to_e2_node_type in s_nodes:
                    to_e2_node_type = "S"
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
                    transition_label = to_node["text"] if "text" in to_node else "No Label"
                    if not (transition_type in transitions_dict):
                        transitions_dict[transition_type] = dict()
                    if not (transition_label in transitions_dict[transition_type]):
                        transitions_dict[transition_type][transition_label] = 0
                    transitions_dict[transition_type][transition_label] += 1


def generate_transitions_per_input_table(dir_name):
    transitions_dict = dict()
    input_node_types = ["I", "L", "S", "TA", "YA"]
    input_pairs = [(i, j) for i in input_node_types for j in input_node_types]

    for filename in tqdm(os.listdir(dir_name)):
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
                if transition.startswith(from_node + " &rarr; ") and transition.endswith(
                    " &rarr; " + to_node
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


def get_relation_and_node_dicts(file_name: str) -> tuple[dict, dict, set]:

    # load the nodes and edges from the json file
    with open(file_name) as f:
        data = json.load(f)

    # collect all incoming and outgoing edges for each node
    src2trg = defaultdict(set)
    trg2src = defaultdict(set)
    edge_set = set()
    for edge in data["edges"]:
        src = edge["fromID"]
        trg = edge["toID"]
        src2trg[src].add(trg)
        trg2src[trg].add(src)
        edge_set.add((src, trg))

    # collect the node-id-mapping and relation ids
    node_id2node = dict()
    type2ids = defaultdict(set)
    for node in data["nodes"]:
        node_id = node.pop("nodeID")
        node_id2node[node_id] = node
        node_type = node["type"]
        if node_type in ["RA", "CA", "MA"]:
            node_type = "S"
        type2ids[node_type].add(node_id)

    relation_type2arg_types = {
        "TA": [("L", "L")],
        "S": [("I", "I")],
        "YA": [("L", "I"), ("TA", "S")],
    }

    # assemble relations
    filtered_used_edges = set()
    filtered_relations = defaultdict(list)
    all_relations = defaultdict(list)
    all_used_edges = set()
    for rel_type, arg_types in relation_type2arg_types.items():
        for rel_id in type2ids[rel_type]:
            for src_id in trg2src[rel_id]:
                for trg_id in src2trg[rel_id]:
                    all_relations[(src_id, trg_id)].append(node_id2node[rel_id])
                    all_used_edges.add((src_id, rel_id))
                    all_used_edges.add((rel_id, trg_id))
                    for src_type, trg_type in arg_types:
                        if src_id in type2ids[src_type] and trg_id in type2ids[trg_type]:
                            filtered_relations[(src_id, trg_id)].append(node_id2node[rel_id])
                            filtered_used_edges.add((src_id, rel_id))
                            filtered_used_edges.add((rel_id, trg_id))

    # check for multiple relations between the same nodes
    for (src, trg), rels in filtered_relations.items():
        if len(rels) > 1:
            src_node = node_id2node[src]
            trg_node = node_id2node[trg]
            relation_types = set(rel["type"] for rel in rels)
            if relation_types != {"TA"}:
                logger.warning(
                    f"{file_name}: Multiple relation nodes (types: {relation_types}) between {src} "
                    f"(type: {src_node['type']}) and {trg} (type: {trg_node['type']})"
                )

    # unused_edges = edge_set - filtered_used_edges
    # return filtered_relations, node_id2node, unused_edges
    unused_edges = edge_set - all_used_edges
    return all_relations, node_id2node, unused_edges


def relation_dict_to_df(relation_dict) -> pd.DataFrame:
    """Converts a dict that maps pairs of argument types to anything into a pivot table, e.g.
    {
        ("A", "B"): x,
        ("A", "C"): y,
        ("B", "C"): z,
    }
    will be converted to
    |       | A    | B    | C    |
    |-------|------|------|------|
    | A     |      | x    | y    |
    | B     |      |      | z    |
    | C     |      |      |      |
    """
    s = pd.Series(relation_dict)
    df = pd.DataFrame(s).unstack()
    df.columns = df.columns.droplevel()
    return df


def relation_types_to_count_string(relation_types: list | float) -> str:
    """Converts a list of relation types to a string with counts. Also handles the case where the
    input is a float, which is the case when the input is NaN.

    Example: ["A", "B", "A"] -> "A: 2<br>B: 1"
    """
    if not isinstance(relation_types, list):
        return "-"
    relation_types_wo_white_space = [rt.replace(" ", "") for rt in relation_types]
    counter = Counter(relation_types_wo_white_space)
    # sort by key and create a string with counts
    return "<br>".join([f"{k}: {counter[k]}" for k in sorted(counter)])


def maybe_convert_node_class_to_type(class_name: str) -> str:
    if class_name in ["RA", "CA", "MA"]:
        return "S"
    return class_name


def generate_relation_stats_table(dir_name) -> str:
    """Generates a markdown table with the counts of relation types between different argument
    types. The table has the following format:

    .. code-block:: markdown

        |       | A    | B    | C    |
        |-------|------|------|------|
        | A     |      | a: 3 | c: 5 |
        |       |      | b: 2 |      |
        | B     |      |      | d: 10|
        | C     |      |      |      |
    """
    argument_types_to_relation_type = defaultdict(list)
    unused_edge_args = defaultdict(list)
    for filename in os.listdir(dir_name):
        relations, all_nodes, unused_edges = get_relation_and_node_dicts(
            os.path.join(dir_name, filename)
        )
        for (source_id, target_id), relation_nodes in relations.items():
            source_node = all_nodes[source_id]
            target_node = all_nodes[target_id]
            arg_types = (
                maybe_convert_node_class_to_type(source_node["type"]),
                maybe_convert_node_class_to_type(target_node["type"]),
            )
            for relation_node in relation_nodes:
                relation_node_type = maybe_convert_node_class_to_type(relation_node["type"])
                argument_types_to_relation_type[arg_types].append(
                    f"{relation_node_type}/{relation_node['text']}"
                )
        for src, trg in unused_edges:
            unused_edge_args[(all_nodes[src]["type"], all_nodes[trg]["type"])].append(
                (filename, src, trg)
            )

    if len(unused_edge_args) > 0:
        unused_edge_arg_counter = Counter({k: len(v) for k, v in unused_edge_args.items()})
        logger.warning(f"Unused edges: {unused_edge_arg_counter}")

    df = relation_dict_to_df(argument_types_to_relation_type)
    df_count_strings = df.map(relation_types_to_count_string)
    return df_count_strings.to_markdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    args = parser.parse_args()
    # markdown_table = generate_stats_table(args.nodeset_dir)

    # tatiana's statistics
    # transitions_per_input_table = generate_transitions_per_input_table(args.nodeset_dir)
    # print(transitions_per_input_table)

    # arne's statistics
    result = generate_relation_stats_table(args.nodeset_dir)
    print(result)
