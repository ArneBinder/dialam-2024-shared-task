import argparse
import json
import os

from graphviz import Digraph


def create_visualization(args):
    # Read the JSON data
    nodeset_dir = args.nodeset_dir
    output_dir = args.output_dir
    nodeset = args.nodeset
    filename = os.path.join(nodeset_dir, "nodeset" + str(nodeset) + ".json")
    with open(filename) as f:
        data = json.load(f)

    # Collect all nodes and store the corresponding text & node type in a dictionary
    node_id2text = dict()
    node2type = dict()
    for n in data["nodes"]:
        if n["type"] in ["L", "I"]:
            node_text_splitted = ""
            chunk_tokens = []
            for token in n["text"].split():
                chunk_tokens.append(token)
                if len(chunk_tokens) > 10:
                    node_text_splitted += " ".join(chunk_tokens) + "<br/>"
                    chunk_tokens = []
            if len(chunk_tokens) > 0:
                node_text_splitted += " ".join(chunk_tokens)
            node_text = "<" + n["nodeID"] + ": " + n["type"] + "<br/>" + node_text_splitted + ">"
        else:
            node_text = (
                "<"
                + n["nodeID"]
                + ": "
                + n["type"]
                + "<br/>"
                + (n["scheme"] if "scheme" in n else "")
                + ">"
            )
        node_id2text[n["nodeID"]] = node_text
        node2type[n["nodeID"]] = n["type"]

    # Collect all edges
    ta_edges = []  # default transitions for L-nodes
    ya_edges = []  # transitions between L and I-nodes, S and TA nodes or I and TA nodes
    # sometimes YA nodes also mean transitions between two L nodes!
    # E.g., see nodeset18291.json nodes 540800-540805 for L-L transition
    s_edges = []  # transitions between I-nodes

    ta_edge_node_types = ["L", "TA"]
    s_edge_node_types = ["MA", "RA", "CA", "I"]
    for e in data["edges"]:
        node_from = e["fromID"]
        node_to = e["toID"]
        node_tuple = (node_from, node_to)
        node_type_from = node2type[node_from]
        node_type_to = node2type[node_to]
        if (node_type_from in ta_edge_node_types) and (node_type_to in ta_edge_node_types):
            ta_edges.append(node_tuple)
        elif (
            (node_type_from == "L" and node_type_to == "YA")
            or (node_type_from == "YA" and node_type_to == "I")
            or (node_type_from == "YA" and node_type_to == "L")
            or (node_type_from == "YA" and node_type_to == "TA")
            or (node_type_from in s_edge_node_types and node_type_to == "YA")
            or (node_type_from == "I" and node_type_to == "YA")
        ):
            ya_edges.append(node_tuple)
        elif (node_type_from in s_edge_node_types) and (node_type_to in s_edge_node_types):
            s_edges.append(node_tuple)

    # Sort by the first node ids
    ya_edges = sorted(ya_edges, key=lambda x: x[0])
    s_edges = sorted(s_edges, key=lambda x: x[0])
    ta_edges = sorted(ta_edges, key=lambda x: x[0])

    # Create two clusters (subgraphs) for I-nodes and L-nodes
    g = Digraph("G", filename="cluster.gv", format="png")
    g.attr(rankdir="LR")
    g.attr(splines="ortho")
    g.attr(overlap="false")

    with g.subgraph(name="cluster_L_nodes") as c:
        # Set cluster attributes for L-nodes
        c.attr(label="L-nodes")
        c.attr(color="blue")
        c.attr(rankdir="TB")
        c.node_attr["style"] = "filled"
        c.node_attr["fillcolor"] = "lightcyan"
        c.attr(splines="ortho")
        c.attr(overlap="false")
        # Add TA and YA edges that connect to L-nodes
        for e in ta_edges:
            c.node(
                e[0], label=node_id2text[e[0]], shape="box" if node2type[e[0]] == "L" else "oval"
            )
            c.node(
                e[1], label=node_id2text[e[1]], shape="box" if node2type[e[1]] == "L" else "oval"
            )
        for e in ya_edges:
            if node2type[e[0]] == "L":
                c.node(e[0], label=node_id2text[e[0]], shape="box")
            elif node2type[e[1]] == "L":
                c.node(e[1], label=node_id2text[e[1]], shape="box")
        c.edge_attr["constraint"] = "false"
        c.edges(ta_edges)

    with g.subgraph(name="cluster_I_nodes") as c:
        # Set cluster attributes for I-nodes
        c.attr(label="I-nodes")
        c.attr(color="red")
        c.attr(rankdir="TB")
        c.node_attr["style"] = "filled"
        c.node_attr["fillcolor"] = "mistyrose"
        c.attr(splines="ortho")
        c.attr(overlap="false")
        # Add S and YA edges that connect to I-nodes
        for e in s_edges:
            c.node(
                e[0], label=node_id2text[e[0]], shape="box" if node2type[e[0]] == "I" else "oval"
            )
            c.node(
                e[1], label=node_id2text[e[1]], shape="box" if node2type[e[1]] == "I" else "oval"
            )
        for e in ya_edges:
            if node2type[e[1]] == "I":
                c.node(e[1], label=node_id2text[e[1]], shape="box")
        c.edge_attr["constraint"] = "false"
        c.edges(s_edges)

    # Add the rest of edges that connect L- and I-nodes
    for e in ya_edges:
        g.node(e[0], label=node_id2text[e[0]])
        g.node(e[1], label=node_id2text[e[1]])
    g.edges(ya_edges)

    # Render the final graph
    filename_visualized = os.path.join(output_dir, "nodeset" + str(nodeset) + ".gv")
    g.render(filename_visualized, view=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    parser.add_argument(
        "output_dir", type=str, help="path to the directory where to store the visualized results"
    )
    parser.add_argument("nodeset", type=int, help="nodeset (argument map) id")
    args = parser.parse_args()
    create_visualization(args)
