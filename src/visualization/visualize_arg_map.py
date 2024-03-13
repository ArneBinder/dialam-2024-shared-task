"""How to use the visualization script:

(1) Make sure that graphviz is installed on your system.
E.g., on Linux you can install it with `apt-get install graphviz`.
If you are getting "RuntimeError: Make sure the Graphviz executables are on your system's path" please check the following discussion: https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft

(2) If you created the environment based on requirements.txt you should already have the correct version installed with pip. Otherwise, you should execute `pip install graphviz`

(3) Run the script as follows:

$ python3 src/visualization/visualize_arg_map.py path_to_nodesets path_to_store_visualizations nodeset_id

For example:
$ python3 src/visualization/visualize_arg_map.py data/dialam-2024-dataset visualizations 21388
`data/dialam-2024-dataset` is the path to the dataset with the nodesets in JSON format
`visualizations` is the path to the directory for storing visualizations
`21388` is the nodeset id (in this example for `nodeset21388.json`).
"""

import argparse
import json
import os

from graphviz import Digraph


def chunk_text(text: str, tokens_per_chunk: int = 10) -> str:
    node_text_split = ""
    chunk_tokens = []
    for token in text.split():
        chunk_tokens.append(token)
        if len(chunk_tokens) > tokens_per_chunk:
            node_text_split += " ".join(chunk_tokens) + "<br/>"
            chunk_tokens = []
    if len(chunk_tokens) > 0:
        node_text_split += " ".join(chunk_tokens)
    return node_text_split


def assemble_node_text(node: dict) -> str:
    if node["type"] in ["L", "I"]:
        text = chunk_text(node["text"])
    else:
        text = node["scheme"] if "scheme" in node else ""
    timestamp = node["timestamp"].split()[1] if "timestamp" in node else ""
    node_text = f"<<b>{node['type']}</b> {node['nodeID']} {timestamp}<br/>{text}>"
    return node_text


def add_node(node: dict, graph) -> None:
    if node["type"] in ["L", "I"]:
        shape = "box"
    else:
        shape = "oval"
    graph.node(node["nodeID"], label=assemble_node_text(node), shape=shape)


def create_visualization(args):
    # Read the JSON data
    nodeset_dir = args.nodeset_dir
    output_dir = args.output_dir
    nodeset = args.nodeset
    filename = os.path.join(nodeset_dir, "nodeset" + str(nodeset) + ".json")
    with open(filename) as f:
        data = json.load(f)

    # Collect all nodes and store the corresponding text & node type in a dictionary
    node_id2node = {n["nodeID"]: n for n in data["nodes"]}

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
        node_type_from = node_id2node[node_from]["type"]
        node_type_to = node_id2node[node_to]["type"]
        if (node_type_from in ta_edge_node_types) and (node_type_to in ta_edge_node_types):
            ta_edges.append(node_tuple)
        elif node_type_from == "YA" or node_type_to == "YA":
            ya_edges.append(node_tuple)
        elif (node_type_from in s_edge_node_types) and (node_type_to in s_edge_node_types):
            s_edges.append(node_tuple)

    # Sort by the first node ids
    ya_edges = sorted(ya_edges, key=lambda x: x[0])
    s_edges = sorted(s_edges, key=lambda x: x[0])
    ta_edges = sorted(ta_edges, key=lambda x: x[0])

    # Create two clusters (subgraphs) for I-nodes and L-nodes
    g = Digraph("G", filename="cluster.gv", format="png")
    g.attr(rankdir="RL")
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
        for from_id, to_id in ta_edges:
            add_node(node=node_id2node[from_id], graph=c)
            add_node(node=node_id2node[to_id], graph=c)
        for from_id, to_id in ya_edges:
            from_node = node_id2node[from_id]
            if from_node["type"] == "L":
                add_node(node=from_node, graph=c)
            to_node = node_id2node[to_id]
            if to_node["type"] == "L":
                add_node(node=to_node, graph=c)
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
        for from_id, to_id in s_edges:
            add_node(node=node_id2node[from_id], graph=c)
            add_node(node=node_id2node[to_id], graph=c)
        for from_id, to_id in ya_edges:
            from_node = node_id2node[from_id]
            if from_node["type"] == "I":
                add_node(node=from_node, graph=c)
            to_node = node_id2node[to_id]
            if to_node["type"] == "I":
                add_node(node=to_node, graph=c)
        c.edge_attr["constraint"] = "false"
        c.edges(s_edges)

    # Add the rest of edges that connect L- and I-nodes
    for from_id, to_id in ya_edges:
        add_node(node=node_id2node[from_id], graph=g)
        add_node(node=node_id2node[to_id], graph=g)
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
