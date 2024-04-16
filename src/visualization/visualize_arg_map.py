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

Note: If no nodeset id is provided, the script will visualize all nodesets in the directory.
"""

import pyrootutils

pyrootutils.setup_root(search_from=__file__, indicator=[".project-root"], pythonpath=True)

import argparse
import datetime
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from graphviz import Digraph

from src.utils.nodeset_utils import sort_nodes_by_hierarchy

logger = logging.getLogger(__name__)


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
    text = node["text"]
    if node["type"] in ["L", "I"]:
        text = chunk_text(text)
    timestamp = node["timestamp"].split()[1] if "timestamp" in node else ""
    node_text = f"<<b>{node['type']}</b> {node['nodeID']} {timestamp}<br/>{text}>"
    return node_text


def add_node(node: dict, graph) -> None:
    if node["type"] in ["L", "I"]:
        shape = "box"
    else:
        shape = "oval"
    graph.node(node["nodeID"], label=assemble_node_text(node), shape=shape)


def filter_edges(
    edges: Set[Tuple[str, str]], allowed_source_ids: List[str], allowed_target_ids: List[str]
) -> Set[Tuple[str, str]]:
    return set(
        (src, trg)
        for (src, trg) in edges
        if src in allowed_source_ids and trg in allowed_target_ids and trg != src
    )


def create_visualization(
    nodeset_dir: str, output_dir: str, nodeset: Optional[str] = None, view: bool = True
):
    # if no nodeset id is provided, visualize all nodesets in the directory
    if nodeset is None:
        nodeset_ids = [
            f.split("nodeset")[1].split(".json")[0]
            for f in os.listdir(nodeset_dir)
            if f.endswith(".json")
        ]
        for nodeset in nodeset_ids:
            try:
                create_visualization(
                    nodeset_dir=nodeset_dir, output_dir=output_dir, nodeset=nodeset, view=False
                )
            except Exception as e:
                logger.error(f"nodeset={nodeset}: Failed to visualize: {e}")
        return

    # Read the JSON data
    filename = os.path.join(nodeset_dir, f"nodeset{nodeset}.json")
    with open(filename) as f:
        data = json.load(f)

    # edge related helper data structures
    src2targets = defaultdict(list)
    trg2sources = defaultdict(list)
    edges = set()
    for edge_dict in data["edges"]:
        src2targets[edge_dict["fromID"]].append(edge_dict["toID"])
        trg2sources[edge_dict["toID"]].append(edge_dict["fromID"])
        edges.add((edge_dict["fromID"], edge_dict["toID"]))

    # node related helper data structures
    node_id2node = {n["nodeID"]: n for n in data["nodes"]}
    node_types2node_ids: Dict[str, Set[str]] = defaultdict(set)
    disconnected_node_ids = set()
    duplicate_node_ids = set()
    for n in data["nodes"]:
        node_type = n["type"]
        if node_type in ["RA", "CA", "MA"]:
            node_type = "S"

        # only collect connected nodes
        if n["nodeID"] in src2targets or n["nodeID"] in trg2sources:
            if n["nodeID"] not in node_types2node_ids[node_type]:
                node_types2node_ids[node_type].add(n["nodeID"])
            else:
                duplicate_node_ids.add(n["nodeID"])
        else:
            disconnected_node_ids.add(n["nodeID"])

    if len(duplicate_node_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Duplicate nodes: {duplicate_node_ids}")

    # Create two clusters (subgraphs) for I related nodes and L related nodes
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

        # sort L-nodes by hierarchy, i.e. that parents are always before children
        l_node_ids_sorted = sort_nodes_by_hierarchy(
            node_ids=node_types2node_ids["L"], edges=data["edges"]
        )

        # add L-nodes and all connected TA-nodes
        l_site_node_ids = []
        for l_site_node_id in l_node_ids_sorted:
            # do not visualize disconnected nodes
            if l_site_node_id in disconnected_node_ids:
                continue
            add_node(node=node_id2node[l_site_node_id], graph=c)
            l_site_node_ids.append(l_site_node_id)
            for ya_trg_node_id in src2targets[l_site_node_id]:
                ya_trg_node = node_id2node[ya_trg_node_id]
                if ya_trg_node["type"] == "TA":
                    add_node(node=ya_trg_node, graph=c)
                    l_site_node_ids.append(ya_trg_node_id)
        c.edge_attr["constraint"] = "false"
        l_site_edges = filter_edges(edges, l_site_node_ids, l_site_node_ids)
        c.edges(l_site_edges)

    # collect YA nodes in the order of L-site-nodes (L- and TA-nodes), i.e. they need to be connected to L-site-nodes
    ya_node_ids = []
    l_anchor_nodes = defaultdict(list)
    for l_site_node_id in l_site_node_ids:
        if l_site_node_id in src2targets:
            for ya_trg_node_id in src2targets[l_site_node_id]:
                if node_id2node[ya_trg_node_id]["type"] == "YA":
                    ya_node_ids.append(ya_trg_node_id)
                    if node_id2node[l_site_node_id]["type"] == "L":
                        l_anchor_nodes[ya_trg_node_id].append(l_site_node_id)

    with g.subgraph(name="cluster_I_nodes") as c:
        # Set cluster attributes for I-nodes
        c.attr(label="I-nodes")
        c.attr(color="red")
        c.attr(rankdir="TB")
        c.node_attr["style"] = "filled"
        c.node_attr["fillcolor"] = "mistyrose"
        c.attr(splines="ortho")
        c.attr(overlap="false")

        # sort L-nodes by order of anchor nodes
        l_node_ids_to_idx = {node_id: idx for idx, node_id in enumerate(l_node_ids_sorted)}
        max_idx = len(l_node_ids_sorted)
        i_nodes_to_max_anchor_idx = {
            i_node_id: max(
                [l_node_ids_to_idx[anchor_node_id] for anchor_node_id in anchor_node_ids]
            )
            for i_node_id, anchor_node_ids in l_anchor_nodes.items()
        }

        i_node_ids_sorted = sorted(
            node_types2node_ids["I"],
            key=lambda x: i_nodes_to_max_anchor_idx.get(x, max_idx),
        )

        # Add I- and S-nodes in the order of YA-nodes
        i_node_ids_added = []
        i_site_node_ids = []
        for ya_node_id in ya_node_ids:
            for ya_trg_node_id in src2targets[ya_node_id]:
                ya_trg_node = node_id2node[ya_trg_node_id]
                if (
                    ya_trg_node["type"] in ["I", "RA", "CA", "MA"]
                    and ya_trg_node_id not in i_site_node_ids
                ):
                    add_node(node=node_id2node[ya_trg_node_id], graph=c)
                    i_site_node_ids.append(ya_trg_node_id)
                    if ya_trg_node["type"] == "I":
                        i_node_ids_added.append(ya_trg_node_id)

        if len(i_node_ids_sorted) != len(i_node_ids_added):
            logger.warning(
                f"nodeset={nodeset}: Missed I-nodes: {set(i_node_ids_sorted) - set(i_node_ids_added)}"
            )
        elif i_node_ids_sorted != i_node_ids_added:
            logger.warning(
                f"nodeset={nodeset}: I-nodes order mismatch: {i_node_ids_sorted} != {i_node_ids_added}"
            )

        c.edge_attr["constraint"] = "false"
        i_site_edges = filter_edges(edges, i_site_node_ids, i_site_node_ids)
        c.edges(i_site_edges)

    # Add the YA-nodes
    ya_edges = set()
    for node_id in ya_node_ids:
        add_node(node=node_id2node[node_id], graph=g)
        for src_node_id in trg2sources[node_id]:
            ya_edges.add((src_node_id, node_id))
        for trg_node_id in src2targets[node_id]:
            ya_edges.add((node_id, trg_node_id))
    g.edges(ya_edges)

    # warn about missed nodes
    if len(disconnected_node_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Disconnected nodes: {disconnected_node_ids}")
    missed_l_site_node_ids = (
        (set(node_types2node_ids["L"]) | set(node_types2node_ids["TA"]))
        - set(l_site_node_ids)
        - disconnected_node_ids
    )
    if len(missed_l_site_node_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Missed L-site nodes: {missed_l_site_node_ids}")
    missed_i_nodes_ids = (
        (set(node_types2node_ids["I"]) | set(node_types2node_ids["S"]))
        - set(i_site_node_ids)
        - disconnected_node_ids
    )
    if len(missed_i_nodes_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Missed I-site nodes: {missed_i_nodes_ids}")
    missed_ya_node_ids = set(node_types2node_ids["YA"]) - set(ya_node_ids) - disconnected_node_ids
    if len(missed_ya_node_ids) > 0:
        logger.warning(f"nodeset={nodeset}: Missed YA nodes: {missed_ya_node_ids}")

    # warn about missed edges
    missed_edges = edges - l_site_edges - i_site_edges - ya_edges
    if len(missed_edges) > 0:
        logger.warning(f"nodeset={nodeset}: Missed edges: {missed_edges}")

    # Render the final graph
    filename_visualized = os.path.join(output_dir, "nodeset" + str(nodeset) + ".gv")
    g.render(filename_visualized, view=view)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodeset_dir", type=str, help="path to the directory with nodesets in JSON format"
    )
    parser.add_argument(
        "output_dir", type=str, help="path to the directory where to store the visualized results"
    )
    parser.add_argument(
        "nodeset",
        nargs="?",
        type=str,
        help="nodeset (argument map) id. This is optional, if not provided, all nodesets of in the "
        "nodeset_dir are visualized",
        default=None,
    )
    args = vars(parser.parse_args())
    create_visualization(**args)
