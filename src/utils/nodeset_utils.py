import json
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Tuple, TypeVar, Union

import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_nodeset_ids_from_directory(nodeset_dir: str) -> List[str]:
    """Get the IDs of all nodesets in a directory."""

    return [
        f.split("nodeset")[1].split(".json")[0]
        for f in os.listdir(nodeset_dir)
        if f.endswith(".json")
    ]


def read_nodeset(nodeset_dir: str, nodeset_id: str) -> Dict[str, Any]:
    """Read a nodeset with a given ID from a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename) as f:
        return json.load(f)


def write_nodeset(nodeset_dir: str, nodeset_id: str, data: Dict[str, Any]) -> None:
    """Write a nodeset with a given ID to a directory."""

    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def process_all_nodesets(
    nodeset_dir: str, func: Callable[..., T], show_progress: bool = True, **kwargs
) -> Iterator[Tuple[str, Union[T, Exception]]]:
    """Process all nodesets in a directory.

    Args:
        nodeset_dir: The directory containing the nodesets.
        func: The function to apply to each nodeset.
        show_progress: Whether to show a progress bar.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        A tuple containing the nodeset ID and the result of applying the function.
        If an exception occurs, the result will be the exception.
    """

    nodeset_ids = get_nodeset_ids_from_directory(nodeset_dir=nodeset_dir)
    for nodeset_id in tqdm.tqdm(
        nodeset_ids, desc="Processing nodesets", disable=not show_progress
    ):
        try:
            result = func(
                nodeset_dir=nodeset_dir,
                nodeset_id=nodeset_id,
                **kwargs,
            )
            yield nodeset_id, result
        except Exception as e:
            yield nodeset_id, e


def get_node_ids(node_id2node: Dict[str, Any], allowed_node_types: List[str]) -> List[str]:
    """Get the IDs of nodes with a given type."""

    return [
        node_id for node_id, node in node_id2node.items() if node["type"] in allowed_node_types
    ]
