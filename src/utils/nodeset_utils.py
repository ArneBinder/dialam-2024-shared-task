import json
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Tuple, TypeVar, Union

import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_nodeset_ids_from_directory(nodeset_dir: str) -> List[str]:
    return [
        f.split("nodeset")[1].split(".json")[0]
        for f in os.listdir(nodeset_dir)
        if f.endswith(".json")
    ]


def read_nodeset(nodeset_dir: str, nodeset_id: str) -> Dict[str, Any]:
    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename) as f:
        return json.load(f)


def write_nodeset(nodeset_dir: str, nodeset_id: str, data: Dict[str, Any]) -> None:
    filename = os.path.join(nodeset_dir, f"nodeset{nodeset_id}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def process_all_nodesets(
    nodeset_dir: str, func: Callable[..., T], show_progress: bool = True, **kwargs
) -> Iterator[Tuple[str, Union[T, Exception]]]:
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
