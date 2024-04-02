"""How to use the evaluation script:

(1) Make sure that you have installed the following libraries (also specified in requirements.txt):
  - networkx # graphs
  - pycm # confusion matrix
  - segeval # segmentation
  - fuzzywuzzy # fuzzy string matching
  - numpy # used in matching.py
  - bs4 # used in matching.py

(2) Install GMatch4py (needed for graph matching) either directly with pip:
  pip install git+https://github.com/Jacobe2169/GMatch4py.git

  Or by cloning the repository first:
  git clone https://github.com/Jacobe2169/GMatch4py
  cd GMatch4py
  pip install .


(3) Run the script as follows:

$ python3 src/evaluation/evaluate.py predicted_nodeset_path gold_nodeset_path

For example:
$ python3 src/evaluation/evaluate.py data/predicted_nodeset1.json data/gold_nodeset1.json

The script prints to the console the following metrics: Kappa, CASS, F1, Accuracy, U-Alpha

You can find more detail about the CASS metric here: https://aclanthology.org/W16-2805.pdf

Note: This script is based on the following code (some adaptations were made to make it work
with the DialAM data): https://github.com/arg-tech/AMF-Evaluation-Metrics
"""

import argparse
import json
from typing import Any, Callable, Dict, List, Sequence, TypeVar

import matching
from pycm import ConfusionMatrix

T = TypeVar("T")


def get_kappa(confusion_matrix: ConfusionMatrix) -> float:
    """Calculate Kappa values (in range between -1 and 1).
    Args:
        confusion_matrix: Confusion matrix for a specific type of relation.

    Returns:
        Kappa value based on graph matching.
    """
    kappa = confusion_matrix.Kappa
    if kappa == "None":
        kappa = confusion_matrix.KappaNoPrevalence
    return kappa


# TODO: THIS IS NOT USED, REMOVE IT?
def text_similarity(
    predicted_data: Dict[str, List[Dict[str, Any]]], gold_data: Dict[str, List[Dict[str, Any]]]
) -> float:
    """Compute similarity between the segmented texts.
    Args:
        predicted_data: A dictionary with nodes, edges and locutions for the predicted nodeset.
        gold_data: A dictionary with nodes, edges and locutions for the gold nodeset.

    Returns:
        Segmentation similarity for texts from two different graphs.
    """

    text1 = predicted_data["text"]
    text2 = gold_data["text"]
    # Check if text1 is a dictionary with 'txt' key
    if isinstance(text1, dict) and "txt" in text1:
        text1 = text1["txt"]

    # Check if text2 is a dictionary with 'txt' key
    if isinstance(text2, dict) and "txt" in text2:
        text2 = text2["txt"]
    # Similarity between two texts
    ss = matching.get_similarity(text1, text2)
    return ss


def CASS_calculation(text_sim_ss: float, k_graph: float) -> float:
    """Compute the CASS metric: https://aclanthology.org/W16-2805.pdf
    Args:
        text_sim_ss: Text segmentation similarity.
        k_graph: Graph matching similarity.

    Returns:
        CASS metric (between 0 and 1).
    """
    if text_sim_ss > 0:
        return (float(k_graph) + float(text_sim_ss)) / 2
    else:
        return 0.0


def get_f1(confusion_matrix: ConfusionMatrix) -> float:
    """Compute macro F1 score based on the confusion matrices for each type of relations.
    Args:
        confusion_matrix: Confusion matrix for a specific type of relation.

    Returns:
        Macro F1 score.
    """
    f1 = confusion_matrix.F1_Macro
    if f1 == "None":
        f1 = 1.0
    return f1


def get_accuracy(confusion_matrix: ConfusionMatrix) -> float:
    """Compute accuracy score based on the confusion matrices for each type of relations.
    Args:
        confusion_matrix: Confusion matrix for a specific type of relation.

    Returns:
        Average accuracy.
    """
    # Get accuracy scores from confusion matrices for each category/class.

    # Handle cases where accuracy is None.
    def handle_accuracy(acc_dict: Dict[str, float]) -> Dict[str, float]:
        acc_dict = {k: v if v is not None else 1 for k, v in acc_dict.items()}
        return acc_dict

    # Calculate the average accuracy for each class.
    def calculate_average_accuracy(acc_dict: Dict[str, float]) -> float:
        values = list(acc_dict.values())
        return sum(values) / len(values) if len(values) > 0 else 0

    acc = confusion_matrix.ACC
    result = calculate_average_accuracy(handle_accuracy(acc))
    return result


def get_u_alpha(confusion_matrix: ConfusionMatrix) -> float:
    """Compute U-Alpha score based on the confusion matrices for each type of relations.
    Args:
        confusion_matrix: Confusion matrix for a specific type of relation.

    Returns:
        U-Alpha score (between 0 and 1).
    """
    # u-alpha calculation
    u_alpha = confusion_matrix.Alpha
    if u_alpha == "None":
        u_alpha = 1.0
    return u_alpha


def map_and_mean(func: Callable[[T], float], inputs: Sequence[T]) -> float:
    mapped_inputs = map(func, inputs)
    result = sum(mapped_inputs) / len(inputs)
    return result


def run_evaluation(predicted_nodeset_path: str, gold_nodeset_path: str):
    """Compute different scores to evaluate how similar given nodesets are to each other: Kappa,
    CASS, Accuracy, F1, U-Alpha.

    The scores are printed to the console.
    Args:
        predicted_nodeset_path: Path to the predicted nodeset.
        gold_nodeset_path: Path to the gold nodeset.
    """
    # Read the JSON data
    with open(predicted_nodeset_path) as f:
        predicted_data = json.load(f)
    with open(gold_nodeset_path) as f:
        gold_data = json.load(f)

    # Evaluate graphs
    confusion_matrix_dicts = matching.calculate_matching(predicted_data, gold_data)
    confusion_matrices = [ConfusionMatrix(matrix=d) for d in confusion_matrix_dicts]

    # Kappa
    kappa = map_and_mean(get_kappa, confusion_matrices)
    print("kappa", kappa)

    # CHANGE: This works only for AIF data structure, in DialAM we do not have "text" field.
    # Text Similarity and CASS
    # Text_similarity = eval.text_similarity(predicted_data, gold_data)
    # print("text similarity", Text_similarity)

    # CHANGE: Set to 1 because we have pre-segmented text in DialAM.
    # # cass = CASS_calculation(1, kappa)
    # CASS = CASS_calculation(Text_similarity, kappa)
    # print("CASS", cass)

    # F1
    f1 = map_and_mean(get_f1, confusion_matrices)
    print("F1", f1)
    # accuracy
    acc = map_and_mean(get_accuracy, confusion_matrices)
    print("Accuracy", acc)

    # U-Alpha
    u_alpha = map_and_mean(get_u_alpha, confusion_matrices)
    print("U-Alpha", u_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predicted_nodeset_path",
        type=str,
        help="path to the predicted nodeset (stored as a graph in JSON format)",
    )
    parser.add_argument(
        "gold_nodeset_path",
        type=str,
        help="path to the gold nodeset (stored as a graph in JSON format)",
    )
    args = vars(parser.parse_args())
    run_evaluation(**args)
