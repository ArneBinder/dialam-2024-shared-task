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

import calculation


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
    # Kappa
    (
        all_s_a_cm,
        prop_rels_comp_cm,
        loc_ya_rels_comp_cm,
        prop_ya_comp_cm,
        loc_ta_cm,
        prop_ya_cm,
    ) = calculation.matching_calculation(predicted_data, gold_data)

    kappa = calculation.kappa_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("kappa", kappa)

    # CHANGE: This works only for AIF data structure, in DialAM we do not have "text" field.
    # Text Similarity and CASS
    # Text_similarity = eval.text_similarity(predicted_data, gold_data)
    # print("text similarity", Text_similarity)

    # CHANGE: Set to 1 because we have pre-segmented text in DialAM.
    CASS = calculation.CASS_calculation(1, kappa)
    # CASS = CASS_calculation(Text_similarity, kappa)
    print("CASS", CASS)

    # F1
    F1 = calculation.F1_Macro_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("F1", F1)
    # accuracy
    Acc = calculation.accuracy_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("Accuracy", Acc)

    # U-Alpha
    U_Alpha = calculation.u_alpha_calculation(
        all_s_a_cm, prop_rels_comp_cm, loc_ya_rels_comp_cm, prop_ya_comp_cm, loc_ta_cm, prop_ya_cm
    )
    print("U-Alpha", U_Alpha)


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
