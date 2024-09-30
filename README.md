<div align="center">

# DialAM-2024 Shared Task

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-PyTorch--IE--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](https://img.shields.io/badge/paper-2024.argmining.1.9-B31B1B.svg)](https://aclanthology.org/2024.argmining-1.9)
[![Conference](https://img.shields.io/badge/ArgMining@ACL-2024-4b44ce.svg)](https://aclanthology.org/volumes/2024.argmining-1)

</div>

## 📌 Description

This repository contains the code for our submission to the DialAM-2024 Shared Task as described in the
paper [DFKI-MLST at DialAM-2024 Shared Task: System Description (Binder et al., ArgMining 2024)](https://aclanthology.org/2024.argmining-1.9/).
The task is part of the ArgMining 2024 workshop and focuses on the identification of argumentative relations in dialogues.
See the [official website](https://dialam.arg.tech/) for more information.

### 📃 Abstract

We present the dfki-mlst submission for the DialAM shared task on identification of argumentative and illocutionary
relations in dialogue. Our model achieves the best results in the global setting: 48.25 F1 at the focused level when
looking only at the related arguments/locutions and 67.05 F1 at the general level when evaluating the complete
argument maps. We describe our implementation of the data pre-processing pipeline, relation encoding and
classification, evaluating 11 different base models and performing experiments with, e.g., node text combination
and data augmentation. Our source code is publicly available.

### ✨ How to Reproduce the Results from Our Paper

1. Set up the environment as described in the [Environment Setup](#environment-setup) section.
2. Train the model with the configuration from the paper: TODO
3. Evaluate the model on the test set: TODO

## 🚀 Quickstart

### Environment Setup

```bash
# clone project
git clone https://github.com/ArneBinder/dialam-2024-shared-task.git
cd dialam-2024-shared-task

# [OPTIONAL] create conda environment
conda create -n dialam-2024-shared-task python=3.9
conda activate dialam-2024-shared-task

# install PyTorch according to instructions
# https://pytorch.org/get-started/

# install remaining requirements
pip install -r requirements.txt

# [OPTIONAL] symlink log directories and the default model directory to
# "$HOME/experiments/dialam-2024-shared-task" since they can grow a lot
bash setup_symlinks.sh $HOME/experiments/dialam-2024-shared-task

# [OPTIONAL] set any environment variables by creating an .env file
# 1. copy the provided example file:
cp .env.example .env
# 2. edit the .env file for your needs!
```

### Model Training

**Have a look into the [train.yaml](configs/train.yaml) config to see all available options.**

Train model with default configuration

```bash
# train on CPU
python src/train.py

# train on GPU
python src/train.py trainer=gpu
```

Execute a fast development run (train for two steps)

```bash
python src/train.py +trainer.fast_dev_run=true
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=conll2003
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

Start multiple runs at once (multirun):

```bash
python src/train.py seed=42,43 --multirun
```

Notes:

- this will execute two experiments (one after the other), one for each seed
- the results will be aggregated and stored in `logs/multirun/`, see the last logging output for the exact path

### Model evaluation

This will evaluate the model on the test set of the chosen dataset using the *metrics implemented within the model*.
See [config/dataset/](configs/dataset/) for available datasets.

**Have a look into the [evaluate.yaml](configs/evaluate.yaml) config to see all available options.**

```bash
python src/evaluate.py dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `trainer=gpu` to run on GPU

### Inference

This will run inference on the given dataset and split. See [config/dataset/](configs/dataset/) for available datasets.
The result documents including the predicted annotations will be stored in the `predictions/` directory (exact
location will be printed to the console).

**Have a look into the [predict.yaml](configs/predict.yaml) config to see all available options.**

```bash
python src/predict.py dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `+pipeline.device=0` to run the inference on GPU 0

### Evaluate Serialized Documents

This will evaluate serialized documents including predicted annotations (see [Inference](#inference)) using a
*document metric*. See [config/metric/](configs/metric/) for available metrics.

**Have a look into the [evaluate_documents.yaml](configs/evaluate_documents.yaml) config to see all available options**

```bash
python src/evaluate_documents.py metric=f1 metric.layer=entities +dataset.data_dir=PATH/TO/DIR/WITH/SPLITS
```

Note: By default, this utilizes the dataset provided by the
[from_serialized_documents](configs/dataset/from_serialized_documents.yaml) configuration. This configuration is
designed to facilitate the loading of serialized documents, as generated during the [Inference](#inference) step. It
requires to set the parameter `data_dir`. If you want to use a different dataset,
you can override the `dataset` parameter as usual with any existing dataset config, e.g `dataset=conll2003`. But
calculating the F1 score on the bare `conll2003` dataset does not make much sense, because it does not contain any
predictions. However, it could be used with statistical metrics such as
[count_text_tokens](configs/metric/count_text_tokens.yaml) or
[count_entity_labels](configs/metric/count_entity_labels.yaml).

## Development

```bash
# run pre-commit: code formatting, code analysis, static type checking, and more (see .pre-commit-config.yaml)
pre-commit run -a

# run tests
pytest -k "not slow" --cov --cov-report term-missing
```

## 📃 Citation

```bibtex
@inproceedings{binder-etal-2024-dfki,
    title = "{DFKI}-{MLST} at {D}ial{AM}-2024 Shared Task: System Description",
    author = "Binder, Arne  and
      Anikina, Tatiana  and
      Hennig, Leonhard  and
      Ostermann, Simon",
    editor = "Ajjour, Yamen  and
      Bar-Haim, Roy  and
      El Baff, Roxanne  and
      Liu, Zhexiong  and
      Skitalinskaya, Gabriella",
    booktitle = "Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.argmining-1.9",
    doi = "10.18653/v1/2024.argmining-1.9",
    pages = "93--102",
    abstract = "This paper presents the dfki-mlst submission for the DialAM shared task (Ruiz-Dolz et al., 2024) on identification of argumentative and illocutionary relations in dialogue. Our model achieves best results in the global setting: 48.25 F1 at the focused level when looking only at the related arguments/locutions and 67.05 F1 at the general level when evaluating the complete argument maps. We describe our implementation of the data pre-processing, relation encoding and classification, evaluating 11 different base models and performing experiments with, e.g., node text combination and data augmentation. Our source code is publicly available.",
}
```
