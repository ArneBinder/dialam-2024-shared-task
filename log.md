# Experimentation Log

This file is meant to log the development and experimentation process of this project.

## 2024-04-25

### Merged relations with BERT (lr=1e-4)

- training a single model for all relation types with bert-base-cased
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/mq64dj9d
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/ii14wa8i
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/yaxnlnkg
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-24_20-10-57`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_00-12-00`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_04-13-25`
  - aggregated metric values: macro/f1/val: 0.276, micro/f1/val: 0.649

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 8.21769e-05 | 8.73216e-05 | 8.79183e-05 |     3 | 8.85151e-05 | 8.42897e-05 | 7.70323e-05 | 6.31332e-06 |
| loss/train_epoch                                     | 8.21769e-05 | 8.73216e-05 | 8.79183e-05 |     3 | 8.85151e-05 | 8.42897e-05 | 7.70323e-05 | 6.31332e-06 |
| loss/train_step                                      | 3.58683e-05 | 4.31987e-05 |  7.7035e-05 |     3 | 0.000110871 | 6.08693e-05 | 2.85379e-05 | 4.39192e-05 |
| loss/val                                             |     2.64262 |     2.99905 |     3.00462 |     3 |      3.0102 |     2.76515 |      2.2862 |     0.41482 |
| metric/macro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                                  |    0.271045 |    0.276405 |    0.281551 |     3 |    0.286697 |    0.276263 |    0.265685 |   0.0105068 |
| metric/micro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                                  |    0.646191 |    0.652026 |    0.653512 |     3 |    0.654997 |    0.649127 |    0.640356 |  0.00773895 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.172246 |       0.175 |    0.190948 |     3 |    0.206897 |    0.183796 |    0.169492 |   0.0201943 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.329741 |    0.334328 |    0.345952 |     3 |    0.357576 |    0.339019 |    0.325153 |   0.0167124 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.306256 |    0.323988 |    0.327619 |     3 |     0.33125 |    0.314587 |    0.288525 |   0.0228613 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.508375 |    0.531343 |    0.536139 |     3 |    0.540936 |    0.519229 |    0.485407 |   0.0296805 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.973943 |     0.97619 |    0.976933 |     3 |    0.977675 |    0.975187 |    0.971695 |  0.00311343 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.251225 |    0.294118 |    0.341503 |     3 |    0.388889 |    0.297113 |    0.208333 |   0.0903151 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.231061 |        0.25 |       0.285 |     3 |        0.32 |    0.260707 |    0.212121 |   0.0547306 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.730783 |    0.731707 |    0.756551 |     3 |    0.781395 |    0.747654 |    0.729858 |   0.0292359 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.285368 |    0.291667 |    0.304924 |     3 |    0.318182 |    0.296306 |     0.27907 |   0.0199645 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.339772 |    0.358025 |    0.360068 |     3 |     0.36211 |    0.347218 |    0.321519 |   0.0223496 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |     0.34072 |    0.354167 |    0.392137 |     3 |    0.430108 |    0.370516 |    0.327273 |   0.0533312 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |   0.0951018 |     0.10687 |    0.111406 |     3 |    0.115942 |    0.102049 |   0.0833333 |   0.0168306 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.666011 |    0.666987 |    0.672905 |     3 |    0.678822 |    0.670281 |    0.665034 |  0.00746061 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |     0.37148 |    0.373494 |    0.382048 |     3 |    0.390602 |    0.377854 |    0.369466 |   0.0112225 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with BERT (lr=1e-3)

- training a single model for all relation types with bert-base-cased
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-3 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/8gugq7f7
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/zqv85k9r
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/tj42p8fd
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-24_20-09-46`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_00-09-30`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_04-09-44`
  - aggregated metric values: macro/f1/val: 0.282, micro/f1/val: 0.656

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 2.23664e-05 | 2.90249e-05 | 2.96288e-05 |     3 | 3.02327e-05 | 2.49885e-05 | 1.57079e-05 | 8.05991e-06 |
| loss/train_epoch                                     | 2.23664e-05 | 2.90249e-05 | 2.96288e-05 |     3 | 3.02327e-05 | 2.49885e-05 | 1.57079e-05 | 8.05991e-06 |
| loss/train_step                                      | 1.44875e-05 | 1.54095e-05 | 3.18415e-05 |     3 | 4.82735e-05 | 2.57495e-05 | 1.35655e-05 | 1.95282e-05 |
| loss/val                                             |     2.58121 |     3.03879 |     3.11821 |     3 |     3.19764 |     2.78669 |     2.12364 |    0.579687 |
| metric/macro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                                  |    0.279751 |    0.281877 |    0.283551 |     3 |    0.285224 |    0.281575 |    0.277624 |  0.00380915 |
| metric/micro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                                  |    0.653087 |    0.658392 |    0.660407 |     3 |    0.662423 |    0.656199 |    0.647783 |  0.00756241 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.186598 |    0.205128 |    0.210844 |     3 |    0.216561 |    0.196585 |    0.168067 |   0.0253503 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.359383 |    0.397554 |    0.400015 |     3 |    0.402477 |    0.373747 |    0.321212 |   0.0455635 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.304382 |    0.326797 |     0.35438 |     3 |    0.381963 |    0.330242 |    0.281967 |   0.0500868 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.532315 |     0.54314 |    0.559124 |     3 |    0.575107 |    0.546579 |     0.52149 |   0.0269736 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.975716 |    0.975724 |    0.976092 |     3 |     0.97646 |    0.975964 |    0.975708 | 0.000429695 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.253205 |     0.25641 |    0.298937 |     3 |    0.341463 |    0.282625 |        0.25 |   0.0510566 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.186957 |         0.2 |         0.2 |     3 |         0.2 |    0.191304 |    0.173913 |   0.0150613 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.743619 |    0.745455 |    0.753461 |     3 |    0.761468 |    0.749569 |    0.741784 |    0.010467 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.240418 |    0.285714 |    0.313589 |     3 |    0.341463 |      0.2741 |    0.195122 |   0.0738588 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.386015 |     0.39779 |    0.398656 |     3 |    0.399522 |    0.390517 |    0.374241 |   0.0141227 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.379675 |    0.409836 |    0.413251 |     3 |    0.416667 |    0.392006 |    0.349515 |   0.0369566 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.142312 |    0.163934 |     0.17465 |     3 |    0.185366 |    0.156663 |     0.12069 |   0.0329455 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.647481 |    0.650127 |     0.65711 |     3 |    0.664093 |    0.653018 |    0.644836 |  0.00994861 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.397668 |     0.39881 |    0.401973 |     3 |    0.405136 |    0.400157 |    0.396527 |  0.00445981 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (lr=1e-4)

- training a single model for all relation types with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/cguagkv8
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/7pbji8ve
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/5a7afy2m
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-24_20-25-36`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_01-20-28`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_06-16-26`
  - aggregated metric values: macro/f1/val: 0.375 micro/f1/val: 0.720

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000146487 | 0.000147932 | 0.000156053 |     3 | 0.000164173 | 0.000152382 | 0.000145042 | 1.03129e-05 |
| loss/train_epoch                                     | 0.000146487 | 0.000147932 | 0.000156053 |     3 | 0.000164173 | 0.000152382 | 0.000145042 | 1.03129e-05 |
| loss/train_step                                      | 2.13495e-05 | 2.45606e-05 | 7.26447e-05 |     3 | 0.000120729 | 5.44759e-05 | 1.81384e-05 | 5.74664e-05 |
| loss/val                                             |     2.05101 |     2.13155 |     2.15825 |     3 |     2.18496 |     2.09566 |     1.97048 |    0.111652 |
| metric/macro/f1/train                                |    0.959927 |    0.959943 |    0.979955 |     3 |    0.999968 |    0.973274 |    0.959912 |   0.0231176 |
| metric/macro/f1/val                                  |    0.362106 |    0.365462 |    0.382362 |     3 |    0.399262 |    0.374491 |     0.35875 |   0.0217126 |
| metric/micro/f1/train                                |    0.999942 |    0.999953 |    0.999965 |     3 |    0.999977 |    0.999953 |     0.99993 | 2.32756e-05 |
| metric/micro/f1/val                                  |    0.715892 |    0.719711 |    0.724485 |     3 |    0.729259 |    0.720348 |    0.712073 |  0.00861092 |
| metric/s_nodes:Default Conflict/f1/train             |    0.999332 |    0.999332 |    0.999666 |     3 |           1 |    0.999555 |    0.999332 | 0.000385664 |
| metric/s_nodes:Default Conflict/f1/val               |    0.347645 |    0.387597 |    0.392146 |     3 |    0.396694 |    0.363994 |    0.307692 |   0.0489708 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.479188 |    0.490683 |    0.509493 |     3 |    0.528302 |    0.495559 |    0.467692 |   0.0305976 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.461344 |    0.467066 |    0.468998 |     3 |     0.47093 |    0.464539 |    0.455621 |   0.0079611 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.637472 |    0.643059 |    0.664465 |     3 |    0.685871 |    0.653605 |    0.631884 |   0.0284966 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.980718 |    0.981308 |    0.981595 |     3 |    0.981882 |    0.981106 |    0.980129 | 0.000893966 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.355145 |    0.385965 |    0.437427 |     3 |    0.488889 |    0.399726 |    0.324324 |   0.0831409 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.482883 |     0.52459 |    0.536489 |     3 |    0.548387 |    0.504718 |    0.441176 |   0.0563002 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.807343 |    0.807512 |    0.810006 |     3 |      0.8125 |    0.809062 |    0.807175 |  0.00298198 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.343035 |    0.378378 |    0.411411 |     3 |    0.444444 |    0.376838 |    0.307692 |   0.0683891 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |         0.5 |     3 |           1 |    0.333333 |           0 |     0.57735 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.450073 |    0.471769 |    0.474241 |     3 |    0.476712 |    0.458953 |    0.428377 |   0.0265948 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |     0.44202 |        0.48 |    0.499259 |     3 |    0.518519 |     0.46752 |     0.40404 |   0.0582506 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998934 |    0.999289 |     0.99929 |     3 |     0.99929 |    0.999053 |     0.99858 | 0.000410046 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.278846 |    0.307692 |    0.308318 |     3 |    0.308943 |    0.288878 |        0.25 |   0.0336756 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999919 |    0.999946 |    0.999946 |     3 |    0.999946 |    0.999928 |    0.999891 | 3.13156e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.727732 |    0.732618 |    0.733117 |     3 |    0.733616 |    0.729694 |    0.722846 |  0.00595071 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.524533 |    0.524823 |    0.543065 |     3 |    0.561308 |    0.536791 |    0.524242 |   0.0212342 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (lr=1e-3)

- training a single model for all relation types with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-3 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/e2o36sol
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/hwnnn3hd
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/7mfgovmk
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-24_20-41-39`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_01-37-11`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_06-33-07`
  - aggregated metric values: macro/f1/val: 0.374, micro/f1/val: 0.727

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           |  0.00012635 | 0.000146321 | 0.000198245 |     3 | 0.000250168 | 0.000167623 | 0.000106378 | 7.42239e-05 |
| loss/train_epoch                                     |  0.00012635 | 0.000146321 | 0.000198245 |     3 | 0.000250168 | 0.000167623 | 0.000106378 | 7.42239e-05 |
| loss/train_step                                      | 4.56794e-06 | 6.89477e-06 | 3.42543e-05 |     3 | 6.16139e-05 | 2.35833e-05 | 2.24111e-06 | 3.30176e-05 |
| loss/val                                             |     1.87567 |     2.04329 |     2.58947 |     3 |     3.13565 |     2.29566 |     1.70806 |    0.746507 |
| metric/macro/f1/train                                |    0.959927 |    0.959943 |    0.979944 |     3 |    0.999946 |    0.973266 |    0.959911 |   0.0231053 |
| metric/macro/f1/val                                  |    0.370755 |    0.378717 |     0.37972 |     3 |    0.380723 |    0.374078 |    0.362794 |  0.00982386 |
| metric/micro/f1/train                                |    0.999942 |    0.999953 |    0.999953 |     3 |    0.999953 |    0.999946 |     0.99993 | 1.34554e-05 |
| metric/micro/f1/val                                  |    0.722788 |    0.723106 |    0.729472 |     3 |    0.735837 |    0.727138 |     0.72247 |  0.00754057 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998998 |    0.999332 |    0.999666 |     3 |           1 |    0.999332 |    0.998665 | 0.000667542 |
| metric/s_nodes:Default Conflict/f1/val               |    0.357475 |    0.390625 |    0.447164 |     3 |    0.503704 |    0.406218 |    0.324324 |   0.0907006 |
| metric/s_nodes:Default Inference-rev/f1/train        |    0.999861 |           1 |           1 |     3 |           1 |    0.999907 |    0.999722 | 0.000160226 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.562561 |    0.569231 |    0.573223 |     3 |    0.577215 |    0.567446 |    0.555891 |   0.0107735 |
| metric/s_nodes:Default Inference/f1/train            |    0.999874 |           1 |           1 |     3 |           1 |    0.999916 |    0.999748 | 0.000145669 |
| metric/s_nodes:Default Inference/f1/val              |    0.433376 |    0.443077 |    0.482527 |     3 |    0.521978 |     0.46291 |    0.423676 |   0.0520658 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999932 |           1 |           1 |     3 |           1 |    0.999955 |    0.999864 | 7.84955e-05 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.638361 |     0.65043 |    0.676669 |     3 |    0.702908 |    0.659877 |    0.626292 |   0.0391715 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.981877 |    0.982436 |    0.982595 |     3 |    0.982754 |    0.982169 |    0.981319 |  0.00075325 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.419958 |    0.461538 |    0.472149 |     3 |    0.482759 |    0.440892 |    0.378378 |   0.0551681 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |       0.125 |     3 |        0.25 |   0.0833333 |           0 |    0.144338 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.510015 |    0.545455 |    0.554417 |     3 |     0.56338 |    0.527804 |    0.474576 |   0.0469596 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.811927 |    0.825688 |    0.836425 |     3 |    0.847162 |    0.823672 |    0.798165 |   0.0245604 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.305556 |    0.311111 |    0.333333 |     3 |    0.355556 |    0.322222 |         0.3 |   0.0293972 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.467624 |    0.484615 |    0.489832 |     3 |     0.49505 |    0.476766 |    0.450633 |   0.0232254 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.494855 |    0.525424 |    0.546045 |     3 |    0.566667 |    0.518792 |    0.464286 |   0.0515116 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.999289 |    0.999289 |    0.999289 |     3 |    0.999289 |    0.999289 |    0.999289 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.308697 |    0.318182 |    0.334416 |     3 |    0.350649 |    0.322681 |    0.299213 |   0.0260119 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999946 |    0.999946 |    0.999946 |     3 |    0.999946 |    0.999946 |    0.999946 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.718144 |        0.72 |    0.720913 |     3 |    0.721826 |    0.719371 |    0.716288 |  0.00282195 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.531384 |     0.54242 |     0.55205 |     3 |     0.56168 |    0.541483 |    0.520349 |   0.0206814 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (lr=2e-5)

- training a single model for all relation types with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=2e-5 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/0v4695xz
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/e8p0rxtg
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/2ko52o83
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-24_20-41-51`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_01-35-49`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_06-29-52`
  - aggregated metric values: macro/f1/val: 0.360, micro/f1/val: 0.715

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000185441 |  0.00019368 | 0.000363953 |     3 | 0.000534226 | 0.000301703 | 0.000177202 | 0.000201539 |
| loss/train_epoch                                     | 0.000185441 |  0.00019368 | 0.000363953 |     3 | 0.000534226 | 0.000301703 | 0.000177202 | 0.000201539 |
| loss/train_step                                      | 5.38459e-05 | 8.22632e-05 |  0.00029619 |     3 | 0.000510117 | 0.000205936 | 2.54285e-05 | 0.000264956 |
| loss/val                                             |     1.69444 |     1.86108 |     2.02227 |     3 |     2.18346 |     1.85744 |     1.52779 |    0.327851 |
| metric/macro/f1/train                                |    0.959898 |    0.959912 |    0.959927 |     3 |    0.959943 |    0.959913 |    0.959883 | 2.96293e-05 |
| metric/macro/f1/val                                  |    0.355124 |    0.356498 |    0.362717 |     3 |    0.368937 |    0.359728 |    0.353751 |   0.0080918 |
| metric/micro/f1/train                                |    0.999918 |     0.99993 |    0.999942 |     3 |    0.999953 |     0.99993 |    0.999907 | 2.33054e-05 |
| metric/micro/f1/val                                  |    0.713558 |    0.717165 |     0.71759 |     3 |    0.718014 |    0.715044 |    0.709951 |  0.00443042 |
| metric/s_nodes:Default Conflict/f1/train             |    0.999332 |    0.999332 |    0.999332 |     3 |    0.999332 |    0.999332 |    0.999332 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.360606 |    0.393939 |    0.404662 |     3 |    0.415385 |    0.378866 |    0.327273 |   0.0459493 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.448008 |    0.463415 |    0.475031 |     3 |    0.486647 |    0.460888 |    0.432602 |    0.027111 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.445123 |    0.451807 |    0.454717 |     3 |    0.457627 |    0.449291 |    0.438438 |   0.0098387 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.628322 |    0.640559 |    0.643125 |     3 |    0.645691 |    0.634112 |    0.616085 |   0.0158211 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.980003 |    0.980163 |    0.980874 |     3 |    0.981584 |     0.98053 |    0.979842 | 0.000927097 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.352941 |    0.352941 |    0.382353 |     3 |    0.411765 |    0.372549 |    0.352941 |   0.0339618 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |    0.142857 |     3 |    0.285714 |   0.0952381 |           0 |    0.164957 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.495902 |         0.5 |     0.55137 |     3 |     0.60274 |    0.531514 |    0.491803 |    0.061819 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.794231 |         0.8 |    0.814414 |     3 |    0.828829 |    0.805763 |    0.788462 |   0.0207916 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.333333 |    0.333333 |    0.386179 |     3 |    0.439024 |    0.368564 |    0.333333 |   0.0610208 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.446458 |    0.460916 |    0.468721 |     3 |    0.476526 |    0.456481 |       0.432 |   0.0225919 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |    0.999441 |           1 |           1 |     3 |           1 |    0.999628 |    0.998883 | 0.000645067 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.433634 |    0.452174 |    0.466828 |     3 |    0.481481 |    0.449583 |    0.415094 |   0.0332693 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998934 |    0.999289 |    0.999289 |     3 |    0.999289 |    0.999053 |     0.99858 | 0.000409753 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.298696 |    0.305085 |    0.315663 |     3 |    0.326241 |    0.307878 |    0.292308 |   0.0171383 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999891 |    0.999891 |    0.999919 |     3 |    0.999946 |     0.99991 |    0.999891 | 3.13156e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.721312 |    0.725118 |    0.729553 |     3 |    0.733988 |    0.725537 |    0.717506 |  0.00824892 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.999845 |           1 |           1 |     3 |           1 |    0.999897 |    0.999691 | 0.000178465 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.533399 |    0.536252 |    0.540983 |     3 |    0.545714 |    0.537504 |    0.530547 |    0.007661 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>
