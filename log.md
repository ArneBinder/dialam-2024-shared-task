# Experimentation Log

This file is meant to log the development and experimentation process of this project.

## 2024-04-25

### Merged relations with BERT (task_learning_rate=1e-4, learning_rate=1e-5)

- training a single model for all relation types with bert-base-uncased
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

### Merged relations with BERT (task_learning_rate=1e-3, learning_rate=1e-5)

- training a single model for all relation types with bert-base-uncased
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

### Merged relations with BERT (task_learning_rate=2e-5, learning_rate=1e-5)

- training a single model for all relation types with bert-base-uncased
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=2e-5 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/moygbnxz
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/056tuuti
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/airf1mip
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_11-57-24`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_15-49-52`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_19-42-23`
  - aggregated metric values: macro/f1/val: 0.271, micro/f1/val: 0.649

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000167621 | 0.000202459 | 0.000207885 |     3 | 0.000213311 | 0.000182851 | 0.000132783 | 4.36981e-05 |
| loss/train_epoch                                     | 0.000167621 | 0.000202459 | 0.000207885 |     3 | 0.000213311 | 0.000182851 | 0.000132783 | 4.36981e-05 |
| loss/train_step                                      | 0.000135734 | 0.000161774 | 0.000165532 |     3 | 0.000169291 |  0.00014692 | 0.000109695 |  3.2456e-05 |
| loss/val                                             |     1.75869 |     2.26289 |     2.43178 |     3 |     2.60066 |     2.03935 |     1.25449 |    0.700377 |
| metric/macro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                                  |    0.265468 |    0.272741 |    0.277855 |     3 |     0.28297 |    0.271302 |    0.258195 |   0.0124501 |
| metric/micro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                                  |    0.645449 |    0.654997 |    0.655421 |     3 |    0.655846 |    0.648914 |    0.635901 |   0.0112781 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.146771 |    0.150685 |    0.156424 |     3 |    0.162162 |    0.151901 |    0.142857 |  0.00970982 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.339651 |    0.349693 |    0.354178 |     3 |    0.358663 |    0.345988 |    0.329609 |    0.014877 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.279739 |    0.291022 |    0.324282 |     3 |    0.357542 |    0.305673 |    0.268456 |   0.0463148 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.504711 |    0.522936 |    0.526864 |     3 |    0.530792 |    0.513405 |    0.486486 |   0.0236404 |
| metric/s_nodes:NONE/f1/train                         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:NONE/f1/val                           |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.974865 |    0.975907 |    0.976049 |     3 |     0.97619 |    0.975307 |    0.973822 |  0.00129341 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.258333 |    0.266667 |    0.345455 |     3 |    0.424242 |    0.313636 |        0.25 |   0.0961495 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.173611 |    0.222222 |    0.245727 |     3 |    0.269231 |    0.205484 |       0.125 |   0.0735578 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.737148 |    0.777778 |     0.78795 |     3 |    0.798122 |    0.757472 |    0.696517 |   0.0537597 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.218644 |    0.237288 |    0.296422 |     3 |    0.355556 |    0.264281 |         0.2 |   0.0812149 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.350131 |    0.360494 |    0.361968 |     3 |    0.363443 |    0.354568 |    0.339768 |   0.0129018 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.332923 |    0.337079 |    0.360459 |     3 |    0.383838 |    0.349895 |    0.328767 |   0.0296884 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.116384 |    0.120879 |    0.135066 |     3 |    0.149254 |     0.12734 |    0.111888 |   0.0195028 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.663848 |    0.676306 |    0.677802 |     3 |    0.679299 |    0.668998 |     0.65139 |   0.0153227 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.338988 |    0.376506 |    0.394349 |     3 |    0.412192 |    0.363389 |    0.301471 |   0.0565139 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=1e-5)

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

### Merged relations with RoBERTa (task_learning_rate=1e-3, learning_rate=1e-5)

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

### Merged relations with RoBERTa (task_learning_rate=2e-5, learning_rate=1e-5)

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

## 2024-04-26

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=1e-4)

- training a single model for all relation types with xlm-roberta-large, experimenting with the base-model learning rate
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      model.learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/cj95qf29
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/a9p9d2uv
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/7afiqtki
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_15-03-03`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_19-54-33`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-26_00-47-25`
  - aggregated metric values: macro/f1/val: 0.160, micro/f1/val: 0.591

<details>

|                                                      |       25% |       50% |       75% | count |       max |      mean |       min |         std |
| :--------------------------------------------------- | --------: | --------: | --------: | ----: | --------: | --------: | --------: | ----------: |
| loss/train                                           |   2.09113 |   2.09115 |   2.09117 |     3 |   2.09119 |   2.09115 |   2.09111 | 3.87039e-05 |
| loss/train_epoch                                     |   2.09113 |   2.09115 |   2.09117 |     3 |   2.09119 |   2.09115 |   2.09111 | 3.87039e-05 |
| loss/train_step                                      |   2.01184 |    2.2699 |   2.28398 |     3 |   2.29806 |   2.10725 |   1.75379 |     0.30643 |
| loss/val                                             |   1.25156 |   1.26906 |   1.27378 |     3 |   1.27851 |   1.26055 |   1.23407 |   0.0234084 |
| metric/macro/f1/train                                | 0.0223256 | 0.0223256 | 0.0223256 |     3 | 0.0223256 | 0.0223256 | 0.0223256 |           0 |
| metric/macro/f1/val                                  |  0.155718 |  0.162623 |   0.16588 |     3 |  0.169136 |  0.160191 |  0.148813 |   0.0103778 |
| metric/micro/f1/train                                |  0.365946 |  0.365946 |  0.365946 |     3 |  0.365946 |  0.365946 |  0.365946 |           0 |
| metric/micro/f1/val                                  |  0.582325 |  0.608954 |  0.609272 |     3 |   0.60959 |  0.591414 |  0.555697 |   0.0309333 |
| metric/s_nodes:Default Conflict/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Conflict/f1/val               | 0.0117647 | 0.0235294 | 0.0433437 |     3 | 0.0631579 | 0.0288958 |         0 |   0.0319191 |
| metric/s_nodes:Default Inference-rev/f1/train        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |         0 |         0 |  0.133874 |     3 |  0.267748 | 0.0892495 |         0 |    0.154585 |
| metric/s_nodes:Default Inference/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference/f1/val              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Rephrase/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               | 0.0899148 |  0.165441 |  0.264302 |     3 |  0.363162 |  0.180997 | 0.0143885 |    0.174907 |
| metric/s_nodes:NONE/f1/train                         |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:NONE/f1/val                           |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |  0.535814 |  0.535814 |  0.535814 |     3 |  0.535814 |  0.535814 |  0.535814 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |   0.97296 |  0.975197 |  0.975382 |     3 |  0.975567 |  0.973829 |  0.970723 |  0.00269608 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |         0 |         0 | 0.0833333 |     3 |  0.166667 | 0.0555556 |         0 |    0.096225 |
| metric/ya_i2l_nodes:Challenging/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |  0.763775 |  0.771186 |  0.772047 |     3 |  0.772908 |  0.766819 |  0.756364 |  0.00909587 |
| metric/ya_i2l_nodes:Restating/f1/train               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    | 0.0454545 | 0.0909091 |  0.125455 |     3 |      0.16 | 0.0836364 |         0 |   0.0802475 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |   0.15442 |  0.156425 |  0.212015 |     3 |  0.267606 |  0.192149 |  0.152416 |   0.0653782 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       | 0.0447761 | 0.0895522 |   0.21719 |     3 |  0.344828 |  0.144793 |         0 |    0.178928 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |         0 |         0 |  0.106509 |     3 |  0.213018 | 0.0710059 |         0 |    0.122986 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |  0.558952 |  0.568065 |  0.568782 |     3 |    0.5695 |  0.562468 |  0.549839 |   0.0109603 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |  0.142235 |   0.28447 |  0.321914 |     3 |  0.359358 |   0.21461 |         0 |    0.189592 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=2e-4)

- training a single model for all relation types with xlm-roberta-large, experimenting with the base-model learning rate
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      model.learning_rate=2e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/giydsjkw
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/vzyrutym
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/ruqsz94o
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_15-02-46`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_19-55-11`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-26_00-48-32`
  - aggregated metric values: macro/f1/val: 0.042, micro/f1/val: 0.433

<details>

|                                                      |       25% |       50% |       75% | count |       max |      mean |       min |         std |
| :--------------------------------------------------- | --------: | --------: | --------: | ----: | --------: | --------: | --------: | ----------: |
| loss/train                                           |   2.09113 |   2.09114 |   2.09116 |     3 |   2.09117 |   2.09114 |   2.09111 | 3.05176e-05 |
| loss/train_epoch                                     |   2.09113 |   2.09114 |   2.09116 |     3 |   2.09117 |   2.09114 |   2.09111 | 3.05176e-05 |
| loss/train_step                                      |   2.01171 |   2.26988 |   2.28389 |     3 |   2.29789 |   2.10711 |   1.75354 |    0.306514 |
| loss/val                                             |   1.79043 |   2.11249 |   2.11414 |     3 |   2.11578 |   1.89888 |   1.46837 |    0.372836 |
| metric/macro/f1/train                                | 0.0223256 | 0.0223256 | 0.0223256 |     3 | 0.0223256 | 0.0223256 | 0.0223256 |           0 |
| metric/macro/f1/val                                  | 0.0265545 | 0.0265545 | 0.0501591 |     3 | 0.0737637 | 0.0422909 | 0.0265545 |   0.0272563 |
| metric/micro/f1/train                                |  0.365946 |  0.365946 |  0.365946 |     3 |  0.365946 |  0.365946 |  0.365946 |           0 |
| metric/micro/f1/val                                  |  0.361553 |  0.361553 |  0.469022 |     3 |  0.576491 |  0.433199 |  0.361553 |    0.124094 |
| metric/s_nodes:Default Conflict/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference-rev/f1/train        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Inference/f1/val              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Rephrase/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:NONE/f1/train                         |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/s_nodes:NONE/f1/val                           |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |  0.535814 |  0.535814 |  0.535814 |     3 |  0.535814 |  0.535814 |  0.535814 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |  0.531089 |  0.531089 |  0.738484 |     3 |  0.945878 |  0.669352 |  0.531089 |    0.239479 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/train             |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Restating/f1/train               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |         0 |         0 |  0.264698 |     3 |  0.529396 |  0.176465 |         0 |    0.305647 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |         0 |         0 |         0 |     3 |         0 |         0 |         0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=2e-5)

- training a single model for all relation types with xlm-roberta-large, experimenting with the base-model learning rate
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      model.learning_rate=2e-5 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/l97qx2cy
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/nxzu2pkw
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/6pufb6nn
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_15-02-30`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-25_19-56-06`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-26_00-54-35`
  - aggregated metric values: macro/f1/val: 0.223, micro/f1/val: 0.638

<details>

|                                                      |       25% |       50% |      75% | count |      max |      mean |         min |        std |
| :--------------------------------------------------- | --------: | --------: | -------: | ----: | -------: | --------: | ----------: | ---------: |
| loss/train                                           |   1.04563 |   2.09111 |  2.09112 |     3 |  2.09114 |   1.39414 | 0.000159486 |    1.20722 |
| loss/train_epoch                                     |   1.04563 |   2.09111 |  2.09112 |     3 |  2.09114 |   1.39414 | 0.000159486 |    1.20722 |
| loss/train_step                                      |   1.13499 |   2.26995 |  2.28369 |     3 |  2.29742 |   1.52247 | 1.92493e-05 |    1.31855 |
| loss/val                                             |   1.29621 |   1.30486 |  1.68418 |     3 |  2.06349 |   1.55197 |     1.28757 |   0.443073 |
| metric/macro/f1/train                                | 0.0223256 | 0.0223256 | 0.491134 |     3 | 0.959943 |  0.334865 |   0.0223256 |   0.541334 |
| metric/macro/f1/val                                  |  0.145062 |  0.155403 | 0.266902 |     3 | 0.378401 |  0.222842 |    0.134722 |   0.135115 |
| metric/micro/f1/train                                |  0.365946 |  0.365946 |  0.68295 |     3 | 0.999953 |  0.577282 |    0.365946 |   0.366044 |
| metric/micro/f1/val                                  |   0.59198 |   0.60959 | 0.669531 |     3 | 0.729472 |   0.63781 |    0.574369 |  0.0813111 |
| metric/s_nodes:Default Conflict/f1/train             |         0 |         0 | 0.499666 |     3 | 0.999332 |  0.333111 |           0 |   0.576965 |
| metric/s_nodes:Default Conflict/f1/val               |         0 |         0 |  0.13913 |     3 | 0.278261 | 0.0927536 |           0 |   0.160654 |
| metric/s_nodes:Default Inference-rev/f1/train        |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/s_nodes:Default Inference-rev/f1/val          | 0.0143541 | 0.0287081 | 0.279928 |     3 | 0.531148 |  0.186619 |           0 |   0.298716 |
| metric/s_nodes:Default Inference/f1/train            |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/s_nodes:Default Inference/f1/val              |         0 |         0 | 0.223926 |     3 | 0.447853 |  0.149284 |           0 |   0.258568 |
| metric/s_nodes:Default Rephrase/f1/train             |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/s_nodes:Default Rephrase/f1/val               |  0.222023 |  0.331075 | 0.507783 |     3 | 0.684492 |  0.376179 |    0.112971 |   0.288418 |
| metric/s_nodes:NONE/f1/train                         |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/s_nodes:NONE/f1/val                           |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |  0.535814 |  0.535814 | 0.767907 |     3 |        1 |  0.690542 |    0.535814 |   0.267998 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |  0.970159 |   0.97252 | 0.977524 |     3 | 0.982528 |  0.974282 |    0.967798 | 0.00752125 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |         0 |         0 | 0.222222 |     3 | 0.444444 |  0.148148 |           0 |     0.2566 |
| metric/ya_i2l_nodes:Challenging/f1/train             |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Challenging/f1/val               |         0 |         0 |      0.2 |     3 |      0.4 |  0.133333 |           0 |    0.23094 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:NONE/f1/val                      |         0 |         0 | 0.269841 |     3 | 0.539683 |  0.179894 |           0 |   0.311586 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |  0.787376 |  0.804878 | 0.819479 |     3 | 0.834081 |  0.802944 |    0.769874 |  0.0321468 |
| metric/ya_i2l_nodes:Restating/f1/train               |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Restating/f1/val                 |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |         0 |         0 | 0.129032 |     3 | 0.258065 | 0.0860215 |           0 |   0.148994 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |  0.140688 |  0.271715 | 0.386192 |     3 | 0.500669 |  0.260682 |  0.00966184 |    0.24569 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |  0.104167 |  0.208333 | 0.358884 |     3 | 0.509434 |  0.239256 |           0 |   0.256121 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |         0 |         0 | 0.499645 |     3 | 0.999289 |  0.333096 |           0 |    0.57694 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |         0 |         0 |  0.12782 |     3 | 0.255639 |  0.085213 |           0 |   0.147593 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |         0 |         0 | 0.499973 |     3 | 0.999946 |  0.333315 |           0 |   0.577319 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |  0.594967 |  0.603563 | 0.667166 |     3 | 0.730769 |  0.640235 |    0.586371 |  0.0788752 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Restating/f1/val                |  0.212571 |  0.289474 | 0.419415 |     3 | 0.549356 |  0.324832 |    0.135667 |   0.209099 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |         0 |         0 |      0.5 |     3 |        1 |  0.333333 |           0 |    0.57735 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |         0 |         0 |        0 |     3 |        0 |         0 |           0 |          0 |

</details>

### S-relations with RoBERTa (lr=1e-4)

- training a single model for s-relation types with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_s \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=roberta-single-relations \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/roberta-single-relations-training/runs/kcufnalb
    - seed2: https://wandb.ai/tanikina/roberta-single-relations-training/runs/bi8nnrw2
    - seed3: https://wandb.ai/tanikina/roberta-single-relations-training/runs/a1d7gnh1
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_09-20-12`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_11-00-54`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_12-42-38`
  - aggregated metric values: macro/f1/val: 0.393, micro/f1/val: 0.469

<details>

|                                       |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :------------------------------------ | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                            |  0.00018739 | 0.000196616 |  0.00020003 |     3 | 0.000203443 | 0.000192741 | 0.000178163 | 1.30781e-05 |
| loss/train_epoch                      |  0.00018739 | 0.000196616 |  0.00020003 |     3 | 0.000203443 | 0.000192741 | 0.000178163 | 1.30781e-05 |
| loss/train_step                       | 8.09023e-05 | 0.000111095 | 0.000136851 |     3 | 0.000162606 | 0.000108137 | 5.07096e-05 | 5.60069e-05 |
| loss/val                              |     2.37783 |     2.46812 |     2.48552 |     3 |     2.50292 |     2.41953 |     2.28755 |    0.115618 |
| metric/Default Conflict/f1/train      |    0.999666 |           1 |           1 |     3 |           1 |    0.999777 |    0.999332 | 0.000385664 |
| metric/Default Conflict/f1/val        |    0.341492 |     0.34965 |    0.351296 |     3 |    0.352941 |    0.345308 |    0.333333 |   0.0105003 |
| metric/Default Inference-rev/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Default Inference-rev/f1/val   |    0.483294 |    0.491018 |      0.5011 |     3 |    0.511182 |     0.49259 |     0.47557 |    0.017858 |
| metric/Default Inference/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Default Inference/f1/val       |    0.485525 |    0.485876 |    0.488552 |     3 |    0.491228 |    0.487426 |    0.485175 |  0.00331099 |
| metric/Default Rephrase/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Default Rephrase/f1/val        |    0.634252 |    0.652893 |    0.653218 |     3 |    0.653543 |    0.640682 |    0.615611 |   0.0217147 |
| metric/NONE/f1/train                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/NONE/f1/val                    |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/macro/f1/train                 |    0.899933 |           1 |           1 |     3 |           1 |    0.933289 |    0.799866 |    0.115547 |
| metric/macro/f1/val                   |    0.391479 |    0.395887 |    0.396267 |     3 |    0.396647 |    0.393201 |     0.38707 |  0.00532343 |
| metric/micro/f1/train                 |    0.999939 |           1 |           1 |     3 |           1 |    0.999959 |    0.999878 | 7.03397e-05 |
| metric/micro/f1/val                   |    0.460133 |     0.47619 |    0.481174 |     3 |    0.486157 |    0.468808 |    0.444075 |   0.0219909 |

</details>

### YA S2TA relations with RoBERTa (lr=1e-4)

- training a single model for YA-relation types (between S and TA nodes) with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_ya_s2ta \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=roberta-single-relations \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/roberta-single-relations-training/runs/yu2egcp2
    - seed2: https://wandb.ai/tanikina/roberta-single-relations-training/runs/mxvmb5oj
    - seed3: https://wandb.ai/tanikina/roberta-single-relations-training/runs/08tov8mg
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_09-18-57`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_11-05-15`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_12-51-45`
  - aggregated metric values: macro/f1/val: 0.266, micro/f1/val: 0.484

<details>

|                                        |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                             | 0.000227897 | 0.000234378 | 0.000239494 |     3 | 0.000244609 | 0.000233468 | 0.000221416 | 1.16236e-05 |
| loss/train_epoch                       | 0.000227897 | 0.000234378 | 0.000239494 |     3 | 0.000244609 | 0.000233468 | 0.000221416 | 1.16236e-05 |
| loss/train_step                        | 6.15492e-05 | 7.71235e-05 | 0.000119097 |     3 |  0.00016107 | 9.47229e-05 | 4.59749e-05 | 5.95319e-05 |
| loss/val                               |     1.46896 |     1.55501 |      2.2539 |     3 |      2.9528 |     1.96357 |      1.3829 |    0.861007 |
| metric/Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Arguing/f1/val                  |    0.584126 |    0.606952 |    0.608653 |     3 |    0.610354 |    0.592869 |      0.5613 |   0.0273923 |
| metric/Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Default Illocuting/f1/val       |    0.570713 |    0.588235 |    0.597148 |     3 |    0.606061 |    0.582496 |    0.553191 |   0.0268978 |
| metric/Disagreeing/f1/train            |    0.999289 |    0.999289 |    0.999645 |     3 |           1 |    0.999526 |    0.999289 | 0.000410338 |
| metric/Disagreeing/f1/val              |    0.340663 |    0.342342 |    0.397734 |     3 |    0.453125 |     0.37815 |    0.338983 |   0.0649519 |
| metric/NONE/f1/train                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/NONE/f1/val                     |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Restating/f1/val                |    0.568239 |    0.588435 |    0.589969 |     3 |    0.591503 |    0.575994 |    0.548043 |   0.0242549 |
| metric/Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/macro/f1/train                  |    0.899929 |    0.899929 |    0.949964 |     3 |           1 |    0.933286 |    0.899929 |   0.0577761 |
| metric/macro/f1/val                    |    0.258159 |    0.260697 |    0.271472 |     3 |    0.282247 |    0.266189 |    0.255622 |   0.0141365 |
| metric/micro/f1/train                  |    0.999877 |    0.999877 |    0.999938 |     3 |           1 |    0.999918 |    0.999877 | 7.11656e-05 |
| metric/micro/f1/val                    |    0.471604 |    0.478842 |    0.493318 |     3 |    0.507795 |    0.483667 |    0.464365 |   0.0221134 |

</details>

### YA I2L relations with RoBERTa (lr=1e-4)

- training a single model for YA-relation types (between S and TA nodes) with xlm-roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_ya_i2l \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=roberta-single-relations \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/roberta-single-relations-training/runs/5ilcukyf
    - seed2: https://wandb.ai/tanikina/roberta-single-relations-training/runs/zzhwvfvs
    - seed3: https://wandb.ai/tanikina/roberta-single-relations-training/runs/pj6yzcqa
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_09-20-09`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_11-05-43`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/roberta-single-relations/2024-04-25_12-51-43`
  - aggregated metric values: macro/f1/val: 0.357, micro/f1/val: 0.960

<details>

|                                        |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                             | 3.78374e-05 | 4.22252e-05 |  5.7257e-05 |     3 | 7.22887e-05 | 4.93212e-05 | 3.34495e-05 | 2.03687e-05 |
| loss/train_epoch                       | 3.78374e-05 | 4.22252e-05 |  5.7257e-05 |     3 | 7.22887e-05 | 4.93212e-05 | 3.34495e-05 | 2.03687e-05 |
| loss/train_step                        | 1.90816e-05 | 3.31962e-05 | 4.01399e-05 |     3 | 4.70837e-05 | 2.84156e-05 |   4.967e-06 | 2.14615e-05 |
| loss/val                               |    0.331133 |    0.340382 |     0.34974 |     3 |    0.359098 |    0.340455 |    0.321884 |   0.0186072 |
| metric/Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Arguing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Asserting/f1/val                |    0.985943 |    0.986838 |    0.986842 |     3 |    0.986846 |    0.986244 |    0.985048 |  0.00103564 |
| metric/Assertive Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Assertive Questioning/f1/val    |    0.479266 |    0.484848 |    0.492424 |     3 |         0.5 |    0.486178 |    0.473684 |   0.0132081 |
| metric/Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Challenging/f1/val              |           0 |           0 |    0.166667 |     3 |    0.333333 |    0.111111 |           0 |     0.19245 |
| metric/Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Default Illocuting/f1/val       |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/NONE/f1/train                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/NONE/f1/val                     |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Pure Questioning/f1/val         |    0.800975 |    0.803828 |    0.804692 |     3 |    0.805556 |    0.802502 |    0.798122 |  0.00389011 |
| metric/Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Restating/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/Rhetorical Questioning/f1/val   |    0.299683 |    0.372093 |    0.396573 |     3 |    0.421053 |    0.340139 |    0.227273 |    0.100764 |
| metric/macro/f1/train                  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                    |    0.344627 |    0.353578 |     0.36711 |     3 |    0.380641 |    0.356632 |    0.335676 |   0.0226373 |
| metric/micro/f1/train                  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                    |     0.95914 |    0.961828 |    0.962097 |     3 |    0.962366 |    0.960215 |    0.956452 |   0.0032703 |

</details>
