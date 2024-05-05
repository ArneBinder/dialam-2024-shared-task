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

- training a single model for all relation types with roberta-large
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

- training a single model for all relation types with roberta-large
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

- training a single model for all relation types with roberta-large
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

- training a single model for all relation types with roberta-large, experimenting with the base-model learning rate
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

- training a single model for all relation types with roberta-large, experimenting with the base-model learning rate
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

- training a single model for all relation types with roberta-large, experimenting with the base-model learning rate
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

- training a single model for s-relation types with roberta-large
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

- training a single model for YA-relation types (between S and TA nodes) with roberta-large
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

- training a single model for YA-relation types (between I and L nodes) with roberta-large
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

## 2024-04-29

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=2e-6)

- training a single model for all relation types with roberta-large, experimenting with the base-model learning rate
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      model.learning_rate=2e-6 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/xm1j3aem
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/slr2ujms
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/0iemuxr3
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_17-09-09`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_22-01-01`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-28_02-52-38`
  - aggregated metric values: macro/f1/val: 0.331, micro/f1/val: 0.678

<details>

|                                                      |         25% |         50% |         75% | count |        max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ---------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000680504 | 0.000700527 |  0.00231092 |     3 | 0.00392131 |  0.00176077 | 0.000660482 |  0.00187119 |
| loss/train_epoch                                     | 0.000680504 | 0.000700527 |  0.00231092 |     3 | 0.00392131 |  0.00176077 | 0.000660482 |  0.00187119 |
| loss/train_step                                      | 0.000464073 | 0.000653444 | 0.000665977 |     3 | 0.00067851 | 0.000535552 | 0.000274702 |  0.00022625 |
| loss/val                                             |      2.1554 |     2.33091 |       2.333 |     3 |     2.3351 |      2.2153 |     1.97989 |    0.203883 |
| metric/macro/f1/train                                |    0.994516 |    0.994808 |    0.996878 |     3 |   0.998949 |    0.995994 |    0.994225 |  0.00257579 |
| metric/macro/f1/val                                  |    0.327649 |    0.327807 |     0.33276 |     3 |   0.337713 |    0.331004 |    0.327491 |  0.00581243 |
| metric/micro/f1/train                                |    0.999873 |    0.999902 |    0.999922 |     3 |   0.999941 |    0.999895 |    0.999843 | 4.93506e-05 |
| metric/micro/f1/val                                  |    0.674776 |    0.674955 |    0.679964 |     3 |   0.684973 |    0.678175 |    0.674598 |  0.00588984 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998665 |    0.998665 |    0.998998 |     3 |   0.999332 |    0.998887 |    0.998665 | 0.000385148 |
| metric/s_nodes:Default Conflict/f1/val               |    0.223216 |    0.230216 |    0.243763 |     3 |    0.25731 |    0.234581 |    0.216216 |   0.0208917 |
| metric/s_nodes:Default Inference-rev/f1/train        |    0.999861 |           1 |           1 |     3 |          1 |    0.999907 |    0.999722 | 0.000160226 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.344434 |    0.351852 |    0.359782 |     3 |   0.367713 |    0.352194 |    0.337017 |   0.0153511 |
| metric/s_nodes:Default Inference/f1/train            |    0.999874 |           1 |           1 |     3 |          1 |    0.999916 |    0.999748 | 0.000145669 |
| metric/s_nodes:Default Inference/f1/val              |    0.349395 |    0.358744 |    0.382846 |     3 |   0.406948 |    0.368579 |    0.340045 |   0.0345188 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.501919 |      0.5086 |    0.519277 |     3 |   0.529954 |    0.511264 |    0.495238 |   0.0175106 |
| metric/s_nodes:NONE/f1/train                         |    0.999876 |    0.999876 |    0.999907 |     3 |   0.999938 |    0.999897 |    0.999876 | 3.58237e-05 |
| metric/s_nodes:NONE/f1/val                           |    0.654159 |    0.654655 |    0.663781 |     3 |   0.672907 |    0.660409 |    0.653664 |   0.0108355 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |    0.973684 |           1 |           1 |     3 |          1 |    0.982456 |    0.947368 |   0.0303868 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |    0.928571 |           1 |           1 |     3 |          1 |    0.952381 |    0.857143 |   0.0824786 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |    0.999952 |    0.999968 |    0.999968 |     3 |   0.999968 |    0.999958 |    0.999936 | 1.83764e-05 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.966387 |    0.966735 |    0.967097 |     3 |   0.967458 |    0.966744 |    0.966038 | 0.000710286 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.197198 |    0.206897 |    0.236782 |     3 |   0.266667 |    0.220354 |      0.1875 |   0.0412635 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |    0.988372 |           1 |           1 |     3 |          1 |    0.992248 |    0.976744 |   0.0134268 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.223724 |       0.325 |    0.386638 |     3 |   0.448276 |    0.298575 |    0.122449 |    0.164513 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.744115 |    0.752381 |     0.75945 |     3 |    0.76652 |    0.751583 |    0.735849 |   0.0153509 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |    0.998674 |           1 |           1 |     3 |          1 |    0.999116 |    0.997347 |  0.00153144 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.298413 |    0.311111 |    0.360101 |     3 |   0.409091 |    0.335305 |    0.285714 |   0.0651496 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |    0.999863 |           1 |           1 |     3 |          1 |    0.999909 |    0.999726 |  0.00015792 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.431326 |     0.43257 |    0.441832 |     3 |   0.451093 |    0.437915 |    0.430082 |   0.0114801 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |    0.964286 |           1 |           1 |     3 |          1 |     0.97619 |    0.928571 |   0.0412393 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.460663 |    0.472727 |    0.473206 |     3 |   0.473684 |    0.465003 |    0.448598 |   0.0142153 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998934 |    0.999289 |     0.99929 |     3 |    0.99929 |    0.999053 |     0.99858 | 0.000410046 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.191218 |     0.19403 |    0.215659 |     3 |   0.237288 |    0.206575 |    0.188406 |   0.0267469 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999919 |    0.999946 |    0.999946 |     3 |   0.999946 |    0.999928 |    0.999891 | 3.13156e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.686236 |    0.686522 |    0.688648 |     3 |   0.690774 |    0.687749 |     0.68595 |  0.00263557 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.450073 |    0.462366 |    0.462485 |     3 |   0.462604 |     0.45425 |    0.437781 |   0.0142631 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |          1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |          0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, learning_rate=1e-6)

- training a single model for all relation types with roberta-large, experimenting with the base-model learning rate
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      model.learning_rate=1e-6 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/dbpr5oxe
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/3psjaqxy
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/wdbni044
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_17-09-17`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_22-01-19`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-28_02-53-56`
  - aggregated metric values: macro/f1/val: 0.217, micro/f1/val: 0.496

<details>

|                                                      |       25% |       50% |       75% | count |      max |      mean |       min |         std |
| :--------------------------------------------------- | --------: | --------: | --------: | ----: | -------: | --------: | --------: | ----------: |
| loss/train                                           |  0.485969 |  0.487393 |  0.489783 |     3 | 0.492172 |  0.488037 |  0.484544 |  0.00385462 |
| loss/train_epoch                                     |  0.485969 |  0.487393 |  0.489783 |     3 | 0.492172 |  0.488037 |  0.484544 |  0.00385462 |
| loss/train_step                                      |  0.369703 |  0.493096 |  0.512468 |     3 |  0.53184 |  0.423749 |  0.246311 |    0.154882 |
| loss/val                                             |   2.33167 |   2.38502 |   2.50561 |     3 |   2.6262 |   2.42985 |   2.27833 |    0.178216 |
| metric/macro/f1/train                                |  0.598031 |  0.620516 |  0.629133 |     3 |  0.63775 |  0.611271 |  0.575546 |   0.0321158 |
| metric/macro/f1/val                                  |  0.215375 |  0.218867 |  0.220265 |     3 | 0.221664 |  0.217471 |  0.211883 |  0.00503767 |
| metric/micro/f1/train                                |  0.691928 |  0.692026 |  0.692937 |     3 | 0.693849 |  0.692568 |   0.69183 |  0.00111372 |
| metric/micro/f1/val                                  |  0.494633 |   0.49517 |  0.497496 |     3 | 0.499821 |  0.496363 |  0.494097 |  0.00304289 |
| metric/no_relation/f1/train                          |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/no_relation/f1/val                            |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |  0.529131 |  0.548971 |  0.553326 |     3 |  0.55768 |  0.538647 |  0.509291 |   0.0257936 |
| metric/s_nodes:Default Conflict/f1/val               |  0.117661 |  0.118919 |  0.128209 |     3 |   0.1375 |  0.124274 |  0.116402 |   0.0115232 |
| metric/s_nodes:Default Inference-rev/f1/train        |   0.45018 |  0.450704 |  0.464216 |     3 | 0.477728 |  0.459363 |  0.449656 |   0.0159134 |
| metric/s_nodes:Default Inference-rev/f1/val          |  0.102795 |  0.113924 |  0.138929 |     3 | 0.163934 |  0.123175 | 0.0916667 |   0.0370114 |
| metric/s_nodes:Default Inference/f1/train            |  0.468024 |  0.473527 |   0.47584 |     3 | 0.478153 |    0.4714 |  0.462521 |  0.00803032 |
| metric/s_nodes:Default Inference/f1/val              |  0.162247 |  0.184143 |  0.196106 |     3 | 0.208068 |  0.177521 |  0.140351 |   0.0343408 |
| metric/s_nodes:Default Rephrase/f1/train             |  0.524299 |  0.530789 |  0.530872 |     3 | 0.530955 |  0.526518 |  0.517809 |  0.00754242 |
| metric/s_nodes:Default Rephrase/f1/val               |  0.234786 |   0.24235 |  0.247951 |     3 | 0.253552 |  0.241041 |  0.227222 |   0.0132139 |
| metric/s_nodes:NONE/f1/train                         |  0.458411 |  0.464684 |  0.464875 |     3 | 0.465067 |   0.46063 |  0.452138 |  0.00735652 |
| metric/s_nodes:NONE/f1/val                           |  0.273207 |  0.302477 |  0.328757 |     3 | 0.355036 |  0.300484 |  0.243937 |   0.0555763 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |  0.697479 |  0.823529 |  0.856209 |     3 | 0.888889 |  0.761282 |  0.571429 |    0.167634 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |       0.2 |       0.4 |       0.4 |     3 |      0.4 |  0.266667 |         0 |     0.23094 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |  0.997714 |  0.998126 |  0.998523 |     3 | 0.998919 |  0.998116 |  0.997303 | 0.000808406 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |  0.954185 |   0.95448 |  0.956656 |     3 | 0.958832 |  0.955734 |  0.953891 |  0.00269913 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |   0.90544 |   0.91906 |   0.94808 |     3 | 0.977099 |  0.929327 |  0.891821 |   0.0435564 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |  0.256536 |  0.277778 |  0.295139 |     3 |   0.3125 |  0.275191 |  0.235294 |   0.0386679 |
| metric/ya_i2l_nodes:Challenging/f1/train             |   0.72973 |  0.756757 |  0.828378 |     3 |      0.9 |  0.786486 |  0.702703 |    0.101953 |
| metric/ya_i2l_nodes:Challenging/f1/val               |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |       0.5 |  0.533333 |  0.627778 |     3 | 0.722222 |  0.574074 |  0.466667 |    0.132559 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |  0.916695 |  0.946984 |  0.954775 |     3 | 0.962567 |  0.931985 |  0.886406 |   0.0402347 |
| metric/ya_i2l_nodes:NONE/f1/val                      |  0.196429 |  0.214286 |  0.232143 |     3 |     0.25 |  0.214286 |  0.178571 |   0.0357143 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |  0.990079 |  0.990586 |  0.994506 |     3 | 0.998427 |  0.992862 |  0.989572 |  0.00484613 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |  0.719734 |  0.727273 |  0.747916 |     3 | 0.768559 |  0.736009 |  0.712195 |   0.0291798 |
| metric/ya_i2l_nodes:Restating/f1/train               |  0.660714 |      0.75 |      0.75 |     3 |     0.75 |  0.690476 |  0.571429 |    0.103098 |
| metric/ya_i2l_nodes:Restating/f1/val                 |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |  0.931131 |  0.947368 |  0.965663 |     3 | 0.983957 |   0.94874 |  0.914894 |   0.0345522 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |  0.296703 |  0.307692 |  0.325275 |     3 | 0.342857 |  0.312088 |  0.285714 |   0.0288239 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |  0.403935 |    0.4375 |    0.4375 |     3 |   0.4375 |  0.415123 |   0.37037 |   0.0387573 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |   0.54468 |  0.548615 |  0.552411 |     3 | 0.556207 |  0.548522 |  0.540744 |  0.00773173 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |  0.196398 |  0.206035 |  0.221308 |     3 | 0.236581 |  0.209792 |  0.186761 |   0.0251212 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |  0.285714 |  0.285714 |  0.328042 |     3 |  0.37037 |  0.313933 |  0.285714 |   0.0488762 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |  0.379973 |  0.413793 |   0.45305 |     3 | 0.492308 |  0.417418 |  0.346154 |   0.0731443 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |  0.502244 |  0.523652 |  0.539435 |     3 | 0.555218 |  0.519902 |  0.480836 |   0.0373323 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |  0.175939 |  0.207547 |  0.220278 |     3 |  0.23301 |  0.194962 |   0.14433 |   0.0456597 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |  0.480728 |   0.48578 |  0.514562 |     3 | 0.543345 |    0.5016 |  0.475676 |   0.0365032 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              | 0.0833565 | 0.0847458 | 0.0959443 |     3 | 0.107143 | 0.0912853 | 0.0819672 |   0.0138032 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |  0.621148 |  0.621653 |  0.623977 |     3 | 0.626301 |  0.622865 |  0.620642 |  0.00301796 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |  0.366634 |  0.385305 |  0.403416 |     3 | 0.421528 |  0.384932 |  0.347964 |   0.0367836 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |  0.530769 |       0.6 |  0.684615 |     3 | 0.769231 |  0.610256 |  0.461538 |    0.154102 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.5049 |  0.507397 |  0.508822 |     3 | 0.510246 |  0.506682 |  0.502403 |  0.00397013 |
| metric/ya_s2ta_nodes:Restating/f1/val                |  0.208331 |  0.215385 |  0.238546 |     3 | 0.261708 |  0.226124 |  0.201278 |   0.0316139 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |       0.4 |       0.4 |  0.533333 |     3 | 0.666667 |  0.488889 |       0.4 |     0.15396 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |         0 |         0 |         0 |     3 |        0 |         0 |         0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, window_size=256)

- training a single model for all relation types with roberta-large, experimenting with the window size
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      taskmodule.max_window=256 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/fk6pyjqy
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/26f0sj3j
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/xayp3w82
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_17-21-18`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_19-46-01`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_22-11-29`
  - aggregated metric values: macro/f1/val: 0.378, micro/f1/val: 0.718

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000186682 | 0.000206277 |  0.00085849 |     3 |   0.0015107 | 0.000628022 | 0.000167086 | 0.000764674 |
| loss/train_epoch                                     | 0.000186682 | 0.000206277 |  0.00085849 |     3 |   0.0015107 | 0.000628022 | 0.000167086 | 0.000764674 |
| loss/train_step                                      |  2.7559e-05 | 2.93062e-05 | 0.000389215 |     3 | 0.000749123 |  0.00026808 | 2.58118e-05 | 0.000416599 |
| loss/val                                             |     1.58306 |     1.59055 |       1.743 |     3 |     1.89545 |     1.68719 |     1.57556 |    0.180514 |
| metric/macro/f1/train                                |    0.999874 |     0.99988 |     0.99991 |     3 |     0.99994 |    0.999896 |    0.999868 | 3.86082e-05 |
| metric/macro/f1/val                                  |    0.376783 |    0.381968 |    0.382199 |     3 |    0.382429 |    0.378665 |    0.371599 |  0.00612401 |
| metric/micro/f1/train                                |    0.999912 |    0.999921 |    0.999941 |     3 |    0.999961 |    0.999928 |    0.999902 | 3.00132e-05 |
| metric/micro/f1/val                                  |     0.71229 |    0.714798 |    0.721336 |     3 |    0.727875 |    0.717485 |    0.709781 |  0.00934144 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998997 |    0.999331 |    0.999332 |     3 |    0.999332 |    0.999109 |    0.998663 | 0.000385922 |
| metric/s_nodes:Default Conflict/f1/val               |    0.329811 |    0.339623 |    0.349122 |     3 |    0.358621 |    0.339414 |        0.32 |   0.0193112 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.429308 |     0.42963 |    0.435627 |     3 |    0.441624 |    0.433413 |    0.428986 |   0.0071184 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.394243 |    0.406321 |    0.412879 |     3 |    0.419437 |    0.402641 |    0.382166 |   0.0189063 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.555053 |    0.574257 |    0.579124 |     3 |    0.583991 |    0.564699 |    0.535849 |   0.0254545 |
| metric/s_nodes:NONE/f1/train                         |    0.999907 |    0.999938 |    0.999938 |     3 |    0.999938 |    0.999917 |    0.999876 | 3.59097e-05 |
| metric/s_nodes:NONE/f1/val                           |    0.686343 |    0.698413 |    0.704664 |     3 |    0.710916 |    0.694534 |    0.674273 |   0.0186271 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.979873 |    0.980415 |    0.980757 |     3 |    0.981099 |    0.980282 |     0.97933 | 0.000891885 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.332721 |    0.352941 |    0.432881 |     3 |    0.512821 |    0.392754 |      0.3125 |    0.105929 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.455923 |    0.459016 |    0.519831 |     3 |    0.580645 |    0.497497 |     0.45283 |   0.0720746 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.777478 |    0.794393 |    0.804276 |     3 |    0.814159 |    0.789705 |    0.760563 |   0.0271037 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.304236 |    0.315789 |    0.330309 |     3 |    0.344828 |    0.317767 |    0.292683 |   0.0261285 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |    0.999863 |           1 |           1 |     3 |           1 |    0.999909 |    0.999726 | 0.000158368 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.453276 |    0.464419 |    0.478587 |     3 |    0.492754 |    0.466435 |    0.442133 |   0.0253706 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.504589 |    0.504673 |    0.510957 |     3 |    0.517241 |    0.508806 |    0.504505 |   0.0073055 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998223 |    0.998578 |    0.998933 |     3 |    0.999288 |    0.998578 |    0.997868 | 0.000710219 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.292403 |    0.308943 |    0.328714 |     3 |    0.348485 |    0.311097 |    0.275862 |   0.0363593 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999864 |    0.999891 |    0.999918 |     3 |    0.999946 |    0.999891 |    0.999837 | 5.44488e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.708632 |    0.711382 |    0.726274 |     3 |    0.741165 |    0.719477 |    0.705882 |   0.0189832 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.525841 |    0.527439 |    0.538056 |     3 |    0.548673 |    0.533451 |    0.524242 |   0.0132785 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, window_size=128)

- training a single model for all relation types with roberta-large, experimenting with the window size
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      taskmodule.max_window=128 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/9uyfy2wj
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/wvw5t44v
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/7lk3qnip
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_17-26-02`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_18-48-26`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-27_20-11-28`
  - aggregated metric values: macro/f1/val: 0.400, micro/f1/val: 0.719

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000109819 | 0.000119359 | 0.000127856 |     3 | 0.000136354 | 0.000118664 | 0.000100279 | 1.80474e-05 |
| loss/train_epoch                                     | 0.000109819 | 0.000119359 | 0.000127856 |     3 | 0.000136354 | 0.000118664 | 0.000100279 | 1.80474e-05 |
| loss/train_step                                      | 2.31727e-05 |  3.9323e-05 | 8.26563e-05 |     3 |  0.00012599 |  5.7445e-05 | 7.02248e-06 | 6.15191e-05 |
| loss/val                                             |     1.42807 |     1.68703 |     1.77197 |     3 |     1.85691 |     1.57102 |      1.1691 |    0.358284 |
| metric/macro/f1/train                                |    0.999891 |    0.999906 |    0.999938 |     3 |     0.99997 |    0.999917 |    0.999876 | 4.79724e-05 |
| metric/macro/f1/val                                  |    0.390912 |    0.391248 |    0.405019 |     3 |     0.41879 |    0.400205 |    0.390576 |   0.0160986 |
| metric/micro/f1/train                                |     0.99993 |     0.99994 |     0.99996 |     3 |     0.99998 |    0.999947 |     0.99992 | 3.05335e-05 |
| metric/micro/f1/val                                  |    0.716194 |    0.717194 |    0.721011 |     3 |    0.724827 |    0.719072 |    0.715194 |  0.00508366 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998957 |    0.999304 |    0.999305 |     3 |    0.999305 |    0.999073 |    0.998609 | 0.000401494 |
| metric/s_nodes:Default Conflict/f1/val               |    0.298438 |         0.3 |    0.314773 |     3 |    0.329545 |    0.308807 |    0.296875 |    0.018028 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.369871 |     0.39548 |    0.408771 |     3 |    0.422062 |    0.387268 |    0.344262 |   0.0395448 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |     0.39999 |    0.403893 |    0.424664 |     3 |    0.445434 |    0.415138 |    0.396088 |   0.0265256 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.551044 |    0.568365 |     0.57488 |     3 |    0.581395 |    0.561161 |    0.533724 |   0.0246383 |
| metric/s_nodes:NONE/f1/train                         |    0.999905 |    0.999937 |    0.999937 |     3 |    0.999937 |    0.999916 |    0.999874 | 3.64431e-05 |
| metric/s_nodes:NONE/f1/val                           |    0.697969 |      0.7049 |    0.709292 |     3 |    0.713684 |    0.703207 |    0.691038 |   0.0114177 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |     0.97541 |    0.976961 |    0.977836 |     3 |    0.978711 |     0.97651 |    0.973858 |  0.00245772 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.411338 |    0.432432 |    0.492078 |     3 |    0.551724 |    0.458133 |    0.390244 |   0.0837519 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |    0.333333 |     3 |    0.666667 |    0.222222 |           0 |      0.3849 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.329932 |    0.333333 |    0.359649 |     3 |    0.385965 |     0.34861 |    0.326531 |   0.0325289 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.769307 |    0.823529 |    0.827019 |     3 |    0.830508 |    0.789707 |    0.715084 |   0.0647199 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.363698 |    0.368421 |    0.384211 |     3 |         0.4 |    0.375798 |    0.358974 |   0.0214848 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.453221 |    0.459184 |    0.469251 |     3 |    0.479319 |     0.46192 |    0.447257 |   0.0162049 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.394645 |    0.433735 |     0.45091 |     3 |    0.468085 |    0.419125 |    0.355556 |   0.0576698 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998519 |    0.998519 |    0.999259 |     3 |           1 |    0.999012 |    0.998519 | 0.000855329 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.329593 |    0.330827 |    0.334531 |     3 |    0.338235 |    0.332474 |    0.328358 |  0.00514027 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999889 |    0.999889 |    0.999945 |     3 |           1 |    0.999926 |    0.999889 | 6.39045e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.718056 |    0.719842 |    0.728245 |     3 |    0.736648 |    0.724254 |     0.71627 |   0.0108818 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.509885 |    0.515397 |    0.527445 |     3 |    0.539493 |    0.519755 |    0.504373 |    0.017961 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, window_size=64)

- training a single model for all relation types with roberta-large, experimenting with the window size
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      taskmodule.max_window=64 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      name=bert-base-uncased-model \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/whdwaa9p
    - seed2: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/mi1c6gxj
    - seed3: https://wandb.ai/tanikina/bert-base-uncased-model-training/runs/i3jkfjam
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-28_22-20-52`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-28_22-49-42`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/bert-base-uncased-model/2024-04-28_23-18-31`
  - aggregated metric values: macro/f1/val: 0.353, micro/f1/val: 0.733

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000128074 | 0.000128848 | 0.000143496 |     3 | 0.000158144 | 0.000138097 | 0.000127299 | 1.73782e-05 |
| loss/train_epoch                                     | 0.000128074 | 0.000128848 | 0.000143496 |     3 | 0.000158144 | 0.000138097 | 0.000127299 | 1.73782e-05 |
| loss/train_step                                      | 2.25985e-05 |   2.681e-05 | 8.30262e-05 |     3 | 0.000139242 | 6.14798e-05 |  1.8387e-05 |  6.7476e-05 |
| loss/val                                             |     1.72642 |     1.98983 |     2.22527 |     3 |     2.46072 |     1.97118 |       1.463 |     0.49912 |
| metric/macro/f1/train                                |    0.999965 |     0.99997 |    0.999975 |     3 |     0.99998 |     0.99997 |     0.99996 | 9.89653e-06 |
| metric/macro/f1/val                                  |    0.340788 |    0.351542 |    0.364619 |     3 |    0.377696 |    0.353091 |    0.330034 |   0.0238687 |
| metric/micro/f1/train                                |    0.999924 |    0.999935 |    0.999946 |     3 |    0.999956 |    0.999935 |    0.999913 | 2.17557e-05 |
| metric/micro/f1/val                                  |    0.730624 |    0.731021 |    0.735084 |     3 |    0.739148 |    0.733465 |    0.730228 |  0.00493686 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.288894 |    0.350515 |    0.370173 |     3 |     0.38983 |     0.32254 |    0.227273 |    0.084813 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.348023 |     0.35443 |    0.374707 |     3 |    0.394984 |    0.363677 |    0.341615 |   0.0278602 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |     0.32842 |    0.360544 |    0.365457 |     3 |     0.37037 |    0.342404 |    0.296296 |   0.0402312 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999745 |     0.99983 |     0.99983 |     3 |     0.99983 |    0.999773 |     0.99966 | 9.82828e-05 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.455005 |    0.459559 |    0.474546 |     3 |    0.489533 |    0.466514 |     0.45045 |   0.0204486 |
| metric/s_nodes:NONE/f1/train                         |    0.999895 |     0.99993 |     0.99993 |     3 |     0.99993 |    0.999907 |    0.999861 | 4.02285e-05 |
| metric/s_nodes:NONE/f1/val                           |     0.72123 |    0.722679 |    0.724663 |     3 |    0.726648 |    0.723036 |     0.71978 |  0.00344752 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.977146 |    0.977791 |    0.977829 |     3 |    0.977867 |    0.977386 |    0.976501 | 0.000767449 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.280392 |    0.294118 |    0.298574 |     3 |     0.30303 |    0.287938 |    0.266667 |    0.018953 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |         0.5 |     3 |           1 |    0.333333 |           0 |     0.57735 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.348679 |    0.377358 |    0.415346 |     3 |    0.453333 |    0.383564 |        0.32 |   0.0668829 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.791047 |     0.79638 |    0.802841 |     3 |    0.809302 |    0.797132 |    0.785714 |    0.011812 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.226678 |    0.324324 |    0.328829 |     3 |    0.333333 |     0.26223 |    0.129032 |    0.115441 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.401324 |     0.40678 |    0.428979 |     3 |    0.451178 |    0.417942 |    0.395869 |   0.0292956 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.579051 |    0.636364 |    0.641259 |     3 |    0.646154 |    0.601419 |    0.521739 |   0.0691781 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.284853 |    0.316832 |    0.337363 |     3 |    0.357895 |      0.3092 |    0.252874 |   0.0529249 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999878 |    0.999878 |    0.999908 |     3 |    0.999939 |    0.999898 |    0.999878 | 3.52387e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |     0.74536 |    0.747748 |     0.74939 |     3 |    0.751033 |    0.747251 |    0.742972 |  0.00405349 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.999613 |    0.999613 |     0.99971 |     3 |    0.999806 |    0.999677 |    0.999613 | 0.000111841 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.426214 |    0.444008 |     0.44443 |     3 |    0.444853 |    0.432427 |    0.408421 |   0.0207943 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

## 2024-04-30

### Merged relations with RoBERTa (task_learning_rate=1e-4, fixed validation set)

- training a single model for all relation types with roberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/1nxd3psb
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/7dlwxmr8
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/7vrkcnrt
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_21-25-02`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_23-58-57`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_02-33-17`
  - aggregated metric values: macro/f1/val: 0.395, micro/f1/val: 0.712

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000245281 | 0.000269889 | 0.000323578 |     3 | 0.000377266 | 0.000289276 | 0.000220672 | 8.00771e-05 |
| loss/train_epoch                                     | 0.000245281 | 0.000269889 | 0.000323578 |     3 | 0.000377266 | 0.000289276 | 0.000220672 | 8.00771e-05 |
| loss/train_step                                      | 5.10611e-06 | 7.11278e-06 | 6.78644e-05 |     3 | 0.000128616 | 4.62761e-05 | 3.09943e-06 | 7.13367e-05 |
| loss/val                                             |     1.97426 |     2.19679 |     2.32533 |     3 |     2.45386 |     2.13413 |     1.75172 |     0.35524 |
| metric/macro/f1/train                                |    0.999844 |    0.999846 |    0.999881 |     3 |    0.999915 |    0.999868 |    0.999842 |  4.0977e-05 |
| metric/macro/f1/val                                  |    0.392823 |    0.393396 |    0.396318 |     3 |     0.39924 |    0.394962 |    0.392251 |  0.00374882 |
| metric/micro/f1/train                                |    0.999842 |    0.999842 |    0.999872 |     3 |    0.999901 |    0.999862 |    0.999842 | 3.41375e-05 |
| metric/micro/f1/val                                  |    0.708068 |     0.71753 |    0.718906 |     3 |    0.720282 |    0.712139 |    0.698607 |   0.0118004 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998649 |    0.998649 |    0.998987 |     3 |    0.999325 |    0.998874 |    0.998649 | 0.000390378 |
| metric/s_nodes:Default Conflict/f1/val               |    0.324164 |    0.335404 |    0.341895 |     3 |    0.348387 |    0.332239 |    0.312925 |   0.0179416 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.347758 |    0.348485 |    0.354614 |     3 |    0.360743 |    0.352087 |    0.347032 |  0.00753162 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.387198 |    0.393443 |    0.405055 |     3 |    0.416667 |    0.397021 |    0.380952 |    0.018124 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999724 |    0.999724 |    0.999793 |     3 |    0.999862 |     0.99977 |    0.999724 | 7.95623e-05 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.524649 |    0.570755 |    0.576101 |     3 |    0.581448 |    0.543582 |    0.478544 |   0.0565782 |
| metric/s_nodes:NONE/f1/train                         |    0.999783 |    0.999814 |    0.999814 |     3 |    0.999814 |    0.999794 |    0.999753 | 3.57204e-05 |
| metric/s_nodes:NONE/f1/val                           |    0.686734 |    0.694915 |    0.695025 |     3 |    0.695135 |    0.689534 |    0.678553 |  0.00951103 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.984842 |       0.985 |    0.985844 |     3 |    0.986689 |    0.985458 |    0.984684 |  0.00107794 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.322917 |    0.333333 |    0.380952 |     3 |    0.428571 |    0.358135 |      0.3125 |   0.0618828 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.487146 |    0.487805 |    0.493902 |     3 |         0.5 |     0.49143 |    0.486486 |  0.00745066 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.797455 |    0.816667 |    0.819122 |     3 |    0.821577 |    0.805495 |    0.778243 |   0.0237289 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.362132 |    0.411765 |    0.439216 |     3 |    0.466667 |    0.396977 |      0.3125 |   0.0781399 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |    0.999931 |           1 |           1 |     3 |           1 |    0.999954 |    0.999862 | 7.99408e-05 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.462215 |    0.463646 |    0.466763 |     3 |     0.46988 |     0.46477 |    0.460784 |  0.00465061 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.520782 |    0.553191 |    0.587707 |     3 |    0.622222 |    0.554595 |    0.488372 |   0.0669361 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998559 |    0.998559 |     0.99892 |     3 |     0.99928 |    0.998799 |    0.998559 | 0.000416257 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.296178 |    0.321678 |    0.329589 |     3 |      0.3375 |    0.309952 |    0.270677 |    0.034921 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999783 |    0.999783 |    0.999837 |     3 |    0.999891 |    0.999819 |    0.999783 | 6.26656e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.702781 |     0.71571 |    0.717836 |     3 |    0.719963 |    0.708508 |    0.689852 |   0.0162958 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.999608 |    0.999687 |    0.999765 |     3 |    0.999843 |    0.999687 |     0.99953 | 0.000156671 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.495058 |    0.532637 |    0.535451 |     3 |    0.538265 |     0.50946 |    0.457478 |   0.0451057 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RemBERT (task_learning_rate=1e-4, fixed validation set)

- training a single model for all relation types with rembert
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      +model.classifier_dropout=0.0 \
      base_model_name=google/rembert \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/a3n1eoqu
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/atsw0ymy
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/q00b7pe1
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_21-25-57`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_01-19-51`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_05-16-47`
  - aggregated metric values: macro/f1/val: 0.387, micro/f1/val: 0.699

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 3.77811e-05 |  3.8977e-05 | 0.000120792 |     3 | 0.000202607 |  9.2723e-05 | 3.65853e-05 | 9.51697e-05 |
| loss/train_epoch                                     | 3.77811e-05 |  3.8977e-05 | 0.000120792 |     3 | 0.000202607 |  9.2723e-05 | 3.65853e-05 | 9.51697e-05 |
| loss/train_step                                      | 3.67561e-06 | 5.08624e-06 | 7.80816e-06 |     3 | 1.05301e-05 | 5.96043e-06 | 2.26497e-06 | 4.20133e-06 |
| loss/val                                             |     1.28717 |     1.47559 |     2.01309 |     3 |     2.55059 |     1.70831 |     1.09874 |     0.75338 |
| metric/macro/f1/train                                |    0.999942 |           1 |           1 |     3 |           1 |    0.999961 |    0.999884 | 6.69328e-05 |
| metric/macro/f1/val                                  |    0.382721 |    0.388117 |    0.392188 |     3 |    0.396259 |    0.387233 |    0.377324 |  0.00949859 |
| metric/micro/f1/train                                |    0.999941 |           1 |           1 |     3 |           1 |    0.999961 |    0.999882 | 6.83093e-05 |
| metric/micro/f1/val                                  |     0.69637 |    0.696886 |    0.700843 |     3 |      0.7048 |     0.69918 |    0.695854 |  0.00489399 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.999662 |           1 |           1 |     3 |           1 |    0.999775 |    0.999325 | 0.000389828 |
| metric/s_nodes:Default Conflict/f1/val               |    0.284067 |    0.323529 |    0.347662 |     3 |    0.371795 |     0.31331 |    0.244604 |   0.0642082 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.324245 |    0.325203 |    0.343888 |     3 |    0.362573 |    0.337021 |    0.323288 |   0.0221492 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.370273 |    0.392927 |    0.393559 |     3 |    0.394191 |    0.378246 |    0.347619 |    0.026531 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999862 |           1 |           1 |     3 |           1 |    0.999908 |    0.999724 |  0.00015909 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.532531 |    0.532855 |    0.541338 |     3 |     0.54982 |    0.538294 |    0.532207 |  0.00998693 |
| metric/s_nodes:NONE/f1/train                         |    0.999907 |           1 |           1 |     3 |           1 |    0.999938 |    0.999814 | 0.000107161 |
| metric/s_nodes:NONE/f1/val                           |    0.667202 |    0.670391 |    0.673811 |     3 |    0.677232 |    0.670546 |    0.664014 |   0.0066105 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.984331 |    0.985556 |    0.985689 |     3 |    0.985822 |    0.984828 |    0.983107 |  0.00149629 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.161111 |    0.222222 |    0.304659 |     3 |    0.387097 |     0.23644 |         0.1 |    0.144075 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.508056 |    0.526316 |    0.563158 |     3 |         0.6 |    0.538704 |    0.489796 |   0.0561368 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.814741 |    0.831276 |    0.832993 |     3 |    0.834711 |    0.821398 |    0.798206 |   0.0201576 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.406593 |    0.428571 |    0.475155 |     3 |    0.521739 |    0.444975 |    0.384615 |   0.0700182 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.440488 |    0.451327 |    0.453221 |     3 |    0.455115 |    0.445364 |    0.429648 |   0.0137409 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.528645 |    0.530973 |    0.562784 |     3 |    0.594595 |    0.550628 |    0.526316 |   0.0381474 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |     0.99928 |           1 |           1 |     3 |           1 |     0.99952 |    0.998559 | 0.000831928 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.295813 |    0.338462 |    0.342842 |     3 |    0.347222 |    0.312949 |    0.253165 |   0.0519602 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999919 |           1 |           1 |     3 |           1 |    0.999946 |    0.999837 | 9.40157e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.690239 |    0.690891 |    0.695445 |     3 |         0.7 |    0.693493 |    0.689587 |  0.00567297 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.999922 |           1 |           1 |     3 |           1 |    0.999948 |    0.999843 | 9.04367e-05 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.470372 |    0.476071 |    0.485376 |     3 |    0.494681 |    0.478475 |    0.464674 |   0.0151473 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with ELECTRA (task_learning_rate=1e-4, fixed validation set)

- training a single model for all relation types with electra-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=google/electra-large-generator \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/02gk72x1
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/pylqb0zx
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/ggcr2af2
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_21-26-17`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_22-05-59`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_22-45-38`
  - aggregated metric values: macro/f1/val: 0.303, micro/f1/val: 0.630

<details>

|                                                      |         25% |         50% |        75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ---------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           |  0.00392987 |  0.00406064 | 0.00450584 |     3 |  0.00495104 |  0.00427026 |   0.0037991 | 0.000603901 |
| loss/train_epoch                                     |  0.00392987 |  0.00406064 | 0.00450584 |     3 |  0.00495104 |  0.00427026 |   0.0037991 | 0.000603901 |
| loss/train_step                                      | 0.000503714 | 0.000719176 | 0.00075972 |     3 | 0.000800264 | 0.000602564 | 0.000288252 | 0.000275205 |
| loss/val                                             |     1.78498 |     1.79348 |     2.0882 |     3 |     2.38292 |      1.9843 |     1.77648 |    0.345327 |
| metric/macro/f1/train                                |     0.84561 |    0.850531 |   0.861771 |     3 |     0.87301 |    0.854743 |    0.840689 |   0.0165675 |
| metric/macro/f1/val                                  |    0.300085 |    0.303397 |   0.306673 |     3 |    0.309949 |    0.303373 |    0.296772 |  0.00658869 |
| metric/micro/f1/train                                |     0.99932 |    0.999409 |   0.999448 |     3 |    0.999487 |    0.999376 |    0.999231 | 0.000131272 |
| metric/micro/f1/val                                  |    0.629881 |    0.630139 |   0.630913 |     3 |    0.631688 |    0.630483 |    0.629623 |  0.00107432 |
| metric/no_relation/f1/train                          |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |          1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.132965 |    0.133333 |    0.15841 |     3 |    0.183486 |    0.149805 |    0.132597 |   0.0291708 |
| metric/s_nodes:Default Inference-rev/f1/train        |     0.99986 |           1 |          1 |     3 |           1 |    0.999907 |     0.99972 | 0.000161946 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.244583 |     0.25522 |   0.267241 |     3 |    0.279261 |    0.256142 |    0.233945 |    0.022672 |
| metric/s_nodes:Default Inference/f1/train            |    0.999872 |           1 |          1 |     3 |           1 |    0.999914 |    0.999743 | 0.000148147 |
| metric/s_nodes:Default Inference/f1/val              |    0.263362 |    0.263566 |   0.274889 |     3 |    0.286213 |    0.270979 |    0.263158 |   0.0131946 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999931 |           1 |          1 |     3 |           1 |    0.999954 |    0.999862 | 7.95279e-05 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.412385 |    0.423707 |   0.432494 |     3 |    0.441281 |    0.422017 |    0.401062 |   0.0201626 |
| metric/s_nodes:NONE/f1/train                         |           1 |           1 |          1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:NONE/f1/val                           |    0.596531 |    0.601964 |   0.602312 |     3 |     0.60266 |    0.598574 |    0.591098 |  0.00648398 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |         0.6 |         0.8 |   0.828571 |     3 |    0.857143 |    0.685714 |         0.4 |     0.24908 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |         0.4 |         0.4 |   0.533333 |     3 |    0.666667 |    0.488889 |         0.4 |     0.15396 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |     0.99992 |    0.999968 |   0.999968 |     3 |    0.999968 |    0.999936 |    0.999872 | 5.54734e-05 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.979115 |    0.979944 |   0.981375 |     3 |    0.982806 |    0.980345 |    0.978285 |   0.0022872 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |    0.991425 |    0.995074 |   0.997537 |     3 |           1 |    0.994283 |    0.987775 |  0.00615071 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |   0.0819398 |   0.0869565 |   0.118478 |     3 |        0.15 |    0.104627 |   0.0769231 |   0.0396135 |
| metric/ya_i2l_nodes:Challenging/f1/train             |    0.852068 |    0.878049 |   0.908412 |     3 |    0.938776 |     0.88097 |    0.826087 |   0.0564011 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |    0.815349 |        0.84 |   0.897273 |     3 |    0.954545 |    0.861748 |    0.790698 |    0.084061 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |    0.992015 |    0.998384 |   0.998384 |     3 |    0.998384 |    0.994138 |    0.985646 |  0.00735459 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.222394 |    0.228571 |   0.233333 |     3 |    0.238095 |    0.227628 |    0.216216 |     0.01097 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |          1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.769715 |    0.783333 |   0.790826 |     3 |    0.798319 |     0.77925 |    0.756098 |    0.021405 |
| metric/ya_i2l_nodes:Restating/f1/train               |           0 |           0 |   0.166667 |     3 |    0.333333 |    0.111111 |           0 |     0.19245 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |    0.996097 |    0.997403 |   0.997403 |     3 |    0.997403 |    0.996532 |    0.994792 |  0.00150742 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.322944 |    0.342857 |       0.35 |     3 |    0.357143 |    0.334343 |     0.30303 |   0.0280429 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |    0.838036 |    0.848485 |   0.875855 |     3 |    0.903226 |    0.859766 |    0.827586 |   0.0390612 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |    0.999446 |    0.999585 |   0.999654 |     3 |    0.999723 |    0.999539 |    0.999308 |  0.00021138 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.355226 |    0.359212 |   0.373219 |     3 |    0.387226 |    0.365892 |     0.35124 |   0.0189002 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |    0.524297 |    0.695652 |   0.809365 |     3 |    0.923077 |    0.657223 |    0.352941 |    0.287004 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |    0.844685 |    0.882353 |   0.895722 |     3 |    0.909091 |    0.866154 |    0.807018 |   0.0529297 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |    0.997788 |    0.997788 |    0.99834 |     3 |    0.998893 |    0.998156 |    0.997788 | 0.000637978 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.452709 |    0.457143 |   0.458571 |     3 |        0.46 |     0.45514 |    0.448276 |  0.00611341 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.996056 |    0.997843 |   0.998202 |     3 |    0.998561 |    0.996891 |    0.994269 |  0.00229885 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.114395 |    0.136126 |   0.139129 |     3 |    0.142132 |    0.123641 |   0.0926641 |    0.026994 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999973 |           1 |          1 |     3 |           1 |    0.999982 |    0.999946 | 3.13156e-05 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.628968 |    0.630886 |   0.636114 |     3 |    0.641342 |    0.633092 |    0.627049 |  0.00739735 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |    0.966667 |           1 |          1 |     3 |           1 |    0.977778 |    0.933333 |     0.03849 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |          1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.361735 |     0.37037 |   0.372419 |     3 |    0.374468 |    0.365979 |      0.3531 |   0.0113407 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |          0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with DeBERTa (task_learning_rate=1e-4, fixed validation set)

- training a single model for all relation types with deberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=microsoft/deberta-large \
      +model.classifier_dropout=0.1 \
      datamodule.batch_size=8 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/6akppd29
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/kcq4glio
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/9n6r81r2
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-29_21-25-31`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_02-59-13`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_08-31-51`
  - aggregated metric values: macro/f1/val: 0.412, micro/f1/val: 0.715

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 8.80917e-06 |  1.0019e-05 | 1.39204e-05 |     3 | 1.78218e-05 | 1.18134e-05 | 7.59937e-06 | 5.34226e-06 |
| loss/train_epoch                                     | 8.80917e-06 |  1.0019e-05 | 1.39204e-05 |     3 | 1.78218e-05 | 1.18134e-05 | 7.59937e-06 | 5.34226e-06 |
| loss/train_step                                      | 2.72193e-06 | 3.97362e-06 | 1.39272e-05 |     3 | 2.38808e-05 |  9.7749e-06 | 1.47024e-06 | 1.22801e-05 |
| loss/val                                             |     2.16461 |     2.45645 |     2.46373 |     3 |     2.47101 |     2.26674 |     1.87277 |    0.341268 |
| metric/macro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                                  |    0.403147 |    0.407722 |    0.417989 |     3 |    0.428255 |    0.411516 |    0.398572 |    0.015201 |
| metric/micro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                                  |    0.712111 |    0.712197 |    0.715723 |     3 |     0.71925 |    0.714491 |    0.712025 |   0.0041227 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.311265 |    0.318182 |    0.335192 |     3 |    0.352201 |     0.32491 |    0.304348 |    0.024626 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.349874 |    0.354571 |    0.356355 |     3 |     0.35814 |    0.352629 |    0.345178 |  0.00669547 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.384383 |    0.386792 |    0.404576 |     3 |     0.42236 |    0.397042 |    0.381974 |   0.0220579 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.567741 |     0.57047 |    0.573114 |     3 |    0.575758 |    0.570413 |    0.565012 |  0.00537311 |
| metric/s_nodes:NONE/f1/train                         |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:NONE/f1/val                           |    0.688034 |    0.688034 |    0.690361 |     3 |    0.692688 |    0.689586 |    0.688034 |  0.00268705 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.986088 |    0.986096 |    0.986107 |     3 |    0.986119 |    0.986098 |     0.98608 | 1.94371e-05 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.183448 |    0.206897 |     0.28202 |     3 |    0.357143 |    0.241346 |        0.16 |    0.102988 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |    0.333333 |     3 |    0.666667 |    0.222222 |           0 |      0.3849 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.586771 |    0.594595 |    0.611583 |     3 |    0.628571 |    0.600704 |    0.578947 |     0.02537 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.798149 |    0.808163 |    0.817209 |     3 |    0.826255 |    0.807518 |    0.788136 |   0.0190678 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |     0.40658 |    0.434783 |    0.439614 |     3 |    0.444444 |    0.419202 |    0.378378 |   0.0356827 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.464927 |    0.479616 |    0.488305 |     3 |    0.496994 |    0.475616 |    0.450237 |   0.0236338 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.554455 |    0.554455 |    0.567826 |     3 |    0.581197 |    0.563369 |    0.554455 |    0.015439 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.338308 |    0.343284 |     0.36845 |     3 |    0.393617 |    0.356745 |    0.333333 |   0.0323177 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.697788 |    0.699294 |    0.706059 |     3 |    0.712825 |      0.7028 |    0.696282 |  0.00881153 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.514607 |    0.517711 |    0.524438 |     3 |    0.531165 |    0.520126 |    0.511502 |   0.0100516 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

## 2024-05-01

### Merged relations with DeBERTa (task_learning_rate=1e-4, max_window=128, fixed validation set)

- training a single model for all relation types with deberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=microsoft/deberta-large \
      +model.classifier_dropout=0.1 \
      datamodule.batch_size=8 \
      taskmodule.max_window=128 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/rspve2j3
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/ndfkbjlk
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/u08pq87p
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_20-18-07`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-04-30_23-01-13`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-01_01-32-45`
  - aggregated metric values: macro/f1/val: 0.386, micro/f1/val: 0.719

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 7.60937e-06 | 9.15641e-06 | 1.19509e-05 |     3 | 1.47453e-05 | 9.98802e-06 | 6.06233e-06 | 4.40082e-06 |
| loss/train_epoch                                     | 7.60937e-06 | 9.15641e-06 | 1.19509e-05 |     3 | 1.47453e-05 | 9.98802e-06 | 6.06233e-06 | 4.40082e-06 |
| loss/train_step                                      | 2.98023e-07 | 3.97364e-07 | 1.49011e-06 |     3 | 2.58286e-06 | 1.05964e-06 | 1.98682e-07 | 1.32289e-06 |
| loss/val                                             |     1.61885 |     1.85326 |     1.92791 |     3 |     2.00256 |     1.74675 |     1.38443 |    0.322538 |
| metric/macro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/macro/f1/val                                  |    0.379358 |    0.381192 |    0.383809 |     3 |    0.386427 |    0.381715 |    0.377525 |  0.00447408 |
| metric/micro/f1/train                                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/micro/f1/val                                  |    0.711346 |    0.711873 |    0.715215 |     3 |    0.718558 |     0.71375 |    0.710818 |  0.00419713 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.314063 |    0.328125 |    0.362693 |     3 |     0.39726 |    0.341795 |         0.3 |   0.0500504 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.356064 |    0.358354 |    0.367762 |     3 |    0.377171 |    0.363099 |    0.353774 |   0.0123998 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.366898 |    0.375587 |    0.381092 |     3 |    0.386598 |    0.373465 |    0.358209 |    0.014313 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.523051 |     0.53753 |    0.550367 |     3 |    0.563204 |    0.536435 |    0.508571 |   0.0273327 |
| metric/s_nodes:NONE/f1/train                         |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:NONE/f1/val                           |    0.686485 |    0.691176 |    0.694202 |     3 |    0.697228 |    0.690066 |    0.681793 |  0.00777725 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.983469 |    0.983907 |    0.984458 |     3 |    0.985008 |    0.983982 |    0.983032 | 0.000990322 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.243833 |    0.294118 |    0.297059 |     3 |         0.3 |    0.262555 |    0.193548 |   0.0598341 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.291777 |    0.307692 |    0.315136 |     3 |    0.322581 |    0.302045 |    0.275862 |   0.0238658 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.804844 |    0.813008 |     0.81946 |     3 |    0.825911 |    0.811867 |    0.796681 |   0.0146486 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.442191 |    0.470588 |    0.475294 |     3 |        0.48 |    0.454794 |    0.413793 |   0.0358181 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.444468 |    0.464037 |    0.473991 |     3 |    0.483945 |    0.457627 |    0.424899 |   0.0300406 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.470956 |    0.482143 |    0.514756 |     3 |    0.547368 |    0.496427 |     0.45977 |   0.0455126 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |    0.322932 |    0.345865 |    0.367177 |     3 |    0.388489 |    0.344785 |         0.3 |   0.0442545 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.705881 |    0.708148 |    0.709788 |     3 |    0.711429 |     0.70773 |    0.703614 |  0.00392377 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.504415 |    0.512755 |    0.513394 |     3 |    0.514032 |    0.507621 |    0.496075 |   0.0100192 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

### Merged relations with RoBERTa (task_learning_rate=1e-4, max_window=128, fixed validation set)

- training a single model for all relation types with deberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      taskmodule.max_window=128 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/jgrix1pc
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/nb35pebo
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/6kq89pfj
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-01_08-25-17`
      - seed2: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-01_09-09-31`
      - seed3: `/netscratch/anikina/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-01_09-55-14`
  - aggregated metric values: macro/f1/val: 0.380, micro/f1/val: 0.713

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 0.000192716 | 0.000224376 | 0.000230493 |     3 |  0.00023661 | 0.000207347 | 0.000161056 | 4.05531e-05 |
| loss/train_epoch                                     | 0.000192716 | 0.000224376 | 0.000230493 |     3 |  0.00023661 | 0.000207347 | 0.000161056 | 4.05531e-05 |
| loss/train_step                                      | 9.64588e-06 | 1.19406e-05 | 1.48612e-05 |     3 | 1.77818e-05 | 1.23579e-05 |  7.3512e-06 |  5.2278e-06 |
| loss/val                                             |     1.71314 |      2.0404 |     2.13898 |     3 |     2.23756 |     1.88795 |     1.38587 |     0.44584 |
| metric/macro/f1/train                                |    0.999881 |     0.99989 |    0.999925 |     3 |    0.999961 |    0.999907 |    0.999871 | 4.74688e-05 |
| metric/macro/f1/val                                  |    0.379185 |    0.382437 |    0.383052 |     3 |    0.383667 |    0.380679 |    0.375934 |  0.00415568 |
| metric/micro/f1/train                                |     0.99988 |      0.9999 |     0.99993 |     3 |     0.99996 |    0.999906 |    0.999859 |  5.0523e-05 |
| metric/micro/f1/val                                  |    0.709411 |     0.71117 |    0.715128 |     3 |    0.719085 |    0.712636 |    0.707652 |  0.00585606 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |    0.998951 |    0.999301 |    0.999301 |     3 |    0.999301 |    0.999068 |    0.998601 |  0.00040404 |
| metric/s_nodes:Default Conflict/f1/val               |    0.362289 |    0.364865 |    0.369529 |     3 |    0.374194 |    0.366257 |    0.359712 |  0.00734032 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.312106 |       0.325 |     0.33042 |     3 |     0.33584 |    0.320017 |    0.299213 |    0.018815 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.388942 |    0.410628 |    0.425653 |     3 |    0.440678 |    0.406188 |    0.367257 |   0.0369115 |
| metric/s_nodes:Default Rephrase/f1/train             |    0.999782 |    0.999855 |    0.999855 |     3 |    0.999855 |    0.999807 |     0.99971 | 8.37607e-05 |
| metric/s_nodes:Default Rephrase/f1/val               |     0.55364 |    0.559809 |    0.560364 |     3 |     0.56092 |    0.556067 |    0.547472 |   0.0074638 |
| metric/s_nodes:NONE/f1/train                         |    0.999811 |    0.999811 |    0.999843 |     3 |    0.999874 |    0.999832 |    0.999811 | 3.63399e-05 |
| metric/s_nodes:NONE/f1/val                           |    0.677557 |    0.679775 |    0.687927 |     3 |    0.696078 |    0.683731 |    0.675339 |   0.0109208 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.982811 |    0.983117 |    0.983938 |     3 |     0.98476 |    0.983461 |    0.982505 |  0.00116645 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.215909 |        0.25 |       0.285 |     3 |        0.32 |    0.250606 |    0.181818 |   0.0690929 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.177778 |    0.222222 |     0.25817 |     3 |    0.294118 |    0.216558 |    0.133333 |   0.0805417 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.808453 |    0.813559 |     0.81478 |     3 |       0.816 |    0.810969 |    0.803347 |  0.00671234 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.392473 |    0.451613 |    0.468231 |     3 |    0.484848 |    0.423265 |    0.333333 |   0.0796362 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.446982 |    0.447904 |    0.473952 |     3 |         0.5 |    0.464655 |    0.446061 |   0.0306236 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.558009 |    0.563636 |    0.568947 |     3 |    0.574257 |    0.563425 |    0.552381 |   0.0109398 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |    0.998881 |    0.999254 |    0.999627 |     3 |           1 |    0.999254 |    0.998507 |  0.00074628 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |     0.34294 |    0.386667 |    0.394772 |     3 |    0.402878 |    0.362919 |    0.299213 |   0.0557636 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |    0.999834 |     0.99989 |    0.999945 |     3 |           1 |     0.99989 |    0.999779 | 0.000110447 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.701503 |    0.703795 |    0.707364 |     3 |    0.710934 |    0.704647 |    0.699211 |   0.0059075 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |    0.999753 |    0.999835 |    0.999918 |     3 |           1 |    0.999835 |    0.999671 | 0.000164688 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.495385 |    0.501377 |    0.506535 |     3 |    0.511692 |     0.50082 |    0.489392 |   0.0111605 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>

## 2024-05-03

### Merged relations with DeBERTa and combined L- and I-node text (task_learning_rate=1e-4, fixed validation set)

- training a single model for all relation types with deberta-large
  - command:
    ```bash
      python src/train.py \
      experiment=dialam2024_merged_relations \
      model.task_learning_rate=1e-4 \
      base_model_name=FacebookAI/roberta-large \
      taskmodule.max_window=128 \
      trainer=gpu \
      seed=1,2,3 \
      +hydra.callbacks.save_job_return.integrate_multirun_result=true \
      --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/zutkrtoc
    - seed2: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/wxc1qiqp
    - seed3: https://wandb.ai/tanikina/dialam2024_merged_relations-re_text_classification_with_indices-training/runs/txi1d3um
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/dialam-li-nodes-text/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-02_23-19-25`
      - seed2: `/netscratch/anikina/dialam-li-nodes-text/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-03_06-00-50`
      - seed3: `/netscratch/anikina/dialam-li-nodes-text/dialam-2024-shared-task/models/dialam2024_merged_relations/re_text_classification_with_indices/2024-05-03_12-41-33`
  - aggregated metric values: macro/f1/val: 0.452, micro/f1/val: 0.722

<details>

|                                                      |         25% |         50% |         75% | count |         max |        mean |         min |         std |
| :--------------------------------------------------- | ----------: | ----------: | ----------: | ----: | ----------: | ----------: | ----------: | ----------: |
| loss/train                                           | 1.00964e-05 | 1.31214e-05 | 3.91285e-05 |     3 | 6.51355e-05 | 2.84428e-05 | 7.07134e-06 | 3.19205e-05 |
| loss/train_epoch                                     | 1.00964e-05 | 1.31214e-05 | 3.91285e-05 |     3 | 6.51355e-05 | 2.84428e-05 | 7.07134e-06 | 3.19205e-05 |
| loss/train_step                                      | 5.66244e-07 | 5.96046e-07 | 1.05053e-06 |     3 | 1.50501e-06 | 8.79167e-07 | 5.36441e-07 | 5.42818e-07 |
| loss/val                                             |     1.54226 |     1.96185 |     1.96656 |     3 |     1.97127 |     1.68526 |     1.12267 |    0.487245 |
| metric/macro/f1/train                                |    0.999983 |           1 |           1 |     3 |           1 |    0.999989 |    0.999966 | 1.98906e-05 |
| metric/macro/f1/val                                  |     0.43726 |    0.440062 |     0.46099 |     3 |    0.481918 |    0.452146 |    0.434458 |    0.025935 |
| metric/micro/f1/train                                |     0.99999 |           1 |           1 |     3 |           1 |    0.999993 |     0.99998 | 1.13906e-05 |
| metric/micro/f1/val                                  |    0.716552 |    0.728448 |    0.731293 |     3 |    0.734138 |    0.722414 |    0.704655 |   0.0156403 |
| metric/no_relation/f1/train                          |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/no_relation/f1/val                            |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/s_nodes:Default Conflict/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Conflict/f1/val               |    0.372624 |    0.392308 |    0.400342 |     3 |    0.408377 |    0.384542 |    0.352941 |   0.0285221 |
| metric/s_nodes:Default Inference-rev/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference-rev/f1/val          |    0.305042 |    0.332481 |    0.371993 |     3 |    0.411504 |    0.340529 |    0.277603 |   0.0673128 |
| metric/s_nodes:Default Inference/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Inference/f1/val              |    0.378772 |    0.399334 |    0.406794 |     3 |    0.414254 |    0.390599 |    0.358209 |   0.0290256 |
| metric/s_nodes:Default Rephrase/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:Default Rephrase/f1/val               |    0.582921 |    0.589372 |    0.589399 |     3 |    0.589426 |     0.58509 |    0.576471 |  0.00746435 |
| metric/s_nodes:NONE/f1/train                         |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/s_nodes:NONE/f1/val                           |    0.662583 |    0.697002 |    0.700247 |     3 |    0.703491 |    0.676219 |    0.628164 |   0.0417436 |
| metric/ya_i2l_nodes:Agreeing/f1/train                |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Agreeing/f1/val                  |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/train                 |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Arguing/f1/val                   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Asserting/f1/val                 |    0.987204 |    0.987208 |    0.988326 |     3 |    0.989444 |    0.987951 |    0.987201 |  0.00129325 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/train   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Assertive Questioning/f1/val     |    0.464115 |    0.473684 |    0.486842 |     3 |         0.5 |    0.476077 |    0.454545 |   0.0228215 |
| metric/ya_i2l_nodes:Challenging/f1/train             |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Challenging/f1/val               |        0.25 |         0.5 |        0.75 |     3 |           1 |         0.5 |           0 |         0.5 |
| metric/ya_i2l_nodes:Default Illocuting/f1/train      |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Default Illocuting/f1/val        |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:NONE/f1/train                    |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:NONE/f1/val                      |    0.617216 |    0.619048 |    0.633848 |     3 |    0.648649 |    0.627694 |    0.615385 |   0.0182397 |
| metric/ya_i2l_nodes:Pure Questioning/f1/train        |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Pure Questioning/f1/val          |    0.895097 |    0.900398 |     0.90118 |     3 |    0.901961 |    0.897385 |    0.889796 |  0.00661865 |
| metric/ya_i2l_nodes:Restating/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Restating/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/train  |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_i2l_nodes:Rhetorical Questioning/f1/val    |    0.330688 |    0.518519 |     0.52849 |     3 |    0.538462 |    0.399946 |    0.142857 |    0.222868 |
| metric/ya_s2ta_nodes:Agreeing/f1/train               |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Agreeing/f1/val                 |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Arguing/f1/train                |    0.999931 |           1 |           1 |     3 |           1 |    0.999954 |    0.999861 | 8.01817e-05 |
| metric/ya_s2ta_nodes:Arguing/f1/val                  |    0.445857 |    0.455399 |    0.483797 |     3 |    0.512195 |     0.46797 |    0.436314 |   0.0394713 |
| metric/ya_s2ta_nodes:Asserting/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Asserting/f1/val                |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/train            |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Challenging/f1/val              |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/train     |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Default Illocuting/f1/val       |    0.628011 |    0.628571 |    0.651495 |     3 |    0.674419 |     0.64348 |    0.627451 |   0.0267992 |
| metric/ya_s2ta_nodes:Disagreeing/f1/train            |     0.99964 |           1 |           1 |     3 |           1 |     0.99976 |    0.999279 | 0.000416257 |
| metric/ya_s2ta_nodes:Disagreeing/f1/val              |     0.37142 |     0.37247 |    0.389532 |     3 |    0.406593 |    0.383144 |     0.37037 |   0.0203345 |
| metric/ya_s2ta_nodes:NONE/f1/train                   |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:NONE/f1/val                     |    0.712721 |    0.735294 |    0.739219 |     3 |    0.743144 |    0.722862 |    0.690149 |   0.0286016 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/train       |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Pure Questioning/f1/val         |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/train              |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Restating/f1/val                |    0.549835 |    0.549932 |    0.564274 |     3 |    0.578616 |    0.559429 |    0.549738 |   0.0166173 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/train |           1 |           1 |           1 |     3 |           1 |           1 |           1 |           0 |
| metric/ya_s2ta_nodes:Rhetorical Questioning/f1/val   |           0 |           0 |           0 |     3 |           0 |           0 |           0 |           0 |

</details>
