# Checks of the random start for E2

Date: 2026-11-16.
Sources: `config.json` and `history.csv` in each folder of `e2_checks/`.
These checks are short runs on the laptop. They are not project runs, and no fold mean comes from them.
Each value is the value of the file times 100, with 2 decimals.

## Reason

- With random start weights, `e2_deit3_small_random` did not learn in 20 epochs. See `e2_deit3_small_random_folds.json`.
- The checks look for a setting with which a random start learns.

## Setup

- Fold 0, 3 epochs, batch size 32, no warm-up, and a cosine curve over the 3 epochs.
- The DeiT-III Small checks start from `configs/e2_deit3_small_random.json`. Each check changes the learning rate or the layer scales.
- The config key `init_values` sets the start value of the layer scales. DeiT-III of `timm` starts them at 0.000001.
- The MobileNetV3-Large check starts from `configs/e1_mobilenetv3_large.json`, with no start weights.
- The GPU is the laptop RTX 3050. The code is commit dff20a6 with the key `init_values` added.

## Results

The two accuracy columns give the training accuracy of epoch 0 and of epoch 2.

| Check | Learning rate | Scale start | Epoch 0 | Epoch 2 | F1 |
|---|---|---|---|---|---|
| `diag_ls1e-06`, DeiT-III Small. | 0.0003. | 0.000001. | 10.29. | 12.62. | 5.37. |
| `diag_ls0.0001`, DeiT-III Small. | 0.0003. | 0.0001. | 10.34. | 10.32. | 2.11. |
| `diag_ls1`, DeiT-III Small. | 0.0003. | 1. | 9.82. | 10.26. | 2.11. |
| `diag_lr5e-5`, DeiT-III Small. | 0.00005. | 0.000001. | 10.85. | 10.59. | 2.76. |
| `diag_mobilenet`, MobileNetV3-Large. | 0.001. | No layer scales. | 10.96. | 38.40. | 25.73. |

The scale start is the start value of the layer scales. F1 is the validation macro F1 of epoch 2 on fold 0.

## Result

- No DeiT-III Small check learns in 3 epochs. The training accuracy stays near 10, the level of a random guess.
- The MobileNetV3-Large check learns. Its training accuracy rises to 38.40 in epoch 2.
- A probable cause: a ViT has no built-in bias for images, so it needs more data and more training. The checks did not test this cause.
- Thus the owner added MobileNetV3-Large with random weights to E2 (decision D18).

## Not checked

- A DeiT-III Small check with more than 3 epochs.
- Other causes, for example the augmentation or the weight decay.
