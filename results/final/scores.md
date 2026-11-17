# Test scores of the final model

Date: 2026-11-17.
Sources: `final_deit3_small_noflip_test_metrics.json`, `final_deit3_small_noflip_test_drivers.csv`, and `final_deit3_small_noflip_test_confusion.csv` in this folder.
The model is DeiT-III Small with the paper set and no flip, trained on the 18 development drivers (decision D19).
The scores come from the last epoch, epoch 19, with seed 42.
The test part has the 8 test drivers of the public split list, with 6525 images.
Each value is the value of the result file times 100, with 2 decimals.

This is the second and last unlock of the test lock. No design choice uses these scores.

## Scores

R1 and the paper value are for comparison only. See `results/replication/scores.md`.

| Score | Final model | R1 | Paper |
|---|---|---|---|
| Accuracy. | 91.97. | 89.46. | 90.36. |
| Class-mean accuracy. | 91.55. | 89.16. | No value. |
| Macro F1. | 91.30. | 88.93. | No value. |
| Precision of the status "distracted". | 98.38. | 96.50. | No value. |
| Recall of the status "distracted". | 97.19. | 97.38. | No value. |
| F1 of the status "distracted". | 97.78. | 96.94. | No value. |

- The paper gives one accuracy for the standard ViT, with a DeiT-III Base model.
- The final model has about a quarter of the parameters of DeiT-III Base. See `results/development/e1.md`.

## Scores of each test driver

| Driver | Images | Accuracy | Class-mean accuracy | Macro F1 |
|---|---|---|---|---|
| p002. | 725. | 92.55. | 90.51. | 89.75. |
| p012. | 823. | 87.85. | 86.65. | 85.71. |
| p026. | 1196. | 92.31. | 91.75. | 91.56. |
| p035. | 848. | 96.34. | 96.29. | 96.07. |
| p041. | 605. | 97.02. | 96.95. | 96.92. |
| p045. | 724. | 95.03. | 95.06. | 94.95. |
| p050. | 790. | 79.87. | 76.71. | 75.25. |
| p075. | 814. | 95.82. | 95.87. | 95.74. |

## Confusion matrix

`final_deit3_small_noflip_test_confusion.csv` holds the matrix. The rows are the true classes, and the columns are the predictions.

- The matrix has 524 errors.
- 134 images of `c9` have the prediction `c0`. This is 23.67 percent of the `c9` images.
- 79 images of `c9` have the prediction `c8`. 45 images of `c0` have the prediction `c9`.
- 210 images of other classes have the prediction `c8`.
- 163 distracted images have the prediction "safe". 134 of them are `c9`.
- 93 safe images have a distracted prediction.
- The left and right hand pairs give 27 errors. 24 images of `c1` have the prediction `c3`.
- On the validation folds, these pairs gave 0 errors. See `results/development/e3_errors.md`.

The F1 of each class:

| Class | F1 |
|---|---|
| `c0`, safe driving. | 83.11. |
| `c1`, texting, right hand. | 92.87. |
| `c2`, phone call, right hand. | 98.56. |
| `c3`, texting, left hand. | 96.65. |
| `c4`, phone call, left hand. | 97.47. |
| `c5`, radio. | 98.21. |
| `c6`, drinking. | 98.20. |
| `c7`, reaching behind. | 97.35. |
| `c8`, hair and makeup. | 82.58. |
| `c9`, talking to passenger. | 68.01. |

## Run

- Training: 20 epochs on the 18 development drivers, with 15872 images in each epoch. Training accuracy of epoch 19: 99.98.
- The sum of `epoch_seconds` in `final_deit3_small_noflip/history.csv` is 1611.57 seconds.
- GPU: NVIDIA L4. `final_deit3_small_noflip/environment.json` gives the library versions and commit ff477a5.
- `results/test_access_log.csv` holds two rows: R1 and this evaluation.

## Not checked

- The AI looked at no test image. Thus the cause of the errors of `c1` as `c3` is not known.
- More than one seed.
