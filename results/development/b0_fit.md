# Fit of B0 to the training data

Date: 2026-11-16.
Sources: `history.csv` in the folders `b0_f0`, `b0_f1`, and `b0_f2` of this folder, and `b0_folds.json`.
B0 is the R1 config on the 3 development folds. No test driver is read.
Each accuracy and F1 value is the value of the file times 100, with 2 decimals. Each loss value has 4 decimals.

## Result

- B0 fits the training images almost fully. The training accuracy is 99.99 or more in epoch 19 of each fold.
- The validation scores do not fall in the late epochs. Thus the curves show no classic overfit (a validation score that falls with more training).
- The training accuracy is 10.30 to 13.45 points above the validation accuracy in epoch 19. This gap is the gap to new drivers.

## Fit to the training images

The second column gives the first epoch with a training accuracy of 99.00 or more. The other columns give epoch 19.

| Fold | First epoch at 99.00 | Training accuracy | Training loss | Validation accuracy | Gap |
|---|---|---|---|---|---|
| 0. | 6. | 99.99. | 0.0004. | 86.54. | 13.45. |
| 1. | 4. | 100.00. | 0.0002. | 89.70. | 10.30. |
| 2. | 5. | 100.00. | 0.0002. | 88.70. | 11.30. |

The training accuracy counts the augmented images of each epoch.

## Validation scores through the epochs

Each F1 value is the mean validation macro F1 of the epochs in its column name. The loss values are the validation loss.

| Fold | F1, epochs 3–10 | F1, epochs 11–19 | Lowest loss | Its epoch | Loss, epoch 19 |
|---|---|---|---|---|---|
| 0. | 86.66. | 87.32. | 0.3867. | 8. | 0.5408. |
| 1. | 87.73. | 89.56. | 0.3745. | 3. | 0.5592. |
| 2. | 84.91. | 87.25. | 0.4199. | 9. | 0.5127. |

- In each fold, the mean macro F1 of epochs 11 to 19 is higher than the mean of epochs 3 to 10.
- The validation loss of epoch 19 is above the lowest value in each fold.
- The mean validation loss of epochs 3 to 10 is 0.4988, 0.5332, and 0.6267 for the folds 0, 1, and 2.
- The mean validation loss of epochs 11 to 19 is 0.5168, 0.5497, and 0.5708 for the same folds.
- Thus the validation loss does not climb through the late epochs.
- A probable cause of the higher loss is high confidence on wrong validation images. The run files do not hold the confidence values.

## Noise of the validation scores

Each column gives a value of the validation accuracy. The step is the largest change between two adjacent epochs.

| Fold | Standard deviation, epochs 5 to 19 | Step, epochs 5 to 19 | Range, epochs 15 to 19 |
|---|---|---|---|
| 0. | 1.49. | 3.93. | 2.74. |
| 1. | 1.18. | 2.70. | 0.40. |
| 2. | 3.29. | 12.61. | 0.64. |

- In fold 2, the validation accuracy is 89.90 in epoch 7, 77.30 in epoch 8, and 89.21 in epoch 9.
- The learning rate is small in epochs 15 to 19, and the scores of these epochs are stable. This agrees with decision P12, the score of the last epoch.

## An early stop

- An early stop picks the epoch with the lowest validation loss.
- The epochs with the lowest validation loss have a mean macro F1 of 89.06. The last epochs have a mean macro F1 of 88.16.
- The difference of 0.90 points is optimistic, because the same validation images select the epoch.
- The difference is smaller than the largest change between two epochs in each fold.
- Phase 3 has no validation part to select an epoch. Thus the data gives no reason for an early stop.

## Meaning

- A training accuracy near 100 is not a fault by itself. The score on new drivers is the measure.
- A model can learn the look of each training driver. The split by driver keeps each validation driver out of training.
- Thus the validation score measures new persons, and the gap of 10.30 to 13.45 points is the real problem.
- Live use adds a new car, a new camera, and new light. The size of the gap in live use is not measured.
- The plan tests probable causes of the gap: E2 for the start weights, E3 for the augmentation, and E5 for the number of drivers.
- B0 stays equal to R1, because B0 is the reference for each change of Phase 2.

## Not checked

- The scores of each validation driver and of each class.
- The confidence values of the wrong predictions.
- The run `b0_f1` continued from `last.pt` after epoch 10. A run with no stop can give other values.
