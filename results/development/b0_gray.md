# Scores of B0 on gray validation images

Date: 2026-11-16.
Source: `b0_gray.json` in this folder, written by `scripts/score_gray.py`.
The script scores the last checkpoint of each B0 fold on its validation drivers. No test driver is read.
The gray step turns each image to gray scale with 3 channels after the crop. It is the gray step of 3-Augment.
The GPU is the laptop RTX 3050. Each value is the value of the file times 100, with 2 decimals.

## Scores

| Fold | Images | Color accuracy | Gray accuracy | Color F1 | Gray F1 | F1 change |
|---|---|---|---|---|---|---|
| 0. | 5299. | 86.54. | 80.49. | 86.57. | 81.05. | -5.52. |
| 1. | 5301. | 89.70. | 88.32. | 89.66. | 88.06. | -1.60. |
| 2. | 5299. | 88.70. | 85.36. | 88.26. | 84.62. | -3.63. |
| Mean. | No value. | 88.31. | 84.72. | 88.16. | 84.58. | -3.58. |

F1 is the validation macro F1. The mean validation loss is 0.5376 in color and 0.7166 in gray.

## Result

- The color scores of the laptop are equal to the values of `metrics.json` of each fold, with 2 decimals. Thus the laptop loads the checkpoints correctly.
- Gray images lower the mean macro F1 by 3.58 points and the mean accuracy by 3.59 points.
- The change differs by fold, from 1.60 to 5.52 points of macro F1.
- Thus B0 uses color, but most of its score stays with no color.

## Not checked

- The change of each class and of each driver.
- The score on an infrared camera. A gray copy of a color image is not an infrared image.
- The code of this measure is not in git. `b0_gray.json` names the last commit, 6767bcc, but the code of the measure came after it.
