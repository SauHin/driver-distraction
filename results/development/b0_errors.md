# Errors of B0 on the validation folds

Date: 2026-11-16.
Sources: `b0_val_confusion_color.csv`, `b0_val_confusion_gray.csv`, `b0_val_classes.csv`, and `b0_val_drivers.csv` in this folder.
`scripts/score_gray.py` wrote these files on the laptop GPU. No test driver is read.
The model of each fold scores the 6 validation drivers of that fold.
Thus the tables hold all 18 development drivers and 15899 images. No model scores a driver of its own training part.
Each score is the value of the file times 100, with 2 decimals.

## Result

- Two groups give most of the 1858 errors. Safe driving against talking to the passenger gives 31.86 percent of the errors.
- Images of other classes with the prediction "hair and makeup" give 37.57 percent of the errors.
- The left hand and the right hand give almost no errors.
- The status (safe or distracted) has a precision of 97.52, a recall of 97.26, and an F1 of 97.39.
- The accuracy of the drivers is from 60.98 to 96.04. 7 of the 18 drivers have an accuracy below 85.

## Largest confusions

The share is the share of the images of the true class.

| True class | Prediction | Images | Share |
|---|---|---|---|
| `c9`, talking to passenger. | `c0`, safe driving. | 327. | 20.92. |
| `c0`, safe driving. | `c9`, talking to passenger. | 265. | 15.01. |
| `c6`, drinking. | `c8`, hair and makeup. | 190. | 11.52. |
| `c9`, talking to passenger. | `c8`, hair and makeup. | 150. | 9.60. |
| `c9`, talking to passenger. | `c5`, radio. | 119. | 7.61. |
| `c4`, phone call, left hand. | `c8`, hair and makeup. | 114. | 7.02. |
| `c7`, reaching behind. | `c8`, hair and makeup. | 90. | 6.51. |
| `c2`, phone call, right hand. | `c8`, hair and makeup. | 72. | 4.37. |
| `c8`, hair and makeup. | `c2`, phone call, right hand. | 55. | 3.98. |
| `c8`, hair and makeup. | `c6`, drinking. | 44. | 3.18. |

## Scores of each class

The gray F1 comes from the same images in gray scale. See `b0_gray.md`.

| Class | Images | Precision | Recall | F1 | Gray F1 |
|---|---|---|---|---|---|
| `c0`, safe driving. | 1766. | 78.54. | 80.18. | 79.35. | 75.04. |
| `c1`, texting, right hand. | 1623. | 98.03. | 95.26. | 96.62. | 94.12. |
| `c2`, phone call, right hand. | 1649. | 96.18. | 94.60. | 95.38. | 91.32. |
| `c3`, texting, left hand. | 1646. | 96.87. | 97.63. | 97.25. | 96.13. |
| `c4`, phone call, left hand. | 1623. | 99.06. | 90.82. | 94.76. | 91.51. |
| `c5`, radio. | 1614. | 90.34. | 99.07. | 94.50. | 92.71. |
| `c6`, drinking. | 1649. | 95.65. | 86.66. | 90.93. | 90.26. |
| `c7`, reaching behind. | 1383. | 97.32. | 91.97. | 94.57. | 91.83. |
| `c8`, hair and makeup. | 1383. | 63.44. | 87.56. | 73.57. | 64.68. |
| `c9`, talking to passenger. | 1563. | 72.99. | 59.31. | 65.44. | 60.04. |

## Patterns

### Safe driving and talking to the passenger

- 327 images of `c9` have the prediction `c0`. 265 images of `c0` have the prediction `c9`.
- These 592 images are 31.86 percent of all errors.
- A probable cause: both classes show the hands on the wheel. The classes differ mainly in the direction of the head.
- 387 distracted images have the prediction "safe". 327 of them are `c9`, thus 84.50 percent.
- 350 safe images have a distracted prediction. 265 of them have the prediction `c9`.

### Images with the prediction "hair and makeup"

- 698 images of other classes have the prediction `c8`. Thus `c8` has the lowest precision, 63.44.
- The largest sources are drinking, talking to the passenger, the phone call with the left hand, and reaching behind.
- A probable cause: `c8` shows a hand near the face or the head. Drinking and phone calls show this too.
- In gray scale, 1027 images of other classes have the prediction `c8`. The F1 of `c8` falls by 8.89 points, the largest fall of all classes.

### The left hand and the right hand

- `c1` and `c3` give 4 and 0 errors. `c2` and `c4` give 0 and 1 errors.
- Texting and the phone call with the same hand give 0 errors on the right side and 2 errors on the left side.
- The training flips images with no change of the label. But these numbers show almost no confusion of the left hand and the right hand.
- Experiment E3 tests the training with no flip.

### Drivers

- The median accuracy of the 18 drivers is 88.21.
- These 7 drivers have an accuracy below 85: p072 (60.98), p081 (77.16), p015 (79.20), p064 (80.98), p014 (83.68), p066 (83.95), and p042 (84.26).
- Each fold has at least one of these drivers.
- p072 has the fewest images, 346.
- In gray scale, the accuracy falls most for p014 (12.33 points), p072 (9.25 points), and p051 (7.17 points).
- `b0_val_drivers.csv` holds all scores of each driver.

### Status in gray scale

- In gray scale, the status has a precision of 96.54, a recall of 97.51, and an F1 of 97.02.
- The safe images with a distracted prediction increase from 350 to 494.

## Not checked

- The AI looked at no image. Each cause above is a probable cause only.
- Section 8.4 of the plan asks for example images. Dataset images do not go in git or in the report. Thus examples for the report must come from own recordings.
- The scores of each class for each driver.
