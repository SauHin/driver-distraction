# Errors of DeiT-III Small with and without the flip

Date: 2026-11-17.
Sources: the files `_gray.json`, `_val_confusion_color.csv`, `_val_classes.csv`, and `_val_drivers.csv` of `e1_deit3_small` and `e3_deit3_small_noflip`.
`scripts/score_gray.py` wrote these files on the laptop GPU. No test driver is read.
"Paper set" is `e1_deit3_small`. "No flip" is `e3_deit3_small_noflip`, the winner of E3 (decision P16).
The tables hold all 18 development drivers and 15899 images. Each score is the value of the file times 100, with 2 decimals.

## Result

- No flip gives fewer errors: 1614 against 1809. Its mean macro F1 is 1.04 points higher. See `e3.md`.
- But 9 drivers gain and 9 drivers lose with no flip. Two drivers give most of the gain: p072 and p081.
- No flip misses more distracted images: 637 against 438. Thus its recall of the status "distracted" is lower.
- Neither model confuses the left hand and the right hand.

## Status: safe or distracted

| Value | Precision | Recall | F1 | Missed distracted images | False alarms |
|---|---|---|---|---|---|
| Paper set. | 98.00. | 96.90. | 97.45. | 438. | 280. |
| No flip. | 98.85. | 95.49. | 97.14. | 637. | 157. |

- A missed distracted image has the prediction "safe". A false alarm is a safe image with a distracted prediction.
- Talking to the passenger gives most of the missed images: 382 with the paper set and 482 with no flip.
- Thus no flip gives fewer false alarms, but more missed alerts.

## Largest confusions

| Value | `c9` as `c0` | `c0` and `c9` both ways | Other classes as `c8` |
|---|---|---|---|
| Paper set. | 382. | 446. | 690. |
| No flip. | 482. | 587. | 446. |

- With no flip, 30.84 percent of the `c9` images have the prediction `c0`. With the paper set, the share is 24.44 percent.
- The left and right hand pairs are `c1` with `c3`, and `c2` with `c4`.
- These pairs give 2 errors with the paper set and 0 errors with no flip.

## F1 of each class

| Class | Paper set | No flip |
|---|---|---|
| `c0`, safe driving. | 80.54. | 80.21. |
| `c1`, texting, right hand. | 93.93. | 96.44. |
| `c2`, phone call, right hand. | 96.41. | 95.57. |
| `c3`, texting, left hand. | 97.06. | 97.01. |
| `c4`, phone call, left hand. | 97.94. | 97.18. |
| `c5`, radio. | 90.95. | 95.83. |
| `c6`, drinking. | 92.36. | 93.81. |
| `c7`, reaching behind. | 94.03. | 95.42. |
| `c8`, hair and makeup. | 73.48. | 78.67. |
| `c9`, talking to passenger. | 66.53. | 65.54. |

## Drivers

- With the paper set, the median driver accuracy is 91.23. With no flip, it is 88.52.
- 4 drivers are below 85 with the paper set, and 2 with no flip.
- p072 is the lowest driver of both: 36.71 with the paper set and 60.40 with no flip.
- No flip raises p072 by 23.70 points and p081 by 14.82 points. It lowers p039 by 5.07 points and p047 by 4.79 points.

## Gray scale

- In gray scale, the mean macro F1 falls by 1.30 points with the paper set and by 1.26 points with no flip.
- B0 lost 3.58 points. Thus these two models use color less than B0. See `b0_gray.md`.

## Not checked

- The AI looked at no image. Thus the cause of the gain for p072 and p081 is not known.
- The effect of the smoothing over time on the missed alerts.
