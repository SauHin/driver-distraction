# Final model on changed test images

Date: 2026-11-17.
Source: `final_deit3_small_noflip_test_changes.json` in this folder, written by `scripts/evaluate_changes.py`.
This is step 5 of `docs/PLAN.md` section 9 (decision D20). It opened the test lock one time, with one row in `results/test_access_log.csv`.
The strengths come from decision P20. No design choice uses these scores.
Each score is the value of the file times 100, with 2 decimals. All 8 test drivers, 6525 images.

## Scores

The status recall is the recall of the status "distracted".

| Change | Strength | Accuracy | Macro F1 | Status recall |
|---|---|---|---|---|
| None. | No value. | 91.97. | 91.30. | 97.19. |
| Brightness. | 0.6. | 91.68. | 90.86. | 97.36. |
| Brightness. | 0.4. | 91.54. | 90.65. | 97.40. |
| Brightness. | 0.2. | 90.74. | 89.64. | 97.14. |
| Contrast. | 0.6. | 92.03. | 91.41. | 97.31. |
| Contrast. | 0.4. | 91.56. | 90.98. | 97.29. |
| Contrast. | 0.2. | 90.84. | 90.33. | 97.29. |
| Blur. | 1 pixel. | 91.62. | 91.04. | 97.26. |
| Blur. | 2 pixels. | 90.99. | 90.49. | 97.43. |
| Blur. | 4 pixels. | 91.46. | 90.92. | 96.95. |
| Noise. | 10. | 91.57. | 90.89. | 97.31. |
| Noise. | 20. | 91.33. | 90.63. | 97.28. |
| Noise. | 40. | 90.77. | 90.05. | 97.09. |
| Cover. | 10 percent. | 81.61. | 81.46. | 94.26. |
| Cover. | 20 percent. | 63.40. | 63.67. | 89.62. |
| Cover. | 30 percent. | 47.95. | 47.24. | 88.18. |

## Result

- The scores with no change are equal to the scores of the Colab evaluation in `scores.md`. Thus the laptop gives the same model output.
- Brightness, contrast, blur, and noise lower the macro F1 by 1.66 points or less at all strengths.
- A cover lowers the scores much more. At 30 percent, the macro F1 is 47.24.
- The status recall stays at 88.18 or more with each change. Thus the status changes less than the 10 classes.
- For the live use, the camera must see the hands and the face with no object in the way.

## Not checked

- The place of each cover. A cover over the hands can have a larger effect than a cover over the seat.
- Other changes, for example a camera angle, a night camera, or a sun glare.
- The run used commit 051f230 with the step 5 code not yet in git. The commit after it holds this code.
