# Scores of R1 and the paper

Date: 2026-11-15.
Sources: `r1_vit_s42_test_metrics.json` in this folder, and table II of arXiv 2209.09178.
The scores come from the last epoch of the run `r1_vit_s42`, epoch 19, with seed 42.
The test part has the 8 test drivers of the public split list, with 6525 images.
Each value in this file is the value of the result file times 100, with 2 decimals.

## Gate G1

| Score | Paper | Pass at | R1 | R1 minus the pass value | Result |
|---|---|---|---|---|---|
| Accuracy. | 90.36. | 85.36. | 89.46. | 4.10. | Pass. |
| Class-mean accuracy. | 90.36. | 85.36. | 89.16. | 3.80. | Pass. |

- The paper gives one accuracy for the standard ViT. Section 5 of `docs/PLAN.md` compares both scores with it.
- Both scores reach the pass value. Thus R1 passes, and gate G1 passes.
- Each score is more than 1.00 point above the pass value. Thus the seeds 43 and 44 do not run.
- The accuracy of R1 is 0.90 points below the paper. The class-mean accuracy of R1 is 1.20 points below the paper.

## Other scores of R1

The paper gives only the accuracy per image for the standard ViT.

| Score | R1 |
|---|---|
| Macro F1. | 88.93. |
| Precision of the status "distracted". | 96.50. |
| Recall of the status "distracted". | 97.38. |
| F1 of the status "distracted". | 96.94. |

## Scores of each test driver

`r1_vit_s42_test_drivers.csv` holds all scores of each driver.

| Driver | Images | Accuracy | Class-mean accuracy |
|---|---|---|---|
| p002. | 725. | 84.28. | 81.56. |
| p012. | 823. | 85.78. | 83.15. |
| p026. | 1196. | 91.47. | 91.67. |
| p035. | 848. | 86.08. | 86.76. |
| p041. | 605. | 96.53. | 96.45. |
| p045. | 724. | 95.99. | 96.12. |
| p050. | 790. | 86.84. | 85.02. |
| p075. | 814. | 89.80. | 89.97. |

## Confusion matrix

`r1_vit_s42_test_confusion.csv` holds the matrix. The rows are the true classes, and the columns are the predictions.

- 139 images of `c0` have the prediction `c9`. 119 images of `c9` have the prediction `c0`.
- 291 images of other classes have the prediction `c8`.
- Phase 2 does not use these test numbers for a design choice. Section 8 of the plan uses the validation folds.

## Run

- Training: 20 epochs on the 18 development drivers, with 15872 images in each epoch.
- The sum of `epoch_seconds` in `r1_vit_s42/history.csv` is 3739.70 seconds. The sum does not include the start of the run and the checkpoint saves.
- GPU: NVIDIA L4. `r1_vit_s42/environment.json` gives the library versions and the git commit.
- `r1_vit_s42/recipe_differences.md` lists the differences from the ViT-DD recipe.
- `results/test_access_log.csv` holds one row. This row is the first unlock of the test lock.
