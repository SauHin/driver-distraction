# Plan: driver distraction detection

Written on 2026-11-14 by Claude (Fable 5.2) in the planning session. The owner reviews it.
Sources: `docs/starter/BRIEF.md`, `docs/starter/guideline.docx`, and `docs/LITERATURE.md`.
Each result in this plan comes from a result file in `results/`. Each open number is `TBD`.

## 1. Goal

Build and evaluate a deep learning system that detects driver distraction with an RGB camera, live.
The course is COMP6826001 Deep Learning. The guideline is the authority on the deliverables.

The two deliverables are:

- A working application that reads a camera or a video and shows the result to the user.
- A technical report with the 10 sections of the guideline.

The research questions are:

- RQ1: Does the public ViT-DD recipe reach its reported accuracy on drivers that the model did not see?
- RQ2: How much accuracy does a small model lose, and how much speed does it gain on a laptop?
- RQ3: Which training choices change the accuracy on new drivers?
- RQ4: How does the model behave on own recordings and on changed images?
- RQ5: Is the best model of the test also the right model for the live application?

## 2. Decisions

### 2.1 Decisions of the owner

| ID | Date | Decision | Reason |
|---|---|---|---|
| D1 | 2026-11-13 | DMD is rejected as the main dataset. | The public release has 14 drivers. The owner says that this is too few. |
| D2 | 2026-11-13 | The dataset must have a small size. The system must run live. | Rule of the owner. |
| D3 | 2026-11-13 | The replication passes at 5 points or less below the paper score. | Choice of the owner among 4 options. |
| D4 | 2026-11-13 | The deadline is 3 to 4 weeks after 2026-11-13. The exact date is TBD. | Answer of the owner. |
| D5 | 2026-11-14 | The project starts on StateFarm. The owner requests AUC v2 on the same day. | Choice of the owner among 4 options. |
| D6 | 2026-11-14 | AUC v2 is the test on a second dataset. Void after D7. | Part of D5. |
| D7 | 2026-11-15 | AUC v2 is cancelled. The experiments use StateFarm only. | Choice of the owner. |
| D8 | 2026-11-15 | The development folds come from split v2. Split v1 is void for development. | Choice of the owner among 3 options. The owner approved the file on the same day. Section 4 gives the reason. |
| D9 | 2026-11-15 | The replication runs R1 only. R2 and the emotion experiment E6 are void. | Choice of the owner among 3 options. The face crops and emotion labels of ViT-DD do not download. |
| D10 | 2026-11-15 | R1 uses the ViT-DD config with no face input and no emotion head. | Choice of the owner among 3 options. The paper gives no recipe for R1. See `results/vitdd_recipe_check.md`. |
| D11 | 2026-11-15 | R1 steps the scheduler after each epoch, as ViT-DD does with Lightning 1.7.7. Config key `sched_after_epoch`. | Choice of the owner. The first code stepped it at the start of each epoch. See `results/vitdd_recipe_check.md`. |
| D12 | 2026-11-15 | R1 gives the position embedding the weight decay 0.1, as `ViTDDLM` does. Config key `wd_skip_list`. | Choice of the owner. The first R1 run gave it no weight decay. The owner stopped that run after epoch 10. |
| D13 | 2026-11-15 | The training runs of Phase 2 use the Colab L4, one run at a time. | Choice of the owner among 3 options. All runs have the same GPU and batch size, so their training times compare. |
| D14 | 2026-11-15 | While a training job runs on Colab, the AI runs a short check cell at least every 30 minutes. | Choice of the owner. The runtime of `b0_f1` stopped after epoch 10 with no error in the logs. Colab can stop an idle runtime. |
| D15 | 2026-11-16 | E2 uses DeiT-III Small. | Choice of the owner among 2 options. It has the highest E1 macro F1 and is the live candidate. See `results/development/e1.md`. |
| D16 | 2026-11-16 | E3 uses DeiT-III Small. | Choice of the owner among 2 options. The plan named no model for E3. |
| D17 | 2026-11-16 | E3 has no value with more gray images. | Choice of the owner among 2 options. B0 lost 3.58 points of macro F1 on gray validation images. |
| D18 | 2026-11-16 | E2 adds MobileNetV3-Large with random weights. Its ImageNet value is the E1 run. | Choice of the owner among 2 options. DeiT-III Small with random weights did not learn. See `results/development/e2_checks.md`. |

The owner also selected 9 distraction classes for DMD on 2026-11-13. This choice is void because of D1.

### 2.2 Decisions of the planning session

The owner can change each of these decisions.

| ID | Decision | Reason |
|---|---|---|
| P1 | The replication target is ViT-DD (Ma and Wang 2022) on StateFarm. | It has the highest score by driver with public code, config files, and split lists. |
| P2 | The model has the 10 classes of the dataset. | The paper uses them. |
| P3 | The product status has two values. Class `c0` is safe. Classes `c1` to `c9` are distracted. | The dataset defines these 9 classes as distraction. |
| P4 | The test drivers are the 8 drivers of the public split list. | This makes the score comparable with the paper. |
| P5 | Development uses 3 folds of the 18 training drivers. | Each training driver is a validation driver one time. |
| P6 | The project has its own training code with `timm` and PyTorch. | The public code needs old library versions. Own code serves all experiments. |
| P7 | The application is a desktop program with an OpenCV window. | It is the smallest program that runs in real time on a laptop. |
| P8 | The live speed target is 5 processed frames per second or more on the laptop. | Smoothing needs some results per second. |
| P9 | The gate reads the last epoch of the run with seed 42. | The public config uses seed 42. No selection on test drivers occurs. |
| P10 | The manifests and the public ViT-DD image lists stay in `data/`, not in git. | They hold the image names of the dataset. |
| P11 | The split file holds the 3 folds with the train and the validation drivers of each fold. | A reader sees each part with no code. A test keeps the lists equal to the output of the split method. |
| P12 | A development run gives the scores of its last epoch on its validation fold. | Phase 3 has no validation part to select an epoch. Decision P9 reads the last epoch too. |
| P13 | Values that section 8.2 does not name come from the R1 config. | Each experiment then changes one variable from B0. |
| P14 | The E1 models start from the default ImageNet-1k weights of `timm` 1.0.31. DeiT-III Base starts from the weight file of R1. | B0 and the DeiT-III Base of E1 then have the same start weights. |
| P15 | The E3 value "None" has no random step. The random crop becomes a center crop, as in the evaluation. | The paper set lists the random crop as a step of the augmentation. See `results/vitdd_recipe_check.md`. |
| P16 | The E3 value with the highest mean validation macro F1 wins E3. The paper set with no flip wins. | Section 8.5 selects models by the same score. See `results/development/e3.md`. |

## 3. Data

### 3.1 StateFarm (first dataset)

- 22,424 labeled images of 26 drivers. The size is 640 by 480 pixels, RGB.
- One camera at the side of the driver. A truck pulled the car during the recording.
- The file `driver_imgs_list.csv` gives the driver and the class of each image.
- The competition has more images with no label. This project does not use them.
- Access: Kaggle account and the competition rules.
- The rules limit the data to the competition. The owner asks the lecturer about this point.

### 3.2 AUC v2 (cancelled)

- Decision D7 cancels AUC v2. No part of this plan uses it.

### 3.3 Own recordings (third data source)

- Purpose: evaluation in real use and the test of the live application.
- Place: a parked car only. The camera is at the side of the driver, as in the datasets.
- Each person does the 10 classes as short clips. Each person also does one continuous sequence.
- A label file gives the start and the end of each action.
- Each person signs a consent form. The files stay private.
- No person is in two parts of a split.

### 3.4 Classes

The own recordings use these class names. Their labels map to the StateFarm codes by name.

| Name | StateFarm | Status |
|---|---|---|
| Safe driving. | `c0`. | Safe. |
| Texting, right hand. | `c1`. | Distracted. |
| Phone call, right hand. | `c2`. | Distracted. |
| Texting, left hand. | `c3`. | Distracted. |
| Phone call, left hand. | `c4`. | Distracted. |
| Radio. | `c5`. | Distracted. |
| Drinking. | `c6`. | Distracted. |
| Reaching behind. | `c7`. | Distracted. |
| Hair and makeup. | `c8`. | Distracted. |
| Talking to passenger. | `c9`. | Distracted. |

### 3.5 Data preparation

- Audit: count the images of each driver and class. List missing files, image sizes, and exact copies.
- Resize: 224 by 224 pixels for the models of this plan.
- Scale: the mean and the standard deviation of the pretrained weights. These are constants.
- Augmentation: only on training images. Section 8 gives the values.
- Storage: raw files in `data/raw/`. A manifest holds the SHA-256 hash of each file.
- Colab: the data lives on Google Drive. A run copies it to the local disk first.

## 4. Splits and the test lock

- The driver of each image comes from `driver_imgs_list.csv`.
- Test part: drivers p002, p012, p026, p035, p041, p045, p050, p075. It has 6,525 images.
- Development part: the other 18 drivers. It has 15,899 images.
- These two parts are the public lists `val_list_d_01.csv` and `train_list_d_01.csv` of the ViT-DD repository.
- Folds: 3 folds with 6 development drivers each. Decision D8 selects split v2.
- The file `splits/statefarm_v2.json` holds the driver numbers of each part. It holds no image.
- The replication and the final evaluation use the development part and the test part only. The folds do not change them.
- The owner approved `splits/statefarm_v2.json` on 2026-11-15.

Split v1 and split v2 have the same test part. Only the folds are different.

- Split v1: a shuffle with seed 2026 makes the folds.
- A check of split v1 found two scene groups (a car or a camera position) in the drivers. See `results/data_audit/statefarm_split_check.json`.
- In split v1, the validation part of fold 0 holds 5 of the 6 development drivers of the first scene group.
- Split v2: each fold has 2 development drivers of the first scene group and 4 of the second.
- Of these assignments, split v2 takes the one with the smallest range of validation image counts. A full search finds it, with no seed.
- `results/data_audit/statefarm_v2_split_check.json` holds the check of split v2.
- Limit: the test drivers share their scene groups with the development drivers. The test score measures new persons in known cars.

The test lock has these rules:

- Only two points score the test drivers: the replication (section 7) and the final evaluation (section 9).
- The data loader refuses the test part unless a run sets the unlock flag.
- Each unlock writes a row to `results/test_access_log.csv`. The row holds the time, the git commit, and the reason.
- No design choice reads a score of the test drivers.
- The own test persons follow the same lock. Only the final evaluation scores them.

Tests in `tests/` keep these rules:

- No driver is in two parts.
- The test part is equal to the public list.
- The 3 folds divide the development part with no overlap.
- The folds of split v2 are equal to the output of the search.
- The class map gives each StateFarm code one name and one status.

## 5. Metrics and targets

The metrics are fixed before the first run.

- Accuracy: the share of images with the correct class.
- Class-mean accuracy: the mean of the accuracy of each class.
- Macro F1: the mean of the F1 scores of the 10 classes.
- Binary scores: precision, recall, and F1 of the status "distracted".
- Live scores on continuous video: false alarms per minute of safe driving, and the delay of an alert.

The replication targets follow decision D3:

| Run | Paper score | Pass at |
|---|---|---|
| R2. ViT-DD on StateFarm. Void after decision D9. | 92.51. | 87.51 or more. |
| R1. Standard ViT on StateFarm. | 90.36. | 85.36 or more. |

Rules for the gate G1:

- Decision D9: G1 reads R1 only. R2 does not run.
- If R1 passes, G1 passes. The accuracy and the class-mean accuracy must each reach the threshold.
- The reason for two scores: the paper text and the public code define the accuracy differently.
- If a score is less than 1.00 point from its threshold, run seeds 43 and 44 too. Then G1 reads the mean of the 3 seeds.

Other rules:

- Development compares models by the mean macro F1 of the 3 validation folds.
- The final report gives all metrics, the score of each test driver, and a bootstrap interval over drivers.
- The brief names a wish of about 0.70. The thresholds of G1 are above it.

## 6. Phase 0: setup

Tasks of the owner:

- O1. Ask the lecturer about the topic change, the StateFarm rules, the user test method, the exact deadline, and the report language.
- O2. Sign in to Kaggle and accept the rules. Download `imgs.zip` and `driver_imgs_list.csv` to `data/raw/statefarm/`.
- O3. Void after decision D7.
- O4. Review this plan, `docs/LITERATURE.md`, and `docs/AI_LOG.md`.
- O5. Find the persons for the own recordings and for the user test.

Tasks of Claude:

- C1. Start git. Add `data/`, `runs/`, and weight files to `.gitignore`. Make the package `ddd` and the test setup.
- C2. Write the audit script. It writes `results/data_audit/statefarm.json`.
- C3. Write the split file and the tests of section 4.
- C4. Write the manifest script and the Colab notebook.
- C5. Download the public lists, config files, weights, and face crops of ViT-DD. Record what downloads.
- C6. Measure the first run on Colab and on the laptop. Write `results/compute/first_run.json`.
- C6 status on 2026-11-15: the laptop part and the Colab part are in `results/compute/first_run.json`.
- The Colab part is one full epoch of R1 at batch 128 on an NVIDIA L4. The GPU log is `results/compute/c6_colab_gpu.csv`.
- On 2026-11-15, the Colab page gave about 1.94 compute units per hour for the L4 runtime.
- Source: the key `colab_compute_units` of `results/compute/first_run.json`. The owner read the page.
- On the laptop, the R1 config stops with CUDA out of memory at batch 128. R1 runs on Colab only.

Phase 0 is complete after these points:

- All tests pass.
- The audit file exists and the owner read it. The owner read it on 2026-11-15.
- The owner approved the split file.

## 7. Phase 1: replication and gate G1

The recipe comes from the public config `vitdd_sfddd_sbd.yaml`:

- Model: `deit3_base_patch16_224` from `timm`, with ImageNet weights.
- Input: the driver image at 224 pixels. ViT-DD adds a face crop at 32 pixels.
- Trainable parts: each parameter with `.attn.` in its name, the head, and the position embedding.
- Training: 20 epochs, batch size 128, AdamW, weight decay 0.1, learning rate 0.0003.
- Schedule: 5 warm-up epochs from 0.000001, then a cosine curve to 0.
- Other values: no label smoothing, no Mixup, color jitter 0.3, the 3-Augment set, seed 42, 32-bit numbers.
- Decision D10: R1 uses this recipe with no face input and no emotion head.
- ViT-DD loss: weight 0.999 for the driver task and 0.001 for the emotion task.

Steps:

- R0. Score the public ViT-DD weights on the test drivers with the evaluation code of this project. R0 cannot run because the weights do not download.
- R1. Train the standard ViT on the 18 development drivers. Score the last epoch on the test drivers.
- R2. Train ViT-DD with the face crops and emotion labels of the authors. Void after decision D9.
- R1 is the first unlock of the test lock.

Points to check before R1:

- The `timm` name of the model gives different default weights in old and new versions. Use the weights of the 2022 code.
- The paper text gives batch size 256. The public config gives 128. This plan uses the config.
- Read the public data code and list each augmentation step. Find out whether the code flips images. Done in `results/vitdd_recipe_check.md`.
- Record each difference from the recipe in the run folder. Done in `results/replication/r1_vit_s42/recipe_differences.md`.

Outputs in `results/replication/`:

- One metrics file for each run.
- One table with the paper scores and the scores of this project.
- The confusion matrix and the score of each test driver.

Gate G1:

- If G1 passes, start Phase 2.
- If G1 fails, stop. Report the numbers to the owner. The owner decides the next step.

Status of R1 and G1 on 2026-11-15:

- R1 ran on Colab with an NVIDIA L4, at commit `f9c64dd`. The run folder is `runs/r1_vit_s42` on Drive.
- Decisions D11 and D12 changed the training code before the run. A first run stopped after epoch 10. No test driver was scored with it.
- The evaluation scored epoch 19 on the test drivers one time. `results/test_access_log.csv` holds the row of this unlock.
- R1 has the accuracy 89.46 and the class-mean accuracy 89.16. Source: `results/replication/r1_vit_s42_test_metrics.json`.
- Both scores are more than 1.00 point above 85.36. Thus the seeds 43 and 44 do not run.
- G1 passes. Phase 2 starts after a yes of the owner. `results/replication/scores.md` holds the table of scores.

## 8. Phase 2: development

Phase 2 starts only after G1 passes. The runs of Phase 2 train on the 18 development drivers only.

### 8.1 Gaps of the paper

| Gap | Fact | Action in this plan |
|---|---|---|
| 1. | The config has no validation part. It reads the test list as the validation list. | Select with the 3 folds only. |
| 2. | The paper has one split with 8 test drivers and no spread. | Give the score of each driver and a bootstrap interval. |
| 3. | The accuracy definition is not clear. | Report the accuracy and the class-mean accuracy. |
| 4. | The paper names a horizontal flip. Four classes depend on the left or right hand. | Experiment E3. |
| 5. | Each model is scored on its own dataset only. | Test on own recordings. |
| 6. | The paper has no test on video and no smoothing. | Experiment E8 on own continuous video. |
| 7. | The model is large. The paper reports 15.56 frames per second on an NVIDIA 3090 Ti. | Experiment E1 with small models. Measure the speed on the laptop. |
| 8. | ViT-DD needs a face crop for each frame at run time. | Out of scope after decision D9. |
| 9. | The data is staged and has one camera position. | Own recordings and a guide for the camera position. |

### 8.2 Experiments

Each experiment changes one variable. Each value runs on the 3 folds. E8 is the exception.
E8 reads own video of the persons that are not test persons (section 9).
The guideline asks for a baseline and 2 more scenarios. The "Must" rows give a baseline and 3 scenarios.

| ID | Question | Variable | Values | Priority |
|---|---|---|---|---|
| B0 | What is the baseline? | None. | DeiT-III Base with the recipe of section 7. | Must. |
| E1 | Which model size fits live use? | Architecture. | MobileNetV3-Large, EfficientNet-B0, ResNet-50, DeiT-III Small, DeiT-III Base. | Must. |
| E2 | Does pretraining help on new drivers? | Start weights. | Random weights and ImageNet weights, for the best small model of E1. | Must. |
| E3 | Does augmentation help? Does the flip hurt? | Augmentation. | None. The paper set. The paper set with no flip. | Must. |
| E6 | Does the emotion task help? | Second task. | R1 and R2 of Phase 1. No new run. | Void after decision D9. |
| E4 | How sensitive is training to the learning rate? | Learning rate. | The default, one third of it, and 3 times it. | Next. |
| E5 | How many drivers does training need? | Number of train drivers. | 4, 8, and 12 drivers. | Next. |
| E8 | Does smoothing help on video? | Smoothing window. | None. A majority vote. A moving mean. | Next. |
| E7 | Which layers must train? | Trainable layers. | A linear head only. Attention layers only. All layers. | Last. |

Shared recipe of the development runs:

- ImageNet weights from `timm`. All layers train. B0 and E7 are the exceptions.
- AdamW, 5 warm-up epochs, a cosine curve, 20 epochs, label smoothing 0.1, seed 42.
- Learning rate 0.0003 for the ViT models and 0.001 for the CNN models.
- The augmentation set that wins E3. Before E3, the paper set.
- Decision P13: batch size 128, weight decay 0.1, a warm-up from 0.000001, the scheduler step of D11, no drop path, 32-bit numbers.
- The configs are `configs/b0.json` and `configs/e1_*.json`. The queue file `configs/queue_phase2.txt` gives the order of the runs.
- E2 and E3 change one key of `configs/e1_deit3_small.json` each. Their configs are `configs/e2_*.json` and `configs/e3_*.json`.
- The ImageNet value of E2 and the paper-set value of E3 are the runs of `e1_deit3_small`.
- Decision D18 adds `configs/e2_mobilenetv3_large_random.json`. Its ImageNet value is the runs of `e1_mobilenetv3_large`.
- The config key `init_values` sets the start value of the layer scales of a ViT. Only the checks of `results/development/e2_checks.md` use it.
- `scripts/summarize_folds.py` writes the scores of each fold and their mean to `results/development/`.
- `scripts/score_gray.py` scores the fold runs on gray validation images. It measures the use of color by a model.
- `scripts/score_gray.py` writes the confusion matrix, the scores of each class, and the scores of each driver for section 8.4.
- `scripts/measure_speed.py` measures the processed frames per second for decision P8, on the laptop GPU and CPU.
- For P8, a frame is a 640x480 image through `eval_transform` and the model, one frame at a time. The camera read is not part of it.

### 8.3 Training analysis

Each run writes these values for each epoch: training loss, validation loss, training accuracy, validation accuracy, and validation macro F1.
The training accuracy counts the augmented images during the epoch. The validation loss uses the loss of the training, with its label smoothing.

The analysis covers these points of the guideline:

- Curves: fit too close to the training data, fit too weak, unstable training, and convergence.
- Sensitivity to the learning rate (E4) and to the number of drivers (E5).
- Model comparison: accuracy, macro F1, training time, parameters, file size, and speed.
- The trade-off between accuracy, size, and speed.

### 8.4 Error analysis

- The confusion matrix of the validation folds.
- The F1 score of each class and the accuracy of each driver.
- The largest confusions, with an explanation and example images.
- A search for patterns: left against right hand, safe driving against talking, and drivers with low scores.

### 8.5 Model choice

Phase 2 ends with two models:

- `best`: the model with the highest mean macro F1 on the validation folds.
- `live`: the model with the highest mean macro F1 among the models that reach the speed target P8.

The owner approves the two models before Phase 3.

## 9. Phase 3: final evaluation

This phase is the second unlock of the test lock. It has no model selection.

1. Train `best` and `live` on all 18 development drivers with the selected config.
2. Score them on the 8 test drivers. Give all metrics of section 5.
3. Void after decision D7.
4. Score them on the own test persons, on the short clips and on the continuous video.
5. Change the test images and score again: brightness, contrast, blur, noise, and a covered area. Use 3 strengths.
6. Measure the cost: parameters, file size, memory, and speed on the laptop GPU and CPU.
7. Compare `best` and `live`. Say which model fits the application, and why.

Rule for the own recordings:

- With 6 persons or more, half of the persons select the smoothing window. The other half is the test.
- With fewer persons, the smoothing window is fixed before the test. All persons are the test.

Outputs in `results/final/`: the metrics files, the tables, and the figures for the report.

## 10. Phase 4: application and user test

The application does these steps:

1. Read a frame from a camera, a video file, or an image file.
2. Resize and scale the frame in the same way as the evaluation code.
3. Run the `live` model.
4. Smooth the class scores over time.
5. Show the class, the status (safe or distracted), and the class scores on the screen.
6. Give an alert after a continuous distraction of a set length.
7. Write each result to a log file.

Rules for the application:

- A test compares its preparation of a frame with the evaluation code.
- A guide tells the user where to put the camera.
- The application is a demonstration. It is not a safety system for the road.

User test:

- Persons from the target group use the application in a parked car.
- Each person does a fixed list of actions. A form records the result of each action.
- A short question form records how easy and how useful the application is.
- All answers are real. No answer is made up.
- If a direct user test is not possible, the owner asks the lecturer to approve a different method.

## 11. Phase 5: report

| Report section | Source in this project |
|---|---|
| 1. Introduction. | Section 1 of this plan and the brief. |
| 2. Dataset and data preparation. | The audit files, section 3, and section 4. |
| 3. Deep learning fundamentals and model. | The recipe of section 7 and the model notes of each run. |
| 4. System design and implementation. | The application of section 10 and the package `ddd`. |
| 5. Experimental methodology. | Sections 5 and 8. The research questions of section 1. |
| 6. Results. | `results/replication/`, the fold results, and `results/final/`. |
| 7. Analysis and discussion. | Sections 8.3 and 8.4, and step 7 of section 9. |
| 8. Real-world evaluation. | The own recordings, the user test, and the cost numbers. |
| 9. Conclusion. | The answers to the 5 research questions. |
| 10. AI usage declaration. | `docs/AI_LOG.md`. |

Rules for the report:

- Each number comes from a result file.
- Each claim names its result file.
- Figures with persons come from own recordings with consent.

## 12. Schedule

The exact deadline is TBD. Week 4 depends on the deadline.

| Week | Dates | Work | Complete after |
|---|---|---|---|
| 1 | 2026-11-14 to 2026-11-20. | Phase 0 and Phase 1. | Gate G1 has a result. |
| 2 | 2026-11-21 to 2026-11-27. | The "Must" rows of section 8.2. A first version of the application. The own recordings. | The models `best` and `live` are selected. |
| 3 | 2026-11-28 to 2026-12-04. | Phase 3. The user test. The report draft. The "Next" rows that fit. | All "Must" results are in the report draft. |
| 4 | 2026-12-05 to 2026-12-11. | The final report. A demonstration video. The "Last" row. | The owner submits. |

Cut rules:

- The "Must" rows, Phase 3, the application, and the report come first.
- If week 2 ends late, drop rows of section 8.2 in this order: E7, E5, E4, E8.

## 13. Compute

- The laptop has an NVIDIA RTX 3050 with 6 GB. It runs the tests, the data work, the speed tests, and the application.
- Decision D13: Colab Pro runs all training runs of Phase 2 on an L4, one run at a time.
- `scripts/run_queue.py` starts the runs of a queue file in order. It skips each run with a `metrics.json`.
- The first run measures the epoch time, the GPU use, the GPU memory, and the compute units per hour.
- One Claude Code session drives one notebook. This limit comes from the brief.
- The session drives the notebook through the Colab MCP server. `CLAUDE.md` gives the steps.
- Long jobs run in the background and write logs to Drive. Each epoch saves a checkpoint.
- Decision D14: while a job runs, the session runs a short check cell at least every 30 minutes.

Planned layout of the repository:

```
ddd/          package: data, splits, models, training, evaluation, metrics
configs/      one config file for each run type
scripts/      entry points for the command line
app/          the live application
tests/        tests for the split rules, the class map, the test lock, the metrics
notebooks/    one Colab notebook that starts scripts
splits/       split files with driver numbers only
results/      small result files that go in git
docs/         PLAN, LITERATURE, AI_LOG, and the report
data/         raw data, not in git
runs/         run folders, not in git
```

## 14. Risks

| Risk | Action |
|---|---|
| The StateFarm rules limit the data to the competition. | The owner asks the lecturer. After D7, no reserve dataset exists. |
| The extra files of ViT-DD do not download. | G1 reads R1. The owner gets a message. |
| The public code needs old library versions. | The project has its own training code. R0 cannot check it, because the public weights do not download. |
| The live camera view differs from the dataset view. | The guide for the camera position. The own recordings measure the effect. |
| The compute units run out. | Measure the cost in the first run. Train small models on the laptop. |
| 8 test drivers give an uncertain score. | Report each driver and a bootstrap interval. |
| StateFarm has images with no order in time. | Video tests use own continuous recordings only. |
| A person records in a moving car. | The rule is a parked car only. The consent form states it. |

## 15. Coverage of the guideline

| Rubric item | Weight | Where in this plan |
|---|---|---|
| Data preparation and experimental workflow. | 20%. | Sections 3 to 8. |
| Deep learning model construction. | 20%. | Sections 7 and 8.2. Own training code. |
| Model behavior and performance analysis. | 20%. | Sections 8.3, 8.4, and 9. |
| Real-world suitability and limitations. | 15%. | Section 9, steps 3 to 7. Section 14. |
| Working deep learning application. | 15%. | Section 10. |
| AI usage, checks, and reflection. | 10%. | `docs/AI_LOG.md` and section 10 of the report. |

## 16. Open points for the owner

- The lecturer approved exam cheating detection. The approval of the new topic is open.
- The answer of the lecturer about the StateFarm rules is open.
- The exact deadline and the report language are TBD.
- The number of persons for the own recordings and the user test is TBD.
- Decision P3 counts "talking to passenger" as distracted. The owner can change this.
- Decision P7 selects a desktop program. The owner can ask for a web page.
- Decision P8 sets the speed target. The owner can change the number.
