# Driver distraction detection with a camera: technical report (draft)

Draft of 2026-11-17. The AI wrote this draft from the result files of the repository.
The owner checks each section, rewrites it in the words of the team, and owns the final text (guideline sections 19 to 22).
Each number comes from the result file that its section names. Each score is the value of the file times 100, with 2 decimals.
"TBD" marks a part that waits for a result. `scripts/plot_figures.py` makes the figures in `results/figures/`.

## 1. Introduction

### 1.1 Background

A driver who texts, calls, drinks, or turns to a passenger does not watch the road.
A camera in the car can see these actions. A deep learning model can name the action in each camera image.
Such a system can warn the driver while the distraction continues.

### 1.2 Real-world problem

The problem is to find a distracted driver from one RGB camera, live, on common hardware.
The system must work for persons that it did not see in training. A model that only knows its training drivers has no use in a new car.

### 1.3 Target users and stakeholders

- Drivers who want a warning at the moment their attention leaves the road.
- Fleet operators and driving schools that want a record of distraction.
- Researchers who compare driver monitoring models.

### 1.4 Problem statement

Given one image of a driver from the side, give one of 10 classes. Class `c0` is safe driving. Classes `c1` to `c9` are distracted.
From the class, give a status: safe or distracted. If the distracted status continues, give an alert.

### 1.5 Objectives

The project answers five research questions (`docs/PLAN.md` section 1):

- RQ1: Does the public ViT-DD recipe reach its reported accuracy on drivers that the model did not see?
- RQ2: How much accuracy does a small model lose, and how much speed does it gain on a laptop?
- RQ3: Which training choices change the accuracy on new drivers?
- RQ4: How does the model behave on own recordings and on changed images?
- RQ5: Is the best model of the test also the right model for the live application?

### 1.6 Scope

- One dataset: StateFarm Distracted Driver Detection.
- One RGB camera at the side of the driver. No depth, infrared, or face input.
- A desktop program on a laptop. It is a demonstration, not a safety system for the road.
- Tests in a parked car only.

## 2. Dataset and Data Preparation

### 2.1 Dataset source

The dataset is StateFarm Distracted Driver Detection from Kaggle, under the competition rules.
The project does not share its files. The images and data from them stay out of git and out of public material.

### 2.2 Dataset characteristics

Source: `results/data_audit/statefarm.json`.

- 22,424 labeled images of 26 drivers.
- 10 classes. Each image has one class.
- Each image is an RGB image of 640 by 480 pixels.
- The camera sees the driver from the passenger side of a car with the wheel on the left.
- The competition has more images with no label. This project does not use them.

| Code | Class | Images |
|---|---|---|
| `c0`. | Safe driving. | 2489. |
| `c1`. | Texting, right hand. | 2267. |
| `c2`. | Phone call, right hand. | 2317. |
| `c3`. | Texting, left hand. | 2346. |
| `c4`. | Phone call, left hand. | 2326. |
| `c5`. | Radio. | 2312. |
| `c6`. | Drinking. | 2325. |
| `c7`. | Reaching behind. | 2002. |
| `c8`. | Hair and makeup. | 1911. |
| `c9`. | Talking to passenger. | 2129. |

### 2.3 Data distribution

- The classes are near balance. The largest class has 2489 images, and the smallest has 1911.
- The drivers are not in balance. Driver p072 has 346 images, and driver p021 has 1237.
- Some drivers miss a part of a class. For example, p072 has 2 images of `c7` and 6 images of `c5`.

### 2.4 Data quality

The audit script `scripts/audit_statefarm.py` gives these results:

- No duplicate rows in the label file.
- No missing file and no file that does not open.
- No image file that the label file does not name.
- All images have the same size, 640 by 480 pixels.

### 2.5 Data preprocessing

Each image goes through the same steps in the evaluation and in the application (`ddd/data.py`):

1. Resize the short side to 224 pixels, with bicubic interpolation.
2. Cut the 224 by 224 square from the center.
3. Normalize with the ImageNet mean and standard deviation.

Training adds random steps, the augmentation of the ViT-DD code (`results/vitdd_recipe_check.md`):

1. A random crop of 224 by 224 pixels, with a padding of 4 pixels by reflection.
2. A random horizontal flip. The final model has no flip (decision P16).
3. One of three steps by random choice: gray scale, solarization, or Gaussian blur (3-Augment).
4. A color jitter of 0.3 for brightness, contrast, and saturation.

### 2.6 Train, validation, and test split

The split is by driver. Each driver is in one part only, so each score measures persons that the model did not see.

```
StateFarm, 26 drivers
├── Test part: 8 drivers, 6,525 images
│     Locked. Only two points score it.
└── Development part: 18 drivers, 15,899 images
      3 folds. Each fold trains on 12 drivers and validates on the other 6.
```

- The test drivers are the 8 drivers of the public ViT-DD split list. Thus the test score compares with the paper.
- A test lock in the code refuses the test drivers unless a run gives a reason. Each unlock adds a row to `results/test_access_log.csv`.
- Only the replication R1 and the final evaluation read the test drivers. No design choice uses a test score.
- The file `splits/statefarm_v2.json` holds the drivers of each part and each fold.

### 2.7 Data limitations

- 26 drivers is a small number of persons.
- The test drivers sit in the same cars as the development drivers. Thus the test measures new persons in known cars.
- The cars have the wheel on the left. Cars in Indonesia have the wheel on the right, so the camera view is a mirror (decision D21).
- The dataset has no night images and no infrared images.

## 3. Deep Learning Fundamentals and Model

### 3.1 Relevant theoretical concepts

- A convolutional network (CNN) finds local patterns with small filters. Its layers grow the area that each unit sees.
- A vision transformer (ViT) cuts the image into patches of 16 by 16 pixels. Each patch becomes a vector, a token.
- Self-attention lets each token use all other tokens. Thus the model can join the hands, the face, and the wheel in one step.
- A class token collects the result. A linear head turns it into 10 class scores.
- Transfer learning starts from weights that learned on ImageNet. The training then fits these weights to the new task.

### 3.2 Selected architecture

The final model is DeiT-III Small, the model `deit3_small_patch16_224.fb_in1k` of `timm` 1.0.31.

- Input: one RGB image of 224 by 224 pixels, thus 196 patches.
- Parameters: 21,678,346. Size of the weights: 82.76 MB (`results/final/cost.md`).
- Start weights: the ImageNet-1k weights of `timm`.

### 3.3 Model components

- A patch embedding: a linear layer that turns each 16 by 16 patch into a token.
- A position embedding: one learned vector for each patch place.
- 12 transformer blocks. Each block has self-attention and a small feed-forward network, each with a layer scale.
- A layer norm and a linear head with 10 outputs.

### 3.4 Loss function

The loss is cross entropy with label smoothing 0.1. Label smoothing moves a small part of the target to the other classes.
It keeps the model from scores that are too sure. The replication R1 and the baseline B0 have no label smoothing.

### 3.5 Optimizer

AdamW with the learning rate 0.0003 and the weight decay 0.1. Experiment E4 compares 0.0001, 0.0003, and 0.0009.

### 3.6 Training strategy

| Item | Value |
|---|---|
| Epochs. | 20. |
| Batch size. | 128. The last part batch of each epoch is dropped. |
| Learning rate curve. | 5 warm-up epochs from 0.000001, then a cosine curve to 0. |
| Trained layers. | All layers. |
| Augmentation. | The paper set with no flip. See section 2.5. |
| Numbers. | 32-bit floats. |
| Seed. | 42. |
| Checkpoint. | After each epoch. A run continues after a disconnect. |

The score of a run is the score of its last epoch (decision P12). No epoch is selected with validation or test scores.

### 3.7 Why the model was selected

- In E1, DeiT-III Small has the highest mean validation macro F1 of five models, 88.34 (`results/development/e1.md`).
- It reaches the live target of 5 frames per second on the laptop CPU and GPU (decision P8).
- With the paper set and no flip, its mean macro F1 is 89.39, the highest of all development runs (`results/development/e3.md`).
- The rules of `docs/PLAN.md` section 8.5 select it as `best` and as `live`. The owner approved it (decision D19).

## 4. System Design and Implementation

### 4.1 System architecture

The system has two parts:

- A training and evaluation part: the Python package `ddd`, scripts, and a Colab notebook that starts the scripts.
- A live application: `app/live.py` on the laptop.

```
Training (Colab L4)                         Application (laptop)
configs/*.json                              camera, video, or image
   │                                           │
scripts/train.py ── run folder ── last.pt ──► app/live.py
   │   (config, history, metrics,              │  BGR to RGB, mirror (D21)
   │    environment, checkpoint)               │  eval_transform
scripts/evaluate.py (test lock)                │  DeiT-III Small
   │                                           │  mean of 5 frames, alert
results/*.json, *.csv, *.md                    screen and CSV log
```

### 4.2 Application workflow

1. Read a frame from a camera, a video file, or an image file.
2. Turn the frame from BGR to RGB. In a car with the wheel on the right, flip it and swap the hand classes (decision D21).
3. Prepare the frame with the steps of section 2.5. A test makes sure that the app and the evaluation give the same input.
4. Run the model and turn its outputs into class scores with a softmax.
5. Take the mean scores of the last 5 frames (decision P18).
6. Show the class, the status, and the 10 class scores on the screen.
7. Give an alert after 2.0 seconds of a distracted status (decision P18).
8. Write each frame to a CSV log file.

### 4.3 Model integration

- The app reads the config and the last checkpoint of a run folder. By default, it reads the final run.
- The option `--mirror` serves cars with the wheel on the right. The options `--window` and `--alert-seconds` change the smoothing.
- The option `--truth` shows the true class of a demo video. A demo video uses the fold model that did not train on its driver.

### 4.4 Development framework

- PyTorch and torchvision for the training and the inference.
- `timm` 1.0.31 for the models and the start weights.
- OpenCV for the camera, the video, and the screen of the app.
- NumPy and Pillow for the data steps. Pytest for the tests.

### 4.5 Hardware and software environment

| Item | Training | Application |
|---|---|---|
| Machine. | Google Colab Pro, NVIDIA L4 with 23 GB. | Laptop, NVIDIA RTX 3050 6 GB and AMD Ryzen 7 8845HS. |
| Python. | 3.13.16. | 3.10.9. |
| PyTorch. | 2.14.1, CUDA 13.0. | 2.6.0, CUDA 12.4. |

The Colab values come from `results/final/final_deit3_small_noflip/environment.json`. The laptop values come from `results/final/speed_laptop.json`.
The Colab runtime used about 1.94 compute units per hour (`results/compute/first_run.json`).

### 4.6 Implementation details

- Each run reads one config file. Its folder holds the config, the history of each epoch, the metrics, the GPU type, the library versions, and the git commit.
- A queue script trains the runs one at a time and skips each complete run.
- A short check cell runs at least every 30 minutes during a Colab job, because Colab can stop an idle runtime (decision D14).
- The repository has 49 tests. They cover the split by driver, the test lock, the metrics, the training loop, and the app input.

## 5. Experimental Methodology

### 5.1 Research questions

See section 1.5. RQ1 is the replication. RQ2 and RQ3 are the experiments of Phase 2. RQ4 and RQ5 use the final model.

### 5.2 Baseline

- R1, the replication: the standard ViT of the ViT-DD paper, a DeiT-III Base model, on the 18 development drivers.
- R1 trains the attention layers, the head, and the position embedding only, as the ViT-DD code does.
- B0, the development baseline: the R1 recipe on each of the 3 folds.
- `results/replication/r1_vit_s42/recipe_differences.md` lists the differences from the paper recipe. The main one: no face input and no emotion head (decision D10).

### 5.3 Experimental variables and scenarios

Each experiment changes one variable and runs on the 3 folds.

| ID | Question | Variable | Values |
|---|---|---|---|
| E1. | Which model size fits live use? | Architecture. | MobileNetV3-Large, EfficientNet-B0, ResNet-50, DeiT-III Small, DeiT-III Base. |
| E2. | Does pretraining help on new drivers? | Start weights. | ImageNet and random, for DeiT-III Small and MobileNetV3-Large. |
| E3. | Does augmentation help? Does the flip hurt? | Augmentation. | None, the paper set, and the paper set with no flip. |
| E4. | How sensitive is training to the learning rate? | Learning rate. | 0.0001, 0.0003, and 0.0009. |
| E5. | How many drivers does training need? | Training drivers of each fold. | 4, 8, and 12. |

- E1 trains all layers with label smoothing 0.1. The CNN models use the learning rate 0.001, the ViT models 0.0003.
- The "None" value of E3 has no random step. The random crop becomes a center crop (decision P15).
- E2 adds MobileNetV3-Large, because DeiT-III Small with random weights did not learn (decision D18).
- E4 and E5 use the winner of E3. E5 keeps the first 4 or 8 drivers after a shuffle with seed 42 (decision P21).

### 5.4 Hyperparameters

Section 3.6 gives the values of the final model. The other runs use the same values, except for the variable of their experiment.

### 5.5 Evaluation metrics

- Accuracy: the share of images with the correct class.
- Class-mean accuracy: the mean of the accuracy of each class.
- Macro F1: the mean of the F1 scores of the 10 classes. Development compares models by this score.
- Status scores: precision, recall, and F1 of "distracted" against "safe". The alert of the app uses the status.
- Speed: processed frames per second on the laptop, one frame at a time.

### 5.6 Experimental procedure

1. Replicate R1 and score the test drivers one time. Gate G1: 85.36 or more, 5 points below the paper value of 90.36.
2. Run B0 and E1 to E4 on the 3 folds. Take the score of the last epoch of each fold, and the mean of the 3 folds.
3. Select `best` and `live` by the mean validation macro F1 and the speed target.
4. Train the selected model on all 18 development drivers. Score the test drivers one time.
5. Score the final model on changed test images (decision D20), and measure its cost on the laptop.

## 6. Results

### 6.1 Replication

Source: `results/replication/scores.md`.

| Score | Paper | R1 |
|---|---|---|
| Accuracy. | 90.36. | 89.46. |
| Class-mean accuracy. | No value. | 89.16. |
| Macro F1. | No value. | 88.93. |

Both R1 scores reach the pass value 85.36. Thus gate G1 passed. The answer to RQ1 is yes, within 1 point of the paper.

### 6.2 Training results

- Each development run with ImageNet weights fits the training images to a training accuracy of 99.92 or more in epoch 19.
- The final model reaches a training accuracy of 99.98 in epoch 19 (`results/final/scores.md`).
- DeiT-III Small with random weights stays at a training accuracy of 22.83 or less (`results/development/e2.md`).
- Figure 1, `results/figures/b0_curves.png`: the training and validation curves of B0 on the 3 folds.
- Figure 2, `results/figures/final_recipe_curves.png`: the same curves for the final recipe.
- In both figures, the training accuracy is 98.18 or more in epoch 3. From epoch 3, the validation accuracy stays below it.

### 6.3 Validation results

Sources: the `_folds.json` files and the comparison files in `results/development/`. Each value is the mean of the 3 folds.

| Run | Accuracy | Macro F1 |
|---|---|---|
| B0, DeiT-III Base, R1 recipe. | 88.31. | 88.16. |
| E1, MobileNetV3-Large. | 79.23. | 79.21. |
| E1, EfficientNet-B0. | 76.22. | 75.72. |
| E1, ResNet-50. | 84.56. | 84.46. |
| E1, DeiT-III Small. | 88.62. | 88.34. |
| E1, DeiT-III Base. | 88.85. | 88.27. |
| E2, DeiT-III Small, random weights. | 15.94. | 8.85. |
| E2, MobileNetV3-Large, random weights. | 41.75. | 40.87. |
| E3, DeiT-III Small, no augmentation. | 82.39. | 81.77. |
| E3, DeiT-III Small, no flip. | 89.85. | 89.39. |
| E4, no flip, learning rate 0.0001. | 86.82. | 86.80. |
| E4, no flip, learning rate 0.0009. | 84.73. | 84.45. |
| E5, no flip, 4 training drivers. | 77.63. | 76.57. |
| E5, no flip, 8 training drivers. | 86.11. | 86.03. |

### 6.4 Test results

Source: `results/final/scores.md`. The test lock opened for this evaluation after the selection.

| Score | Final model | R1 | Paper |
|---|---|---|---|
| Accuracy. | 91.97. | 89.46. | 90.36. |
| Class-mean accuracy. | 91.55. | 89.16. | No value. |
| Macro F1. | 91.30. | 88.93. | No value. |
| Status precision. | 98.38. | 96.50. | No value. |
| Status recall. | 97.19. | 97.38. | No value. |
| Status F1. | 97.78. | 96.94. | No value. |

- The accuracy of the 8 test drivers is from 79.87 (p050) to 97.02 (p041).
- A bootstrap over the 8 test drivers gives a 95 percent interval of the accuracy (`results/final/bootstrap.json`).
- The interval is 88.07 to 95.14 for the final model, and 86.79 to 92.38 for R1. The two intervals overlap.
- Thus 8 test drivers do not show that the final model is better than R1 for all drivers.

### 6.5 Model comparison

Source: `results/development/e1.md`. FPS is the number of processed frames per second on the laptop.

| Model | Macro F1 | Parameters | MB | GPU FPS | CPU FPS |
|---|---|---|---|---|---|
| MobileNetV3-Large. | 79.21. | 4,214,842. | 16.27. | 50.27. | 43.51. |
| EfficientNet-B0. | 75.72. | 4,020,358. | 15.62. | 50.17. | 35.52. |
| ResNet-50. | 84.46. | 23,528,522. | 90.06. | 61.93. | 17.73. |
| DeiT-III Small. | 88.34. | 21,678,346. | 82.76. | 61.96. | 18.52. |
| DeiT-III Base. | 88.27. | 85,824,010. | 327.45. | 52.27. | 5.89. |

Figure 3, `results/figures/e1_tradeoff.png`, shows the macro F1 against the CPU speed of each model.

### 6.6 Computational performance

- Training on the Colab L4: among the E1 fold runs, the sum of the epoch times is from 587.91 seconds (MobileNetV3-Large) to 2927.74 seconds (DeiT-III Base).
- The final model trained in 1611.57 seconds of epoch time on the 18 development drivers.
- The final model on the laptop: 62.59 frames per second on the GPU, and 16.30 on the CPU (`results/final/cost.md`).
- Its peak GPU memory in the speed test is 95.74 MB.

## 7. Analysis and Discussion

### 7.1 Model behavior

- The model separates "safe" from "distracted" better than it separates the 10 classes. The test status F1 is 97.78, and the macro F1 is 91.30.
- Gray images lower the validation macro F1 of B0 by 3.58 points (`results/development/b0_gray.md`).
- For DeiT-III Small with all layers trained, the fall is 1.30 points or less (`results/development/e3_errors.md`). Thus the model uses color only a little.

### 7.2 Overfitting and underfitting

Source: `results/development/b0_fit.md`.

- B0 fits the training images almost fully. Its training accuracy is 10.30 to 13.45 points above its validation accuracy in epoch 19.
- The validation macro F1 does not fall in the late epochs. In each fold, the mean of epochs 11 to 19 is higher than the mean of epochs 3 to 10.
- Thus the curves show no classic overfit. The gap is a gap to new drivers.
- An early stop by the lowest validation loss gives no reliable gain. Its difference of 0.90 points is smaller than the change between two epochs.
- No model with ImageNet weights underfits. DeiT-III Small with random weights does not learn, a strong underfit.

### 7.3 Error patterns

Sources: `results/development/b0_errors.md`, `results/development/e3_errors.md`, and `results/final/scores.md`.

- Talking to the passenger against safe driving is the largest error. On the test drivers, 134 images of `c9` have the prediction `c0`, 23.67 percent of that class.
- The model gives the class "hair and makeup" to other classes with a hand near the face. On the test drivers, 210 images get this wrong class.
- On the validation folds, the left and right hand classes give almost no errors. On the test drivers, 24 images of `c1` have the prediction `c3`.
- The scores differ much between drivers. On the folds, p072 has the lowest accuracy of the final recipe, 60.40.

### 7.4 Effect of the training scenarios

- Model size (E1): the two DeiT-III models score 3.81 or more points above the best CNN, ResNet-50. DeiT-III Small equals DeiT-III Base with a quarter of the parameters.
- Start weights (E2): ImageNet weights are necessary. With random weights, MobileNetV3-Large loses 38.34 points, and DeiT-III Small does not learn.
- Augmentation (E3): with no augmentation, the macro F1 is 6.57 points lower. The flip does not help: with no flip, the macro F1 is 1.04 points higher.
- Learning rate (E4): the default value 0.0003 is best. Three times the default costs 4.94 points, and one third costs 2.59 points.
- Training drivers (E5): from 4 to 8 drivers, the macro F1 rises by 9.46 points. From 8 to 12 drivers, it rises by 3.36 points.
- Thus more drivers help, and the score does not reach a limit at 12 drivers. Figure 5, `results/figures/e5_drivers.png`, shows this.
- E5 changes the number of training steps too, because each run has 20 epochs (`results/development/e5.md`).

### 7.5 Strengths

- The test score of the final model is above the paper value for the standard ViT, with a smaller model.
- The split by driver and the test lock make each score a score on new persons.
- The model reaches the live speed target on a laptop CPU.
- Brightness, contrast, blur, and noise change the test macro F1 by 1.66 points or less (`results/final/robustness.md`).

### 7.6 Weaknesses

- Talking to the passenger gives most of the missed alerts. Its test F1 is 68.01.
- A covered part of the image lowers the scores much. With 30 percent of the image under a gray square, the macro F1 is 47.24.
- Some drivers have much lower scores than others, for example p050 on the test.

### 7.7 Trade-offs

- Accuracy against speed: DeiT-III Base and DeiT-III Small have equal F1, but DeiT-III Small is about 3 times faster on the CPU.
- Macro F1 against status recall: with no flip, the macro F1 is higher, but more distracted images get the prediction "safe" on the folds (637 against 438).
- Smoothing over frames: a longer window gives fewer false alarms but a later alert. Experiment E8 measures this on own video. TBD.

### 7.8 Robustness

Source: `results/final/robustness.md`.

| Change | Strongest value | Macro F1 |
|---|---|---|
| Brightness. | Factor 0.2. | 89.64. |
| Contrast. | Factor 0.2. | 90.33. |
| Blur. | Radius 4 pixels. | 90.92. |
| Noise. | Standard deviation 40. | 90.05. |
| Cover. | 30 percent of the image. | 47.24. |

The status recall stays at 88.18 or more with each change. Figure 4, `results/figures/robustness.png`, shows the macro F1 of each change.

### 7.9 Limitations

- One dataset with 26 drivers, all in a few cars with the wheel on the left.
- One seed for each run. The change between folds is larger than some differences between runs.
- No own recordings yet. Thus the behavior in a car of the target users is not measured. TBD.
- The app flips the frames for cars with the wheel on the right. Its effect on own recordings is not measured. TBD.

## 8. Real-World Evaluation

TBD. This section waits for the own recordings and the user test (`docs/recording/GUIDE.md`, `docs/PLAN.md` section 10).

## 9. Conclusion

TBD. This section waits for section 8.

## 10. AI Usage Declaration

TBD. The owner writes this section from `docs/AI_LOG.md`, with the cells marked `OWNER`.
