# Literature survey: driver distraction detection

Written on 2026-11-13 and 2026-11-14 by Claude (Fable 5.2) in the planning session. The owner reviews it.
Every score in this file comes from the cited paper or from its public code repository.
No score comes from a run of this project.

## Terms

- A *frame* is one image of a video.
- A *clip* is a short cut video with one action.
- An *event* is one labeled action interval, with its true start and end.
- A split *by driver* puts each driver in one part only.
- *Top-1 accuracy* is the share of samples with the correct class.
- *Macro F1* is the mean of the F1 scores of all classes. Each class has the same weight.
- The *Read* column tells how the source was read: full text, code, or abstract only.

## Decision of the owner (2026-11-14)

- The project starts on StateFarm. The owner requests AUC v2 on the same day.
- AUC v2 becomes the test on a second dataset as soon as it arrives.
- The replication target is ViT-DD (rows 15 and 16).
- The owner rejected DMD because 14 drivers are too few.
- SAM-DD, Drive&Act, and DAD did not pass the rules of the owner. The section "Comparison for the owner" gives the reasons.

## Facts about the DMD public release (read on 2026-11-13)

- Only RGB video of the real car is public. Simulator, IR, and depth material is removed.
- Only 14 subjects are free to use. Their numbers are 1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37.
- The license is CC BY-NC-ND 4.0, for academic use only. The download needs a form.
- The distraction labels have 7 levels. Level 6 is `driver_actions`.
- The other levels are camera occlusion, gaze on road, talking, hands on wheel, hand on gear, and objects.
- Labels are in OpenLABEL (VCD 5.0 or later). Videos are `.mp4` files.
- The export tool DEx splits the exported files by a ratio, not by driver. Do not use that split.
- Earlier releases had more drivers. Cañas et al. had 5 subjects in 2021. Hasan et al. had 29 subjects in 2023.

Sources: the DMD site, the README of the DMD repository, and the DMD wiki.

Level 6 has 13 classes and one more label, `Unclassified`:

```
Safe Driving, Texting (Right), Phone call (Right), Texting (Left), Phone call (Left),
Radio, Drinking, Reach Side, Hair and Makeup, Talking to passenger, Reach Backseat,
Change Gear, Stand Still-Waiting
```

Not known before the download: the sessions of each subject, the class counts, and the size.

## Facts about the other downloads (read on 2026-11-13 and 2026-11-14)

| Dataset | Size | Access | Problem |
|---|---|---|---|
| SAM-DD. | Three files of 5.58, 5.06, and 4.94 GB on Hugging Face. | Direct link, no form. | No RGB data is in these files. See the note below. |
| Drive&Act. | NIR cameras: 1.68 to 2.89 GB each. Kinect RGB: 5.26 GB. Labels: 9 MB. | Direct link, no login. | Only 15 drivers. Most cameras are NIR. |
| DAD. | About 95 GB (stated in the paper). | Direct link. | Only depth and IR. A usual webcam does not give these. |
| StateFarm. | 22,424 labeled images of 26 drivers. The file size was not read. | Kaggle account and the competition rules. | The rules limit the data to the competition. |
| AUC v2. | 10,555 train images and 1,123 test images of camera 1. | A signed license agreement by email to the authors. | The time of the reply is not known. |
| 100-Driver. | More than 470,000 images. The file size is not known. | A license form by email. | The time of the reply is not known. |

Note on SAM-DD: the three files were uploaded on 2026-11-01.
Each file starts with an error note. The note says that `SAM-DD(RGB).rar` is larger than the allowed limit.
After the note, each file holds an archive of a different dataset and images of `SAM-DD(depth)`.
The file with the name `SAM-DD(RGB).zip` has no index at its end. Thus it is cut off.
Samples at 16 positions of that file showed only depth images.
The folders have the names `Tester` and `Val`, with one folder for each subject.

Note on StateFarm: the rules say that participants "must use the Data solely for the purpose and duration of the Competition".

## Table 1: papers on DMD (method, split, metric, score)

| ID | Paper | Method | Split | Metric | Score |
|---|---|---|---|---|---|
| 1 | Ortega et al. 2020. | 2D CNN on 4 or 8 sampled frames. 3D CNN. | 24, 7, and 6 drivers for train, validation, and test. | Top-1 accuracy per clip. | Inception-v3, RGB, 93.2. 3D-ResNeXt, RGB, 91.6. MobileNet-v2 with MaxP-MLP, RGB, test part, 92.0. RGB with IR and depth, 93.7. |
| 2 | Cañas et al. 2021. | Image models (MobileNetV1, InceptionV3). Video models (MobileNetV1 with LSTM, Conv3D, ConvLSTM). | 80% train and 20% test over clips or images. | Top-1 accuracy per image or clip. | Image MobileNetV1, 99.5. MobileNetV1 with LSTM, 30 frames, 97.3. Conv3D, 97.2. |
| 3 | Video Swin study, arXiv 2207.12148, 2022. | Video Swin Transformer. | "80/20 train/test split". | Accuracy per clip. | 97.5. |
| 4a | Hasan et al. 2024 (DriveCLIP). | Zero-shot CLIP ViT-L/14@336px. No training. | 7 folds. Each fold has 23 train drivers and 6 test drivers. | Top-1 accuracy per frame. | 48.12. |
| 4b | Same paper. | Single-frame CLIP. The CLIP network is frozen. A logistic regression classifies its features. | Same folds. | Top-1 accuracy per frame. Macro F1 per frame. | Top-1 82.65. Macro F1 0.827. The folds give 80.77 to 84.16. |
| 4c | Same paper. | Multi-frame CLIP. A majority vote joins the frame results of one event. | Same folds. | Top-1 accuracy per event. | 93.39. |
| 4d | Same paper. | VideoCLIP. An S3D network with HowTo100M weights reads 30 frames. | Same folds. | Top-1 accuracy per event. | 98.44. |
| 4e | Same paper. | CNN baselines with ImageNet weights, 1 FPS, and a majority vote. | Same folds. | Top-1 accuracy per event. | ResNet18 85.95. MobileNet-v2 85.77. VGG16 56.15. AlexNet 45.86. |

## Table 2: papers on DMD (data, split by driver, reproduction)

| ID | Data | By driver | Reproduction with the 2026 release | Read |
|---|---|---|---|---|
| 1 | dBehaviourMD. Body camera. 13 classes. Clips of about 50 frames. Classes balanced by sampling. | Yes, stated. | Not possible. It needs 37 drivers, IR, and depth. The driver lists are not public. | Full text. |
| 2 | DMD "lite". 1 group, 5 subjects. Body camera, RGB. 9 classes. | No. The paper has no statement about drivers. | Not useful. The same 5 subjects are in all parts. | Full text. |
| 3 | 10 subjects. 9 classes. Sequences of 30 frames. | Not stated. | Not useful. | Full text. |
| 4a to 4e | 29 drivers. Body camera, RGB. 10 classes. Frames at 1 FPS. At most 800 frames per subject and class. | Yes. | In part. Code, fold file, and fold results are public for 4b. | Full text and code. |
| 4d | Same data. | Yes. | Weak. The video code has no manual and no DMD part. | Full text and code. |

Notes on the DMD papers:

- Row 4: only 12 of the 29 drivers are in the 2026 release. Subjects 36 and 37 are public but are not in the fold file.
- Row 4: the folds have no validation part.
- Row 4: the test folds overlap. Only 12 of the 29 drivers are in a test fold. Thus the spread between folds is too small.
- Row 4: `yawning` is one of the 10 classes. It is not a `driver_actions` label. The paper does not state its source.
- Row 4: Hasan et al. mark row 1 as "not leave-driver-out" in their Table VIII. The DMD paper states a split by driver. This conflict is open.
- Row 1: Table 3 of the DMD paper does not state the part. Table 4 gives the test part.

## Table 3: papers on other public datasets (method, split, metric, score)

| ID | Paper | Method | Split | Metric | Score |
|---|---|---|---|---|---|
| 5 | Hasan et al. 2024, on SAM-DD. | Single-frame CLIP. VideoCLIP. MobileNet-v2 with a majority vote. | 7 folds. Each fold leaves drivers out. | Top-1 accuracy per frame or event. | Single-frame 89.14. VideoCLIP 97.86. MobileNet-v2 92.95. |
| 6 | Hasan et al. 2024, on StateFarm. | Single-frame CLIP. CNN baselines. | 8 folds. Each fold leaves drivers out. | Top-1 accuracy per frame. | CLIP 83.15. ResNet18 82.68. MobileNet-v2 80.04. AlexNet 74.84. |
| 7 | Hasan et al. 2024, on SynDD1. | VideoCLIP at 20 FPS. Multi-frame CLIP. Single-frame CLIP. | 7 folds. Each fold leaves drivers out. | Top-1 accuracy per frame or event. | VideoCLIP 81.85. Multi-frame 70.36. Single-frame 57.35. |
| 8a | Eraqi et al. 2019, on AUC. | InceptionV3 and a weighted ensemble. | Random, 75% train and 25% test. | Accuracy per image. | Ensemble 95.98. InceptionV3 95.17. |
| 8b | Same paper. | InceptionV3, ResNet50, VGG-16. | 38 train drivers and 6 test drivers. | Accuracy per image. | InceptionV3 90.07. ResNet50 81.70. VGG-16 76.13. |
| 9 | Martin et al. 2019, on Drive&Act. | I3D. | 3 official splits. Each has 10 train, 2 validation, and 3 test drivers. | Mean accuracy per class, on chunks of 3 s or less. | Test 63.64. Validation 69.57. |
| 10 | Peng et al. 2022 (TransDARC), on Drive&Act. | Video Swin-B. TransDARC adds a feature calibration. | The same 3 official splits. | "Top-1 accuracy" on chunks. | Video Swin, test, 85.74. TransDARC, test, 89.65. |
| 11 | Wang et al. 2023, on 100-Driver. | CNNs with ImageNet weights. | By driver. Day: 47 train, 6 validation, 12 test. Night: 37, 5, 10. | Accuracy per image. | Day, all cameras: EfficientNet-B0 79.0. MobileNetV3 76.4. ResNet50 73.9. Four cameras fused, MobileNetV3: 86.9. |
| 12 | Köpüklü et al. 2021, on DAD. | 3D ResNet-18 with a contrastive loss. It separates normal driving from all other actions. | Fixed. 25 train subjects and 6 test subjects. | Area under the ROC curve, on continuous test video. Accuracy at the best test threshold. | Area 0.9673 with both cameras, both modalities, and smoothing. Accuracy 92.34. One camera with depth: area 0.9128. |
| 13 | Ma et al. 2022, on DAD. | Image models with no temporal model. | The DAD split. | Area under the precision-recall curve. | 97.5%. |
| 14 | Chai et al. 2024, on StateFarm and AI City. | A comparison of splits and metrics. | Leave N drivers out. | Not read. | Not read. The paper is behind a paywall. |
| 15a | Ma and Wang 2022 (ViT-DD), on StateFarm. | ViT-DD. A DeiT-III Base network reads the driver image and a face crop. A second head learns emotion labels. | By driver. 18 train drivers and 8 test drivers. | Accuracy per image. | ViT-DD 92.51. Standard ViT 90.36. DD-RCNN, from its paper, 86.00. |
| 15b | Same paper, on StateFarm. | ViT-DD. | By image. 70% train and 30% test. | Accuracy per image. | 99.63. |
| 16 | Same paper, on AUC v2. | ViT-DD. | The official split by driver. 38 train drivers and 6 test drivers. | Accuracy per image. | ViT-DD 93.59. Standard ViT 90.92. C-SLSTM, from its paper, 92.70. |

## Table 4: other public datasets (data, split by driver, access)

| ID | Data | By driver | Access to data and code | Read |
|---|---|---|---|---|
| 5 | SAM-DD. 42 drivers. RGB and depth. Side and front camera. 10 classes. Laboratory setup. | Yes. | The public files hold no RGB data on 2026-11-14. The fold lists are not public. | Full text, code, site, file samples. |
| 6 | StateFarm. 26 drivers. Images. 10 classes. | Yes. | Kaggle, under the competition rules. The fold generator is public. | Full text and code. |
| 7 | SynDD1. 18 drivers. Video. 3 cameras. | Yes. | The fold file is public. Data access is not known. | Full text and code. |
| 8a | AUC Distracted Driver. 44 drivers. Images. 10 classes. | No. | A signed license agreement by email. | Full text and site. |
| 8b | Same data. | Yes. | The split by driver is fixed in the release. | Full text and site. |
| 9 | Drive&Act. 15 drivers. Video. 34 fine activities of automated driving. NIR cameras and one Kinect. | Yes. | Download with no login. Research use only. | Full text and site. |
| 10 | Drive&Act, NIR camera at the front top. | Yes. | Public code. | Full text. |
| 11 | 100-Driver. 100 drivers. Images. 22 classes. 4 cameras. Day RGB and night NIR. | Yes. | A license form by email is necessary. | Full text. |
| 12 | DAD. 31 subjects. Depth and IR. Front and top camera. Driving simulator. About 95 GB. | Yes. | Data, code, and weights are public. | Full text and code site. |
| 13 | DAD. | Yes. | "Code will be released". | Abstract. |
| 14 | StateFarm and AI City. | Yes. | Not known. | Abstract. |
| 15a | StateFarm. 22,424 labeled images. 640 by 480 pixels, RGB. | Yes. | Code, config files, split lists, and weights are public. | Full text and code. |
| 15b | Same data. | No. | Same code. | Full text and code. |
| 16 | AUC v2, camera 1. 10,555 train images and 1,123 test images. | Yes. | Same code. The data needs the license agreement. | Full text and code. |

Notes on the other datasets:

- Row 10: the metric name differs from row 9. Row 9 reports the mean accuracy per class.
- Row 10: a 2025 paper (arXiv 2503.04078) reports UniFormerV2 on Drive&Act. Its mean accuracy per class is 61.58 and its Top-1 accuracy is 78.63. This came from a search summary, not from the full text.
- Row 12: the test videos hold 16 action types that are not in the training part.
- Row 12: the paper has no validation part. It selects the threshold on the test part.
- Row 15a: the paper text says "18/6 of the total 28 drivers". The public split lists hold 18 and 8 drivers of 26.
- Row 15a: the paper text gives a batch size of 256. The public config gives 128.
- Rows 15 and 16: the paper says that its accuracy gives each class the same weight. The public code logs the usual accuracy.
- Rows 15 and 16: the config has no validation part. It reads the test list as the validation list.
- Rows 15 and 16: the class numbers differ between the two datasets. Number 1 is texting right in StateFarm and phone right in AUC.
- Rows 15 and 16: the links to the face crops and to the weights answer. A download was not tried.

## Findings

1. **A split that is not by driver gives a higher score**. Eraqi et al. report 95.98% with a random split and 90.07% with a split by driver. ViT-DD reports 99.63% by image and 92.51% by driver on StateFarm. On DMD, Cañas et al. report 99.5% with 5 subjects in all parts.
2. **The unit changes the score**. With the same DMD data and folds, Hasan et al. report 82.65% per frame and 93.39% per event. The video model reaches 98.44%. The event scores use the true start and end of each event. A product on continuous video does not know them.
3. **Some scores use the test part for selection**. DriveCLIP and ViT-DD have no validation drivers. DriveCLIP selects the backbone, the frame rate, and the camera by "the highest performance". The public video code of DriveCLIP saves the checkpoint with the best accuracy on the test fold.
4. **The video model of DriveCLIP builds a clip from frames of one label**. The public code takes 30 frames from the frame pool of one person and one class. Thus each clip has one class by construction.
5. **No DMD paper is exactly reproducible today**. The 2026 release has 14 drivers. The DMD paper used 37. DriveCLIP used 29.
6. **Honest scores by driver, per frame or image, are between 79% and 94%**. Examples are 100-Driver 79.0%, DMD 82.65%, StateFarm 83.15% and 92.51%, SAM-DD 89.14%, and AUC 90.07% and 93.59%.
7. **The split changes the score on one dataset**. On StateFarm, 8 folds give 83.15% for a CLIP model. One split of 18 and 8 drivers gives 90.36% for a standard ViT. The models differ too.

## Comparison for the owner (size, live use, drivers)

The owner asked for a dataset with more than 14 drivers, a small size, and live use with a usual camera.

| Dataset | Drivers | Size | Live use with a usual camera | Result |
|---|---|---|---|---|
| SAM-DD. | 42. | Not known for RGB. | Not possible now. The RGB data is missing. | Blocked. |
| Drive&Act. | 15. | Small. 1.68 to 5.26 GB for each camera. | Weak. NIR data and activities of automated driving. | Too few drivers. |
| DAD. | 31. | Large. About 95 GB. | Not possible. It needs a depth camera. | Too large and not live. |
| StateFarm. | 26. | Small. | Yes. RGB camera at the side of the driver. | Selected as the start. |
| AUC v2. | 44. | Small. | Yes. RGB camera at the side of the driver. | Requested. Second dataset. |
| DMD. | 14. | Not known. | Yes. RGB, real car, 3 cameras. | Rejected by the owner. |

StateFarm and AUC v2 have the same 10 classes and a similar camera position.
Thus one model and one application serve the two datasets.

## Not reviewed

- 3MDAD, EBDD, and the AI City Challenge track for naturalistic driving actions (SynDD2).
- The papers that the SAM-DD site lists, for example DualScope (AAAI 2026).
- The full text of rows 13 and 14.
- The papers that ViT-DD compares with (DD-RCNN, C-SLSTM, ADNet).

## Sources

- DMD site: https://dmd.vicomtech.org/
- DMD repository and wiki: https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- Ortega et al. 2020, ECCV workshops: https://arxiv.org/abs/2008.12085
- Cañas et al. 2021, VISAPP: https://www.scitepress.org/Papers/2021/102445/102445.pdf
- Video Swin study: https://arxiv.org/abs/2207.12148
- Hasan et al. 2024, IEEE T-ITS: https://arxiv.org/abs/2306.10159
- DriveCLIP code: https://github.com/zahid-isu/DriveCLIP
- VideoCLIP code of Hasan et al.: https://github.com/jiajingchen113322/DeepInsigth
- SAM-DD site: https://yanghh.io/SAM-DD/
- SAM-DD files: https://huggingface.co/datasets/yhh-ucla/SAM-DD
- Eraqi et al. 2019, Journal of Advanced Transportation: https://arxiv.org/abs/1901.09097
- AUC dataset page: https://heshameraqi.github.io/distraction_detection
- Ma and Wang 2022, ViT-DD: https://arxiv.org/abs/2209.09178
- ViT-DD code: https://github.com/PurdueDigitalTwin/ViT-DD
- StateFarm data and rules: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection
- Drive&Act site: https://driveandact.com/
- Martin et al. 2019, ICCV: https://openaccess.thecvf.com/content_ICCV_2019/papers/Martin_DriveAct_A_Multi-Modal_Dataset_for_Fine-Grained_Driver_Behavior_Recognition_in_ICCV_2019_paper.pdf
- Peng et al. 2022, IROS: https://arxiv.org/abs/2203.00927
- Wang et al. 2023, IEEE T-ITS, 100-Driver site and paper: https://100-driver.github.io
- Köpüklü et al. 2021, WACV: https://arxiv.org/abs/2009.14660
- DAD code: https://github.com/okankop/Driver-Anomaly-Detection
- Ma et al. 2022: https://arxiv.org/abs/2210.09441
- Chai et al. 2024, IEEE T-ITS: https://doi.org/10.1109/TITS.2024.3354506
