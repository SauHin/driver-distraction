# Check of the ViT-DD recipe

Date: 2026-11-15.
Section 7 of `docs/PLAN.md` asks for this check before R1.

## Sources

- The public code: `datasets/distrated_driver.py`, `lib/augment.py`, `models/vitdd.py`, and `train.py` of https://github.com/PurdueDigitalTwin/ViT-DD.
- The public config: `configs/vitdd_sfddd_sbd.yaml`.
- The paper: arXiv 2209.09178, version 4, sections III and IV, and tables I and II.

## Training transforms of the config

The config sets `src: true`, `three_augment: true`, and `color_jitter: 0.3`. The code then applies these steps in order:

1. Resize with bicubic interpolation. The short side becomes 224 pixels. The resize keeps the aspect ratio.
2. Random crop of 224 by 224 pixels, with a padding of 4 pixels by reflection.
3. Random horizontal flip with a chance of 0.5.
4. One of three steps, by random choice: gray scale, solarization, or Gaussian blur with a radius from 0.1 to 2.
5. Color jitter of 0.3 for brightness, contrast, and saturation.
6. Normalize with the ImageNet mean and standard deviation of `timm`.

## Evaluation transforms

1. Resize with bicubic interpolation. The short side becomes 224 pixels.
2. Center crop of 224 by 224 pixels. The StateFarm images are 640 by 480 pixels, so this crop cuts the left and right edges.
3. Normalize as above.

## Model and training in the code

- The config uses the class `ViTDDLM`. This class has the driver head and the emotion head.
- Trained parameters of `ViTDDLM`: each parameter with `.attn.` in its name, the two heads, the two position embeddings, and the patch embedding of the face.
- The patch embedding of the driver image, the MLP layers, the norm layers, and the class token do not train.
- Loss: `CrossEntropyLoss` with no label smoothing. The config gives `smoothing: 0.1`, but `ViTDDLM` does not use it.
- Mixup and CutMix: not in `ViTDDLM`.
- Schedule: the cosine scheduler of `timm`, with one step for each epoch. The warm-up starts at 0.000001 for 5 epochs. The end value is 0.
- Drop path: 0.
- Batch size: 128 in the config. The paper text gives 256.
- Validation: the config reads the test list `val_list_d_01.csv` as the validation list. The code scores the test drivers after each epoch.
- Accuracy: `torchmetrics.Accuracy()` with its default. This is the share of correct images over all images.

## The single-task class of the code

- `train.py` also holds the class `VisionTransformerLM`. It has one head and no face input.
- Trained parameters: each `.attn.` parameter, the head, the position embedding, and the patch embedding.
- Mixup 0.8 and CutMix 1.0 are on by default. Label smoothing acts only through Mixup.
- No public config uses this class for StateFarm.

## The standard ViT of the paper (R1)

- Table II gives "Standard ViT" on StateFarm by driver: accuracy 0.9036.
- The paper gives no recipe for this run. Section IV-C describes the recipe of ViT-DD only.
- The paper says that only the attention layers train, as Touvron et al. suggest.
- The paper names a random crop with a horizontal flip, and 3-Augment for StateFarm.
- The paper says that the accuracy treats all classes equally. The code computes the share of correct images.
- The paper says that 18 of 28 drivers train and 6 test. The public lists hold 18 and 8 of 26 drivers.

## Differences from section 7 of the plan

- The plan names label smoothing 0.1. The code of ViT-DD does not apply it.
- The plan names the attention layers, the heads, and the position embedding as trained parts. The single-task class also trains the patch embedding.
- The code flips the images. Four classes depend on the left or right hand, so experiment E3 stays in the plan.
