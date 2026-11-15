# Differences of R1 from the ViT-DD recipe

Run: `r1_vit_s42`, with `configs/r1_vit.json` and seed 42.
The recipe is the public config `vitdd_sfddd_sbd.yaml` with the class `ViTDDLM`, after decision D10.
Section 7 of `docs/PLAN.md` asks for this list. `results/vitdd_recipe_check.md` holds the sources.

## Differences

| Item | ViT-DD recipe | R1 |
|---|---|---|
| Inputs. | The driver image and a face crop of 32 pixels. | The driver image only. Decision D10. |
| Heads and loss. | A driver head and an emotion head. Loss weights 0.999 and 0.001. | One driver head. One cross entropy loss. Decision D10. |
| Trained parts. | Attention layers, two heads, two position embeddings, and the face patch embedding. | Attention layers, the head, and the position embedding. These are the parts of the driver branch. |
| Training code. | Lightning 1.7.7 and an older `timm`. | Own loop with PyTorch and `timm` 1.0.31. `environment.json` gives the versions. |
| Random numbers. | `seed_everything(42)` and a `RandomSampler`. | Seed 42 and a loader with its own generator. The image order and the augmentation are not the same as in the public code. |
| Validation. | The config scores the test list after each epoch. | No validation part. Only the last epoch is scored on the test drivers. Decision P9. |
| GPU. | Not given for the training. | NVIDIA L4. `environment.json` gives the name. |
| Batch size. | 128 in the config. The paper text gives 256. | 128. |

## Points that are the same

- Model `deit3_base_patch16_224` with the DeiT-III ImageNet-1k weights from the link of `timm` 0.6.12.
- 20 epochs, AdamW, learning rate 0.0003, weight decay 0.1, 5 warm-up epochs from 0.000001, a cosine curve to 0.
- Scheduler step after each epoch, as in Lightning 1.7.7. Config key `sched_after_epoch`.
- Weight decay on the position embedding, as in `ViTDDLM`. Config key `wd_skip_list`.
- No label smoothing, no Mixup, no drop path, 32-bit numbers, no gradient clipping.
- Training transforms: resize, random crop with padding 4, flip, 3-Augment, color jitter 0.3.
- Evaluation transforms: resize to 224 pixels, center crop.
- The last part batch of each epoch is dropped.
- The public split lists: 18 training drivers and 8 test drivers.

## Events of the run

- A first attempt gave the position embedding no weight decay. The owner stopped it after epoch 10.
- Its files are in the Drive folder `runs/void_r1_vit_s42_pos_embed_wd0`. No test driver was scored with it.

## Points not checked

- The weights of the link were not compared with the weights `fb_in1k` of Hugging Face.
- The exact `timm` version of the public code is not known. The scheduler check used `timm` 0.6.12.
