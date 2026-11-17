# Cost of the final model

Date: 2026-11-17.
Sources: `speed_laptop.json` in this folder, written by `scripts/measure_speed.py`, and `final_deit3_small_noflip/history.csv`.
This is step 6 of `docs/PLAN.md` section 9.

## Values

| Item | Value |
|---|---|
| Model. | DeiT-III Small, `deit3_small_patch16_224.fb_in1k` of `timm`. |
| Parameters. | 21678346. |
| Size of the weights. | 82.76 MB. |
| Training on the Colab L4, sum of the epoch times. | 1611.57 seconds for 20 epochs. |
| Speed on the laptop GPU (RTX 3050 6 GB). | 62.59 frames per second, median 16.20 ms per frame. |
| Peak GPU memory of the speed test. | 95.74 MB. |
| Speed on the laptop CPU (Ryzen 7 8845HS). | 16.30 frames per second, median 59.58 ms per frame. |

- A frame is a 640x480 image through `eval_transform` and the model, one frame at a time. The camera read is not part of it.
- The laptop ran on its charger. Each speed comes from 50 frames after 10 warm-up frames.
- Both speeds reach the live target of 5 frames per second (decision P8).
- The size counts the weights of the model only. The file `last.pt` also holds the state of the optimizer.

## Not checked

- The CPU speed of DeiT-III Small in `results/development/e1.md` is 18.52 frames per second. The speed of this run is lower. The cause is not known.
- The memory of the CPU.
- The speed with the camera read and the display of the application.
