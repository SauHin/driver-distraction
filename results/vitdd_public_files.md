# Public files of ViT-DD

Checked on 2026-11-14 by Claude Code (Opus 5.5) for task C5 of `docs/PLAN.md`.
Source: the README and the file tree of https://github.com/PurdueDigitalTwin/ViT-DD at commit `5416599`.
The sizes come from the headers of the servers and from the GitHub API.

## Result

| File | Host | Downloads | Note |
|---|---|---|---|
| `annotations/SFDDD/train_list_d_01.csv` and `val_list_d_01.csv`. | GitHub. | Yes. | The copy is in `data/external/vitdd/annotations/SFDDD/`, not in git. |
| `configs/vitdd_sfddd_sbd.yaml`. | GitHub. | Yes. | The copy is in `data/external/vitdd/configs/`, not in git. |
| Other files of the repository. | GitHub. | Not tried. | The GitHub API lists them. The AffectNet list has 17,577,711 bytes. |
| ViT-DD weights, StateFarm split by driver. | SharePoint of Purdue. | No. | The page says "Sorry, you cannot access this document". |
| ViT-DD weights, StateFarm split by image, and AUC. | SharePoint of Purdue. | No. | The same message. |
| Face crops and emotion labels for StateFarm and AUC. | SharePoint of Purdue. | No. | The same message. |
| ImageNet weights `deit_3_base_224_1k.pth`. | `dl.fbaipublicfiles.com`. | Yes, on 2026-11-14, after a yes of the owner. | 346,399,585 bytes. See the section "Downloaded weights". |
| ImageNet weights `timm/deit3_base_patch16_224.fb_in1k`. | Hugging Face. | Answers, not downloaded. | 346,357,466 bytes. It waits for the owner. |

## Effect on the plan

- R0 cannot run. No public ViT-DD weights download.
- R2 cannot run with the files of the authors. The face crops and the emotion labels do not download.
- Section 5 of the plan gives the rule for this case: G1 reads R1. This file is the message to the owner.
- R1 needs only the ImageNet weights. These two links answer.
- In timm 0.6.12, the name `deit3_base_patch16_224` points to `deit_3_base_224_1k.pth`.
- In the main branch of timm, `fb_in1k` is the first tag of the name. Its link is the same file.

## Downloaded weights

- File: `data/external/weights/deit3_base_224_1k/deit_3_base_224_1k.pth`, not in git.
- Source: https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth, the link of timm 0.6.12.
- Size: 346,399,585 bytes. This is equal to the size in the header of the server.
- SHA-256: `92d4e4ac874725a51c703dce67379d90df4af3805a8b27ec44546e580152003c`.
- Format: a PyTorch zip archive with 178 entries. The archive was read as a zip only, with no load in PyTorch.

## Not checked

- A sign-in with a Purdue account. The links can work for persons in the share list only.
- A published hash of the weights from Meta. The SHA-256 above has no reference to compare with.
- A load of the weights in timm. Load them with `weights_only=True` in PyTorch.
- The Hugging Face weight file. It is not downloaded.
- The default tag of the installed timm version. Load the model one time and read its config.
- An email to the authors for a new link. The owner decides.
