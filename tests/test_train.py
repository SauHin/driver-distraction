import csv
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from ddd.metrics import confusion, scores  # noqa: E402
from ddd.models import load_weights, set_trainable  # noqa: E402
from ddd.train import run  # noqa: E402


def test_scores_from_confusion():
    m = confusion([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], n=3)
    s = scores(m)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["class_mean_accuracy"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert s["macro_f1"] == pytest.approx((2 / 4 + 4 / 5 + 0) / 3)


def test_trainable_parts_follow_vitdd():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, img_size=32)
    set_trainable(m, "attn_head_pos")
    on = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "pos_embed" in on and "head.weight" in on and "blocks.0.attn.qkv.weight" in on
    assert not any(n.startswith("patch_embed") or ".mlp." in n or "norm" in n for n in on)
    assert "cls_token" not in on


def test_load_weights_renames_layer_scales(tmp_path):
    kw = dict(pretrained=False, num_classes=10, img_size=32)
    src = timm.create_model("deit3_small_patch16_224", **kw)
    sd = {k.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2"): v
          for k, v in src.state_dict().items()}
    torch.save({"model": sd}, tmp_path / "w.pth")
    dst = timm.create_model("deit3_small_patch16_224", **kw)
    load_weights(dst, tmp_path / "w.pth")
    assert torch.equal(dst.blocks[0].ls1.gamma, src.blocks[0].ls1.gamma)
    assert torch.equal(dst.pos_embed, src.pos_embed)


def _tiny_data(root):
    rows = []
    for d in ("p001", "p002"):
        for c in ("c0", "c1"):
            for i in range(2):
                name = f"{d}_{c}_{i}.png"
                folder = root / "imgs" / "train" / c
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((24, 32, 3), 40 * i + (100 if c == "c1" else 0), np.uint8)
                                ).save(folder / name)
                rows.append((d, c, name))
    with open(root / "driver_imgs_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "classname", "img"])
        w.writerows(rows)
    split = {"name": "tiny", "dev": ["p001", "p002"], "test": ["p009"],
             "folds": [{"train": ["p001"], "val": ["p002"]}]}
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return {"name": "tiny", "split": str(root / "split.json"), "fold": 0, "data_root": str(root),
            "model": "vit_tiny_patch16_224", "weights": None, "trainable": "attn_head_pos",
            "input_size": 32, "epochs": 2, "batch_size": 2, "lr": 0.001, "weight_decay": 0.1,
            "warmup_epochs": 1, "warmup_lr": 1e-6, "min_lr": 0.0, "label_smoothing": 0.0,
            "color_jitter": 0.3, "three_augment": True, "flip": True, "seed": 42, "num_workers": 0}


def _epochs(run_dir):
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
        return [int(r["epoch"]) for r in csv.DictReader(f)]


def test_run_writes_folder_and_continues(tmp_path):
    cfg = _tiny_data(tmp_path / "sf")
    run_dir = run(cfg, tmp_path / "runs" / "tiny")
    for name in ("config.json", "environment.json", "history.csv", "metrics.json", "last.pt"):
        assert (run_dir / name).exists()
    assert _epochs(run_dir) == [0, 1]
    # A disconnect after the checkpoint of epoch 0 and the history row of epoch 1.
    ck = torch.load(run_dir / "last.pt", weights_only=False)
    ck["epoch"] = 0
    torch.save(ck, run_dir / "last.pt")
    run(cfg, run_dir)
    assert _epochs(run_dir) == [0, 1]
    with pytest.raises(ValueError):
        run({**cfg, "lr": 0.5}, run_dir)
