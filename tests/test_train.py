import csv
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: E402

from ddd.data import eval_transform  # noqa: E402
from ddd.development import gray_scores  # noqa: E402
from ddd.evaluate import evaluate_run  # noqa: E402
from ddd.metrics import binary_scores, confusion, scores  # noqa: E402
from ddd.models import create, load_weights, set_trainable  # noqa: E402
from ddd.train import optimizer, pending_runs, run  # noqa: E402


def test_scores_from_confusion():
    m = confusion([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], n=3)
    s = scores(m)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["class_mean_accuracy"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert s["macro_f1"] == pytest.approx((2 / 4 + 4 / 5 + 0) / 3)
    # Distracted: 2 found of 3, and 1 safe image called distracted.
    b = binary_scores(m)
    assert b["distracted_precision"] == pytest.approx(2 / 3)
    assert b["distracted_recall"] == pytest.approx(2 / 3)
    assert b["distracted_f1"] == pytest.approx(2 / 3)


def test_trainable_parts_follow_vitdd():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, img_size=32)
    set_trainable(m, "attn_head_pos")
    on = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "pos_embed" in on and "head.weight" in on and "blocks.0.attn.qkv.weight" in on
    assert not any(n.startswith("patch_embed") or ".mlp." in n or "norm" in n for n in on)
    assert "cls_token" not in on


def test_wd_skip_list_decays_pos_embed():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, img_size=32)
    set_trainable(m, "attn_head_pos")
    cfg = {"lr": 0.001, "weight_decay": 0.1}
    for extra, want in (({}, 0.0), ({"wd_skip_list": []}, 0.1)):
        groups = optimizer(m, {**cfg, **extra}).param_groups
        assert [g["weight_decay"] for g in groups if any(p is m.pos_embed for p in g["params"])] == [want]
        assert [g["weight_decay"] for g in groups if any(p is m.head.bias for p in g["params"])] == [0.0]


def test_create_cnn_takes_no_img_size():
    m = create({"model": "resnet18", "weights": None, "input_size": 32})
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_pending_runs_skips_complete_runs(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text("# config fold\nconfigs/b0.json 0\n\nconfigs/b0.json 1  # second\n", encoding="utf-8")
    (tmp_path / "runs" / "b0_f0").mkdir(parents=True)
    (tmp_path / "runs" / "b0_f0" / "metrics.json").write_text("{}", encoding="utf-8")
    assert pending_runs(q, tmp_path / "runs") == [("configs/b0.json", 1, "b0_f1")]


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
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
        last = list(csv.DictReader(f))[-1]
    assert 0 <= float(last["train_accuracy"]) <= 1 and float(last["val_loss"]) > 0
    # A disconnect after the checkpoint of epoch 0 and the history row of epoch 1.
    ck = torch.load(run_dir / "last.pt", weights_only=False)
    ck["epoch"] = 0
    torch.save(ck, run_dir / "last.pt")
    run(cfg, run_dir)
    assert _epochs(run_dir) == [0, 1]
    with pytest.raises(ValueError):
        run({**cfg, "lr": 0.5}, run_dir)


def test_evaluate_scores_last_epoch_only(tmp_path):
    cfg = _tiny_data(tmp_path / "sf")
    split = {"name": "tiny", "dev": ["p001"], "test": ["p002"], "folds": []}
    (tmp_path / "sf" / "split.json").write_text(json.dumps(split), encoding="utf-8")
    run_dir = run({**cfg, "fold": None}, tmp_path / "runs" / "tiny")
    log, out = tmp_path / "log.csv", tmp_path / "out"
    m = evaluate_run(run_dir, "unit test", out, log)
    assert m["epoch"] == 1 and m["test_drivers"] == ["p002"] and m["images"] == 4
    assert "distracted_f1" in m and "macro_f1" in m
    with open(out / "tiny_test_confusion.csv", newline="", encoding="utf-8") as f:
        assert sum(int(v) for r in list(csv.reader(f))[1:] for v in r[1:]) == 4
    with open(out / "tiny_test_drivers.csv", newline="", encoding="utf-8") as f:
        assert [r["driver"] for r in csv.DictReader(f)] == ["p002"]
    # A checkpoint of an earlier epoch is refused before the unlock.
    ck = torch.load(run_dir / "last.pt", weights_only=False)
    ck["epoch"] = 0
    torch.save(ck, run_dir / "last.pt")
    with pytest.raises(ValueError):
        evaluate_run(run_dir, "unit test", out, log)
    with open(log, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_gray_scores_repeat_the_color_scores_of_the_run(tmp_path):
    cfg = _tiny_data(tmp_path / "sf")
    run(cfg, tmp_path / "runs" / "tiny_f0")
    f = gray_scores(tmp_path / "runs", "tiny", n_folds=1, out_dir=tmp_path / "out")["folds"][0]
    assert f["images"] == 4 and 0 <= f["gray"]["val_accuracy"] <= 1
    assert f["color"]["val_accuracy"] == pytest.approx(f["file"]["val_accuracy"])
    assert f["color"]["val_loss"] == pytest.approx(f["file"]["val_loss"])
    # The gray step gives 3 equal channels before the normalization.
    x = eval_transform(32, gray=True)(Image.new("RGB", (40, 32), (200, 50, 10)))
    raw = x * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[1], raw[2], atol=1e-5)


def test_sched_after_epoch_follows_lightning(tmp_path):
    cfg = _tiny_data(tmp_path / "sf")
    run_dir = run({**cfg, "sched_after_epoch": True}, tmp_path / "runs" / "tiny")
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
        lrs = [float(r["lr"]) for r in csv.DictReader(f)]
    # Epoch 0 has the start value, and epoch 1 has the value of step 0. Both are the warm-up start.
    assert lrs == pytest.approx([cfg["warmup_lr"]] * 2)
