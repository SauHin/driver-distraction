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
from ddd.metrics import binary_scores, bootstrap_accuracy, class_scores, confusion, scores  # noqa: E402
from ddd.models import create, load_weights, set_trainable  # noqa: E402
from ddd.train import optimizer, pending_runs, run  # noqa: E402


def test_scores_from_confusion():
    m = confusion([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], n=3)
    s = scores(m)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["class_mean_accuracy"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    assert s["macro_f1"] == pytest.approx((2 / 4 + 4 / 5 + 0) / 3)
    c = class_scores(m)
    assert [x["f1"] for x in c] == pytest.approx([2 / 4, 4 / 5, 0])
    assert [x["precision"] for x in c] == pytest.approx([1 / 2, 2 / 3, 0])
    assert [x["recall"] for x in c] == pytest.approx([1 / 2, 1, 0])
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


def test_create_sets_the_start_value_of_the_layer_scales():
    cfg = {"model": "deit3_small_patch16_224", "weights": None, "input_size": 32}
    assert torch.allclose(create(cfg).blocks[0].ls1.gamma, torch.tensor(1e-6))
    assert torch.allclose(create({**cfg, "init_values": 1e-4}).blocks[0].ls1.gamma, torch.tensor(1e-4))


def test_train_transform_with_no_augmentation_equals_eval_transform():
    from ddd.data import train_transform
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8))
    none = train_transform(32, color_jitter=0, three_augment=False, flip=False, random_crop=False)
    assert torch.equal(none(img), eval_transform(32)(img))


def test_measure_counts_the_frames_after_the_warm_up():
    from ddd.speed import measure
    s = measure({"model": "resnet18", "input_size": 32}, torch.device("cpu"), frames=3, warmup=1)
    assert s["frames"] == 3 and s["frames_per_second"] > 0 and s["median_ms_per_frame"] > 0
    assert s["parameters"] > 0 and s["weights_mb"] > 0 and s["gpu_peak_memory_mb"] is None


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
    with open(tmp_path / "out" / "tiny_val_drivers.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["driver"], r["images"]) for r in rows] == [("p002", "4")]
    assert float(rows[0]["color_accuracy"]) == pytest.approx(f["color"]["val_accuracy"])
    with open(tmp_path / "out" / "tiny_val_confusion_gray.csv", newline="", encoding="utf-8") as fh:
        assert sum(int(v) for r in list(csv.reader(fh))[1:] for v in r[1:]) == 4
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


def test_change_image_keeps_the_size_and_changes_the_pixels():
    from ddd.robust import CHANGES, change_image
    img = Image.fromarray(np.random.default_rng(0).integers(60, 200, (48, 64, 3), dtype=np.uint8))
    for kind, strengths in CHANGES.items():
        out = change_image(img, kind, strengths[-1], np.random.default_rng(1))
        assert out.size == img.size and out.mode == "RGB"
        assert not np.array_equal(np.asarray(out), np.asarray(img)), kind
    dark = np.asarray(change_image(img, "brightness", 0.2, None), dtype=float)
    assert dark.mean() < np.asarray(img, dtype=float).mean()
    covered = np.asarray(change_image(img, "cover", 0.3, np.random.default_rng(2)))
    assert (covered == 128).all(axis=2).mean() == pytest.approx(0.3, abs=0.03)
    with pytest.raises(ValueError):
        change_image(img, "rain", 1.0, None)


def test_evaluate_changes_opens_the_lock_one_time(tmp_path):
    from ddd.robust import evaluate_changes
    cfg = _tiny_data(tmp_path / "sf")
    split = {"name": "tiny", "dev": ["p001"], "test": ["p002"], "folds": []}
    (tmp_path / "sf" / "split.json").write_text(json.dumps(split), encoding="utf-8")
    run_dir = run({**cfg, "fold": None}, tmp_path / "runs" / "tiny")
    log = tmp_path / "log.csv"
    out = evaluate_changes(run_dir, "unit test", tmp_path / "out", log, changes={"blur": [1.0], "cover": [0.2]})
    assert [(r["change"], r["strength"]) for r in out["changes"]] == [("none", None), ("blur", 1.0), ("cover", 0.2)]
    assert all(r["images"] == 4 for r in out["changes"])
    with open(log, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_bootstrap_accuracy_over_drivers():
    lo, hi = bootstrap_accuracy([0.8, 0.8, 0.8], [10, 20, 30], reps=200)
    assert lo == pytest.approx(0.8) and hi == pytest.approx(0.8)
    lo, hi = bootstrap_accuracy([0.5, 1.0], [100, 100], reps=2000)
    assert 0.5 <= lo < 0.75 < hi <= 1.0
