"""Training run with one config file. The run folder holds the config, the environment,
the history of each epoch, the metrics, and a checkpoint after each epoch."""
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
from timm.optim import create_optimizer_v2, param_groups_weight_decay
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from ddd.data import StateFarmImages, eval_transform, train_transform
from ddd.metrics import confusion, scores
from ddd.models import create
from ddd.splits import _git_commit, load_split
from ddd.statefarm import load_rows

# train_accuracy is the share of correct augmented images during the epoch.
HISTORY_FIELDS = ["epoch", "lr", "train_loss", "train_accuracy", "train_images", "epoch_seconds",
                  "images_per_second", "gpu_peak_memory_mb", "val_loss", "val_accuracy",
                  "val_class_mean_accuracy", "val_macro_f1"]


def environment():
    cuda = torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else "none",
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__,
            "cuda": torch.version.cuda, "git_commit": _git_commit()}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parts(cfg):
    """Train rows and validation rows. The config never names the test part."""
    split = load_split(cfg["split"])
    csv_path = Path(cfg["data_root"]) / "driver_imgs_list.csv"
    if cfg.get("fold") is None:
        return load_rows(csv_path, split, "dev"), []
    return (load_rows(csv_path, split, "train", cfg["fold"]),
            load_rows(csv_path, split, "val", cfg["fold"]))


def optimizer(model, cfg):
    """AdamW of timm. Biases, 1-D weights, and the `no_weight_decay` list of the model get no
    weight decay. `wd_skip_list` replaces that list. ViT-DD gives timm a wrapper module with
    no such list, so its `pos_embed` decays: `"wd_skip_list": []`."""
    params = model
    if "wd_skip_list" in cfg:
        params = param_groups_weight_decay(model, cfg["weight_decay"],
                                           no_weight_decay_list=cfg["wd_skip_list"])
    return create_optimizer_v2(params, opt="adamw", lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def evaluate(model, loader, device, loss_fn=None):
    """The confusion matrix, and the mean loss with `loss_fn` (None with no `loss_fn`)."""
    model.eval()
    targets, preds, loss_sum = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device, non_blocking=True))
            if loss_fn is not None:
                loss_sum += loss_fn(out, y.to(device)).item() * len(y)
            preds.append(out.argmax(1).cpu())
            targets.append(y)
    t = torch.cat(targets)
    loss = loss_sum / len(t) if loss_fn is not None else None
    return confusion(t.numpy(), torch.cat(preds).numpy()), loss


def run(cfg, run_dir, max_steps=None):
    """Train, or continue a run from its last checkpoint.

    `max_steps` stops each epoch early. Use it only for a speed measurement.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    elif json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != cfg:
        raise ValueError(f"{run_dir} holds a run with another config")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n",
                                              encoding="utf-8")
    _seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, val_rows = _parts(cfg)
    size = cfg.get("input_size", 224)
    train_ds = StateFarmImages(train_rows, cfg["data_root"],
                               train_transform(size, cfg["color_jitter"], cfg["three_augment"],
                                               cfg["flip"], cfg.get("random_crop", True)))
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=cfg["num_workers"], pin_memory=True, generator=g,
                          persistent_workers=cfg["num_workers"] > 0)
    val_dl = None
    if val_rows:
        val_dl = DataLoader(StateFarmImages(val_rows, cfg["data_root"], eval_transform(size)),
                            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                            pin_memory=True)
    model = create(cfg).to(device)
    opt = optimizer(model, cfg)
    sched = CosineLRScheduler(opt, t_initial=cfg["epochs"], lr_min=cfg["min_lr"],
                              warmup_t=cfg["warmup_epochs"], warmup_lr_init=cfg["warmup_lr"])
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    start = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng_cpu"])
        g.set_state(ck["rng_loader"])
        start = ck["epoch"] + 1
    # Keep the rows of the epochs in the checkpoint. A row after the checkpoint is from a lost epoch.
    history = run_dir / "history.csv"
    kept = []
    if history.exists():
        with open(history, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
    with open(history, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, HISTORY_FIELDS)
        w.writeheader()
        w.writerows(kept)

    for epoch in range(start, cfg["epochs"]):
        # ViT-DD (Lightning 1.7) steps the scheduler after each epoch: epoch k trains with the
        # value of k - 1, and epoch 0 with the start value of the scheduler.
        t = epoch - 1 if cfg.get("sched_after_epoch") else epoch
        if t >= 0:
            sched.step(t)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl):
            if max_steps is not None and step >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / max(n, 1),
               "train_accuracy": correct / max(n, 1), "train_images": n, "epoch_seconds": round(seconds, 2),
               "images_per_second": round(n / seconds, 2) if seconds else 0,
               "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20)
               if device.type == "cuda" else 0}
        if val_dl is not None:
            m, row["val_loss"] = evaluate(model, val_dl, device, loss_fn)
            row.update({f"val_{k}": v for k, v in scores(m).items()})
        with open(history, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, HISTORY_FIELDS).writerow(row)
        tmp = run_dir / "last.pt.tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "rng_cpu": torch.get_rng_state(), "rng_loader": g.get_state()}, tmp)
        tmp.replace(ckpt_path)
        print(json.dumps(row), flush=True)
        if epoch == cfg["epochs"] - 1:
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return run_dir


def pending_runs(queue_path, runs_dir):
    """(config path, fold, run name) of each queue line with no `metrics.json` in its run folder.
    A line holds a config path and a fold. Text after # is a comment."""
    out = []
    for line in Path(queue_path).read_text(encoding="utf-8").splitlines():
        parts = line.split("#")[0].split()
        if parts:
            name = f"{Path(parts[0]).stem}_f{parts[1]}"
            if not (Path(runs_dir) / name / "metrics.json").exists():
                out.append((parts[0], int(parts[1]), name))
    return out
