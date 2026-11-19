"""Figures for the report from the saved result files. Writes PNG files to results/figures/.
The figures show scores only. They hold no dataset image."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

DEV = Path("results/development")
FINAL = Path("results/final")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def history(run):
    with open(DEV / run / "history.csv", newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]


def curves(config, title, out):
    """Training and validation accuracy and loss of the 3 folds of one config."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(3):
        h = history(f"{config}_f{k}")
        ep = [r["epoch"] for r in h]
        ax_acc.plot(ep, [100 * r["train_accuracy"] for r in h], "--", color=COLORS[k], label=f"fold {k} train")
        ax_acc.plot(ep, [100 * r["val_accuracy"] for r in h], "-o", ms=3, color=COLORS[k], label=f"fold {k} validation")
        ax_loss.plot(ep, [r["train_loss"] for r in h], "--", color=COLORS[k])
        ax_loss.plot(ep, [r["val_loss"] for r in h], "-o", ms=3, color=COLORS[k])
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_acc.legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def tradeoff(out):
    """Mean validation macro F1 against the CPU speed on the laptop, for the E1 models."""
    speed = {r["config"]: r for r in json.loads((DEV / "speed_laptop.json").read_text())["rows"] if r["device"] == "cpu"}
    names = {"e1_mobilenetv3_large": "MobileNetV3-Large", "e1_efficientnet_b0": "EfficientNet-B0",
             "e1_resnet50": "ResNet-50", "e1_deit3_small": "DeiT-III Small", "e1_deit3_base": "DeiT-III Base"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cfg, name in names.items():
        f1 = 100 * json.loads((DEV / f"{cfg}_folds.json").read_text())["mean"]["val_macro_f1"]
        s = speed[cfg]
        ax.scatter(s["frames_per_second"], f1, s=20 + s["weights_mb"], color="#2a78d6", alpha=0.7)
        ax.annotate(f"{name}\n{s['weights_mb']:.0f} MB", (s["frames_per_second"], f1), fontsize=8,
                    xytext=(6, -4), textcoords="offset points")
    ax.axvline(5, color="#e34948", ls=":", label="live target, 5 FPS (P8)")
    ax.set(xlabel="CPU frames per second on the laptop", ylabel="Mean validation macro F1 (%)",
           title="E1: score against speed")
    ax.set_ylim(74, 90.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def robustness(out):
    """Test macro F1 of the final model for each change and strength."""
    rows = json.loads((FINAL / "final_deit3_small_noflip_test_changes.json").read_text())["changes"]
    base = 100 * rows[0]["macro_f1"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, kind in enumerate(["brightness", "contrast", "blur", "noise", "cover"]):
        r = [x for x in rows if x["change"] == kind]
        ax.plot([0, 1, 2, 3], [base] + [100 * x["macro_f1"] for x in r], "-o", label=kind,
                color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"][i])
    ax.set_xticks([0, 1, 2, 3], ["none", "level 1", "level 2", "level 3"])
    ax.set(ylabel="Test macro F1 (%)", title="Final model on changed test images (P20)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def drivers(out):
    """E5: mean validation macro F1 against the number of training drivers, with the value of each fold."""
    runs = {4: "e5_deit3_small_4drivers", 8: "e5_deit3_small_8drivers", 12: "e3_deit3_small_noflip"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    means = []
    for n, cfg in runs.items():
        s = json.loads((DEV / f"{cfg}_folds.json").read_text())
        means.append(100 * s["mean"]["val_macro_f1"])
        for k, f in enumerate(s["folds"]):
            ax.scatter(n, 100 * f["val_macro_f1"], color=COLORS[k], s=18, label=f"fold {k}" if n == 4 else None)
    ax.plot(list(runs), means, "-o", color="#1a1a1a", label="mean of the folds")
    ax.set_xticks(list(runs))
    ax.set(xlabel="Training drivers in each fold", ylabel="Validation macro F1 (%)",
           title="E5: number of training drivers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    curves("b0", "B0: DeiT-III Base, R1 recipe, 3 folds", out / "b0_curves.png")
    curves("e3_deit3_small_noflip", "Final recipe: DeiT-III Small, no flip, 3 folds", out / "final_recipe_curves.png")
    tradeoff(out / "e1_tradeoff.png")
    robustness(out / "robustness.png")
    drivers(out / "e5_drivers.png")
    print(sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
