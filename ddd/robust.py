"""Step 5 of docs/PLAN.md section 9: the final model on changed test images (decision D20).
Each change has 3 strengths (decision P20). The change acts on the 640x480 image, before
`eval_transform`, as a change of the camera or the light would."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ddd.data import StateFarmImages

# Decision P20. Brightness and contrast: the factor of the image (1 is no change).
# Blur: the radius in pixels. Noise: the standard deviation on the 0 to 255 scale.
# Cover: the share of the image area under a gray square at a random place.
CHANGES = {"brightness": [0.6, 0.4, 0.2], "contrast": [0.6, 0.4, 0.2], "blur": [1.0, 2.0, 4.0],
           "noise": [10.0, 20.0, 40.0], "cover": [0.1, 0.2, 0.3]}


def change_image(img, kind, strength, rng):
    """`img` with one change. `rng` gives the noise and the place of the cover."""
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(strength)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(strength)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(strength))
    a = np.asarray(img, dtype=np.float64)
    if kind == "noise":
        return Image.fromarray(np.clip(a + rng.normal(0, strength, a.shape), 0, 255).astype(np.uint8))
    if kind == "cover":
        side = int(round((strength * img.width * img.height) ** 0.5))
        x, y = rng.integers(0, img.width - side + 1), rng.integers(0, img.height - side + 1)
        a[y:y + side, x:x + side] = 128
        return Image.fromarray(a.astype(np.uint8))
    raise ValueError(f"unknown change {kind!r}")


class ChangedImages(StateFarmImages):
    """StateFarm images with one change. Image i always gets the same noise and cover place."""

    def __init__(self, rows, root, transform, kind, strength):
        super().__init__(rows, root, transform)
        self.kind, self.strength = kind, strength

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = change_image(im.convert("RGB"), self.kind, self.strength, np.random.default_rng(i))
        return self.transform(img), self.targets[i]


def evaluate_changes(run_dir, reason, out_dir, log_path=None, changes=None):
    """Scores of the last epoch of a run on the test drivers, with no change and with each change.
    One call opens the test lock one time. Writes `<run>_test_changes.json`."""
    import torch
    from torch.utils.data import DataLoader

    from ddd.data import eval_transform
    from ddd.metrics import binary_scores, scores
    from ddd.models import create
    from ddd.splits import TEST_ACCESS_LOG, load_split
    from ddd.statefarm import load_rows
    from ddd.train import environment, evaluate

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    if ck["epoch"] != cfg["epochs"] - 1:  # check before the unlock, so a refusal writes no log row
        raise ValueError(f"last.pt holds epoch {ck['epoch']}, not the last epoch {cfg['epochs'] - 1}")
    rows = load_rows(Path(cfg["data_root"]) / "driver_imgs_list.csv", load_split(cfg["split"]), "test",
                     unlock_test=True, reason=reason, log_path=log_path or TEST_ACCESS_LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create({**cfg, "weights": None})
    model.load_state_dict(ck["model"])
    model.to(device)
    tf = eval_transform(cfg.get("input_size", 224))
    results = []
    plan = [("none", None)] + [(k, s) for k, ss in (changes or CHANGES).items() for s in ss]
    for kind, s in plan:
        ds = (StateFarmImages(rows, cfg["data_root"], tf) if kind == "none"
              else ChangedImages(rows, cfg["data_root"], tf, kind, s))
        dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
        m, _ = evaluate(model, dl, device)
        results.append({"change": kind, "strength": s, "images": int(m.sum()), **scores(m), **binary_scores(m)})
        print(json.dumps(results[-1]), flush=True)
    out = {"run": run_dir.name, "epoch": ck["epoch"], "reason": reason, "changes": results,
           "environment": environment()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_dir.name}_test_changes.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
