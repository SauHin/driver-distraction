"""Live prediction for the application of docs/PLAN.md section 10. A frame goes through the
steps of `eval_transform`. The class scores are smoothed over the last frames."""
import bisect
import collections
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ddd.data import CODES
from ddd.models import create

# The class names of docs/PLAN.md section 3.4, in the order of the StateFarm codes c0 to c9.
NAMES = ["safe driving", "texting, right hand", "phone call, right hand", "texting, left hand",
         "phone call, left hand", "radio", "drinking", "reaching behind", "hair and makeup",
         "talking to passenger"]


# Decision D21: in a right-hand-drive car, the app flips each frame. A flip turns the right hand of the
# driver into a left hand in the image, so the scores of c1 and c3, and of c2 and c4, change places.
HAND_SWAP = [0, 3, 4, 1, 2, 5, 6, 7, 8, 9]


def prepare(frame_bgr, tf, mirror=False):
    """An OpenCV frame (BGR, uint8) to the input of the model. OpenCV gives BGR, the model takes RGB.
    `mirror` flips the frame left to right first (decision D21)."""
    rgb = frame_bgr[:, ::-1, ::-1] if mirror else frame_bgr[:, :, ::-1]
    return tf(Image.fromarray(np.ascontiguousarray(rgb)))


def load_model(run_dir, device):
    """The model of the last checkpoint of a run folder, and the config of the run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    model = create({**cfg, "weights": None})  # the checkpoint holds all weights
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.inference_mode()
def class_scores(model, x, device):
    """The softmax scores of the 10 classes for one prepared frame."""
    return torch.softmax(model(x.unsqueeze(0).to(device)), 1)[0].cpu().numpy()


class Smoother:
    """The mean class scores of the last `window` frames. The alert starts when the smoothed
    status stays "distracted" for `alert_seconds`."""

    def __init__(self, window=5, alert_seconds=2.0):
        self.scores = collections.deque(maxlen=window)
        self.alert_seconds = alert_seconds
        self.since = None

    def update(self, scores, now):
        """(mean scores, class index, distracted, alert) after the scores of a frame at time `now`."""
        self.scores.append(np.asarray(scores, dtype=np.float64))
        mean = np.mean(self.scores, axis=0)
        k = int(mean.argmax())
        distracted = k != 0  # decision P3: c0 is safe, c1 to c9 are distracted
        if not distracted:
            self.since = None
        elif self.since is None:
            self.since = now
        return mean, k, distracted, distracted and now - self.since >= self.alert_seconds


def demo_rows(rows, driver, per_class):
    """Up to `per_class` label rows of each class c0 to c9 of one driver, in class order."""
    out = []
    for code in CODES:
        out += [r for r in rows if r["subject"] == driver and r["classname"] == code][:per_class]
    return out


def true_class(schedule, t):
    """The class of a demo video at second `t`. `schedule` is a sorted list of (start second, class)."""
    i = bisect.bisect_right([s for s, _ in schedule], t) - 1
    return schedule[i][1] if i >= 0 else None
