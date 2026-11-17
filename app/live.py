"""Live driver distraction detection from a camera, a video file, or an image file. Press q to stop.
A demonstration only. It is not a safety system for the road (docs/PLAN.md section 10)."""
import argparse
import csv
import time
from pathlib import Path

import cv2
import torch

from ddd.data import CODES, eval_transform
from ddd.live import NAMES, Smoother, class_scores, load_model, prepare

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def frames(source):
    """(BGR frame, seconds) of a camera number, a video file, or one image file. The seconds are the
    clock time for a camera and the position in the file for a video, so a fast laptop does not
    shorten the alert time of a video."""
    if Path(source).suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"cannot read {source}")
        yield img, 0.0
        return
    camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if camera else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame, time.time() if camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    finally:
        cap.release()


def draw(frame, mean, k, distracted, alert, fps):
    """The status, the class, the speed, and a bar for each class score, on a copy of the frame."""
    out = frame.copy()
    color = (0, 0, 255) if distracted else (0, 160, 0)  # BGR: red or green
    cv2.rectangle(out, (0, 0), (out.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(out, "DISTRACTED" if distracted else "SAFE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"{CODES[k]} {NAMES[k]} {mean[k]:.2f}   {fps:.1f} FPS", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, s in enumerate(mean):
        y = 80 + 20 * i
        cv2.rectangle(out, (10, y), (10 + int(150 * s), y + 14), (200, 200, 200), -1)
        cv2.putText(out, f"{CODES[i]} {NAMES[i]}", (166, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    if alert:
        h = out.shape[0]
        cv2.rectangle(out, (0, h - 50), (out.shape[1], h), (0, 0, 255), -1)
        cv2.putText(out, "ALERT: eyes on the road", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", default="0", help="camera number, video file, or image file")
    p.add_argument("--run", default="runs/final_deit3_small_noflip", help="run folder with config.json and last.pt")
    p.add_argument("--window", type=int, default=5, help="number of frames in the mean of the class scores")
    p.add_argument("--alert-seconds", type=float, default=2.0, help="seconds of a distracted status before the alert")
    p.add_argument("--log", default=None, help="CSV log file (default: runs/live_logs/live_<time>.csv)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-show", action="store_true", help="no window: write the log only")
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args()

    device = torch.device(a.device)
    model, cfg = load_model(a.run, device)
    tf = eval_transform(cfg.get("input_size", 224))
    smoother = Smoother(a.window, a.alert_seconds)
    log_path = Path(a.log or f"runs/live_logs/live_{time.strftime('%Y%m%dT%H%M%S')}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fps, last = 0.0, time.perf_counter()
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "frame", "class", "name", "score", "distracted", "alert", "fps"] + CODES)
        for n, (frame, t) in enumerate(frames(a.source)):
            if a.max_frames is not None and n >= a.max_frames:
                break
            mean, k, distracted, alert = smoother.update(class_scores(model, prepare(frame, tf), device), t)
            now = time.perf_counter()
            fps = 1 / (now - last) if n == 0 else 0.9 * fps + 0.1 / (now - last)
            last = now
            w.writerow([f"{t:.3f}", n, CODES[k], NAMES[k], f"{mean[k]:.4f}", int(distracted), int(alert),
                        f"{fps:.2f}"] + [f"{s:.4f}" for s in mean])
            if not a.no_show:
                cv2.imshow("Driver distraction (demo)", draw(frame, mean, k, distracted, alert, fps))
                one_image = Path(a.source).suffix.lower() in IMAGE_SUFFIXES
                if cv2.waitKey(0 if one_image else 1) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
