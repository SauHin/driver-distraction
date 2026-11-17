"""Speed of one model on one device, for the live target P8 of docs/PLAN.md.
A frame is a 640x480 RGB image, the size of StateFarm. Each frame goes through
`eval_transform` and the model, one frame at a time. The camera read is not part of it."""
import io
import statistics
import time

import torch
from PIL import Image

from ddd.data import eval_transform
from ddd.models import create


def measure(cfg, device, frames=50, warmup=10):
    """Processed frames per second, the median time of one frame, the parameters, and the size
    of the weights. The weights do not change the speed, so the model has no start weights."""
    model = create({**cfg, "weights": None}).to(device).eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    tf = eval_transform(cfg.get("input_size", 224))
    img = Image.new("RGB", (640, 480), (90, 120, 150))
    times = []
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(warmup + frames):
            t0 = time.perf_counter()
            model(tf(img).unsqueeze(0).to(device)).argmax(1).item()  # .item() waits for the GPU
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return {"device": str(device), "frames": frames, "frames_per_second": frames / sum(times),
            "median_ms_per_frame": 1000 * statistics.median(times),
            "parameters": sum(p.numel() for p in model.parameters()),
            "weights_mb": buf.tell() / 2**20,
            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None}
