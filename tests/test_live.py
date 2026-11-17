import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ddd.data import eval_transform  # noqa: E402
from ddd.live import Smoother, prepare  # noqa: E402


def test_prepare_equals_eval_transform_of_the_rgb_image():
    bgr = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    tf = eval_transform(32)
    assert torch.equal(prepare(bgr, tf), tf(Image.fromarray(bgr[:, :, ::-1].copy())))
    assert not torch.equal(prepare(bgr, tf), tf(Image.fromarray(bgr)))


def _scores(k):
    s = np.zeros(10)
    s[k] = 1.0
    return s


def test_smoother_alerts_after_a_continuous_distraction():
    sm = Smoother(window=1, alert_seconds=2.0)
    assert sm.update(_scores(6), 0.0)[2:] == (True, False)
    assert sm.update(_scores(6), 1.0)[2:] == (True, False)
    assert sm.update(_scores(6), 2.0)[2:] == (True, True)
    # A safe frame resets the time of the distraction.
    assert sm.update(_scores(0), 3.0)[2:] == (False, False)
    assert sm.update(_scores(6), 4.0)[2:] == (True, False)


def test_smoother_takes_the_mean_of_the_window():
    sm = Smoother(window=3, alert_seconds=2.0)
    for k in (9, 0, 0):
        mean, k_out, _, _ = sm.update(_scores(k), 0.0)
    assert mean[0] == pytest.approx(2 / 3) and k_out == 0
    mean, k_out, _, _ = sm.update(_scores(9), 0.0)
    assert mean[9] == pytest.approx(1 / 3) and k_out == 0
