import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ddd.data import eval_transform  # noqa: E402
from ddd.live import HAND_SWAP, Smoother, demo_rows, demo_run, prepare, true_class  # noqa: E402


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


def test_demo_rows_keep_the_class_order_of_one_driver():
    rows = [{"subject": d, "classname": c, "img": f"{d}_{c}_{i}"}
            for d in ("p001", "p002") for c in ("c9", "c0", "c1") for i in range(3)]
    out = demo_rows(rows, "p001", 2)
    assert [r["classname"] for r in out] == ["c0", "c0", "c1", "c1", "c9", "c9"]
    assert {r["subject"] for r in out} == {"p001"}


def test_true_class_follows_the_schedule():
    schedule = [(0.0, "c0"), (3.0, "c1"), (6.0, "c2")]
    assert [true_class(schedule, t) for t in (0.0, 2.9, 3.0, 9.0)] == ["c0", "c0", "c1", "c2"]
    assert true_class(schedule, -1.0) is None


def test_mirror_flips_the_frame_and_swaps_only_the_hand_classes():
    bgr = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    tf = eval_transform(32)
    assert torch.equal(prepare(bgr, tf, mirror=True), tf(Image.fromarray(bgr[:, ::-1, ::-1].copy())))
    # The model sees "texting, right hand" in the flipped frame: the driver uses the left hand.
    assert int(_scores(1)[HAND_SWAP].argmax()) == 3
    assert int(_scores(4)[HAND_SWAP].argmax()) == 2
    assert [int(_scores(k)[HAND_SWAP].argmax()) for k in (0, 5, 6, 7, 8, 9)] == [0, 5, 6, 7, 8, 9]
    # A second swap gives the first order back.
    assert [HAND_SWAP[i] for i in HAND_SWAP] == list(range(10))


def test_demo_run_gives_a_test_driver_the_final_model():
    split = {"folds": [{"val": ["p014"]}, {"val": ["p015"]}]}
    assert demo_run(split, "p015") == "runs/e3_deit3_small_noflip_f1"
    assert demo_run(split, "p002", "runs/final") == "runs/final"
