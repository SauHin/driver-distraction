import json

import pytest

from ddd.development import summarize


def _fold_run(runs, k, epoch=1, lr=0.1):
    run = runs / f"b0_f{k}"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"fold": k, "epochs": 2, "lr": lr}), encoding="utf-8")
    (run / "environment.json").write_text("{}", encoding="utf-8")
    (run / "history.csv").write_text("epoch,epoch_seconds\n0,1.5\n1,2.5\n", encoding="utf-8")
    m = {"epoch": epoch, "val_loss": 1.0, "val_accuracy": 0.5 + k / 10,
         "val_class_mean_accuracy": 0.5, "val_macro_f1": 0.3 * k}
    (run / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


def test_summarize_reads_last_epoch_of_each_fold(tmp_path):
    for k in range(3):
        _fold_run(tmp_path / "runs", k)
    s = summarize(tmp_path / "runs", "b0", out_dir=tmp_path / "out")
    assert s["mean"]["val_macro_f1"] == pytest.approx(0.3)
    assert s["mean"]["val_accuracy"] == pytest.approx(0.6)
    assert [f["sum_epoch_seconds"] for f in s["folds"]] == [4.0] * 3
    assert (tmp_path / "out" / "b0_f2" / "history.csv").exists()
    assert json.loads((tmp_path / "out" / "b0_folds.json").read_text(encoding="utf-8")) == s


@pytest.mark.parametrize("bad", [{"epoch": 0}, {"lr": 0.5}])
def test_summarize_refuses_an_incomplete_or_other_fold(tmp_path, bad):
    _fold_run(tmp_path / "runs", 0)
    _fold_run(tmp_path / "runs", 1, **bad)
    _fold_run(tmp_path / "runs", 2)
    with pytest.raises(ValueError):
        summarize(tmp_path / "runs", "b0", out_dir=tmp_path / "out")
