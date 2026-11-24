import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
ROOT = Path(__file__).resolve().parents[1]


def test_plot_figures_writes_each_figure(tmp_path):
    subprocess.run([sys.executable, "scripts/plot_figures.py", "--out", str(tmp_path)], cwd=ROOT, check=True)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "b0_curves.png", "confusion.png", "e1_tradeoff.png", "e5_drivers.png", "final_recipe_curves.png",
        "robustness.png"]
