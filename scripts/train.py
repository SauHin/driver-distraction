"""Start or continue one training run from a config file."""
import argparse
import json
from pathlib import Path

from ddd.train import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--runs", default="runs", help="parent folder of the run folders")
    p.add_argument("--name", help="run folder name; default: the name in the config")
    p.add_argument("--data-root", help="replaces data_root of the config, for example on Colab")
    p.add_argument("--fold", type=int, help="replaces fold of the config: development fold 0, 1, or 2")
    p.add_argument("--epochs", type=int, help="speed measurement only: fewer epochs")
    p.add_argument("--max-steps", type=int, help="speed measurement only: steps per epoch")
    p.add_argument("--batch-size", type=int, help="speed measurement only: another batch size")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    for key, value in (("data_root", a.data_root), ("fold", a.fold), ("epochs", a.epochs),
                       ("batch_size", a.batch_size)):
        if value is not None:
            cfg[key] = value
    run(cfg, Path(a.runs) / (a.name or cfg["name"]), max_steps=a.max_steps)


if __name__ == "__main__":
    main()
