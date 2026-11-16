"""Write the scores of the development folds of one config to results/development/."""
import argparse
import json

from ddd.development import OUT_DIR, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", help="parent folder of the run folders, for example the Drive folder runs")
    p.add_argument("name", help="config name: the runs are <name>_f0, <name>_f1, and <name>_f2")
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    print(json.dumps(summarize(a.runs, a.name, out_dir=a.out), indent=2))


if __name__ == "__main__":
    main()
