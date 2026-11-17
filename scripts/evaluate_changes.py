"""Phase 3 step 5: score the last epoch of a run on changed test images (decisions D20 and P20).
Each call opens the test lock one time and adds a row to results/test_access_log.csv."""
import argparse

from ddd.robust import evaluate_changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--reason", required=True, help="the reason for the test log")
    p.add_argument("--out", default="results/final")
    a = p.parse_args()
    evaluate_changes(a.run_dir, a.reason, a.out)


if __name__ == "__main__":
    main()
