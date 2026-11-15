"""Score the last epoch of a run on the test drivers. Each call opens the test lock and adds a
row to results/test_access_log.csv. Only the two points of docs/PLAN.md section 4 may call it."""
import argparse
import json

from ddd.evaluate import OUT_DIR, evaluate_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--reason", required=True, help="the reason for the test log")
    p.add_argument("--out", default=OUT_DIR, help="folder for the result files")
    a = p.parse_args()
    m = evaluate_run(a.run_dir, a.reason, a.out)
    print(json.dumps({k: v for k, v in m.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
