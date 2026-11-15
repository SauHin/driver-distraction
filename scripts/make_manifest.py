"""Write a SHA-256 manifest of a data folder, or check a folder against one."""
import argparse
import sys

from ddd.manifest import build_manifest, compare, read_manifest, write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="data folder, for example data/raw/statefarm")
    p.add_argument("manifest", help="manifest CSV, for example data/manifests/statefarm.csv")
    p.add_argument("--check", action="store_true",
                   help="compare ROOT with an existing manifest instead of a write")
    a = p.parse_args()
    rows = build_manifest(a.root)
    if not a.check:
        write_manifest(rows, a.manifest)
        print(f"{a.manifest}: {len(rows)} files")
        return
    diff = compare(read_manifest(a.manifest), rows)
    for kind, paths in diff.items():
        for path in paths:
            print(f"{kind}: {path}")
    if any(diff.values()):
        sys.exit(1)
    print(f"{a.root}: {len(rows)} files match {a.manifest}")


if __name__ == "__main__":
    main()
