"""SHA-256 manifest of a data folder, and a check of a folder against it."""
import csv
import hashlib
from pathlib import Path


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(root):
    """Sorted rows (relative path, size, sha256) for each file under `root`."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return [(p.relative_to(root).as_posix(), p.stat().st_size, sha256_file(p)) for p in files]


def write_manifest(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "sha256"])
        w.writerows(rows)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["path"], int(r["size"]), r["sha256"]) for r in csv.DictReader(f)]


def compare(expected, actual):
    """Paths that are missing, extra, or changed in `actual` against `expected`."""
    exp = {p: (s, h) for p, s, h in expected}
    act = {p: (s, h) for p, s, h in actual}
    return {
        "missing": sorted(set(exp) - set(act)),
        "extra": sorted(set(act) - set(exp)),
        "changed": sorted(p for p in set(exp) & set(act) if exp[p] != act[p]),
    }
