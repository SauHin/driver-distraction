from ddd.audit import audit_statefarm
from ddd.manifest import build_manifest, compare, read_manifest, write_manifest


def test_audit_finds_fixture_defects(statefarm_root):
    a = audit_statefarm(statefarm_root / "driver_imgs_list.csv", statefarm_root / "imgs" / "train")
    assert a["rows"] == 5
    assert a["drivers"] == 2
    assert a["images_per_driver_and_class"] == {"p001": {"c0": 1, "c1": 1},
                                                "p002": {"c0": 1, "c1": 1, "c2": 1}}
    assert a["missing_files"] == ["c1/img_3.jpg"]
    assert a["files_not_in_label_file"] == ["c2/extra.jpg"]
    assert a["image_sizes"] == {"8x6": 3, "6x8": 1}
    copies = a["exact_copies"]
    assert copies["groups"] == 1
    assert copies["groups_across_drivers"] == 1
    assert copies["list"][0]["drivers"] == ["p001", "p002"]


def test_manifest_round_trip_and_compare(statefarm_root, tmp_path):
    rows = build_manifest(statefarm_root)
    assert len(rows) == 6  # 5 images on disk and the label file
    out = tmp_path / "m.csv"
    write_manifest(rows, out)
    assert read_manifest(out) == rows
    assert compare(rows, build_manifest(statefarm_root)) == {"missing": [], "extra": [],
                                                             "changed": []}

    (statefarm_root / "imgs" / "train" / "c2" / "extra.jpg").unlink()
    (statefarm_root / "imgs" / "train" / "c0" / "new.jpg").write_bytes(b"x")
    with open(statefarm_root / "driver_imgs_list.csv", "a", encoding="utf-8") as f:
        f.write("p003,c0,new.jpg\n")
    diff = compare(rows, build_manifest(statefarm_root))
    assert diff == {"missing": ["imgs/train/c2/extra.jpg"], "extra": ["imgs/train/c0/new.jpg"],
                    "changed": ["driver_imgs_list.csv"]}
