from ddd.classes import NAME_TO_STATEFARM, NAMES, STATEFARM_TO_NAME, STATUS

CODES = sorted(f"c{i}" for i in range(10))


def test_codes_cover_ten_classes_once():
    assert len(set(NAMES)) == 10
    assert sorted(STATEFARM_TO_NAME) == CODES


def test_name_and_code_map_both_ways():
    for code, name in STATEFARM_TO_NAME.items():
        assert NAME_TO_STATEFARM[name] == code


def test_only_safe_driving_is_safe():
    assert [n for n, s in STATUS.items() if s == "safe"] == ["safe_driving"]
