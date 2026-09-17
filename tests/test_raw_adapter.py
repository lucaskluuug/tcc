from logoloc.data.raw_adapter import _find_dir_case_insensitive


def test_find_dir_case_insensitive_matches_different_case(tmp_path):
    masks_dir = tmp_path / "masks"
    (masks_dir / "hp").mkdir(parents=True)

    result = _find_dir_case_insensitive(masks_dir, "HP")

    assert result == masks_dir / "hp"


def test_find_dir_case_insensitive_exact_case_also_matches(tmp_path):
    masks_dir = tmp_path / "masks"
    (masks_dir / "adidas").mkdir(parents=True)

    result = _find_dir_case_insensitive(masks_dir, "adidas")

    assert result == masks_dir / "adidas"


def test_find_dir_case_insensitive_no_match_returns_none(tmp_path):
    masks_dir = tmp_path / "masks"
    (masks_dir / "adidas").mkdir(parents=True)

    result = _find_dir_case_insensitive(masks_dir, "nike")

    assert result is None


def test_find_dir_case_insensitive_parent_missing_returns_none(tmp_path):
    result = _find_dir_case_insensitive(tmp_path / "does_not_exist", "hp")
    assert result is None
