import numpy as np

from models.iris.semantic_part_naming_v1 import classify_part_region_v1


def _fixture():
    h = w = 24
    fg = np.zeros((h, w), dtype=bool)
    fg[2:22, 2:22] = True
    mask = np.zeros((h, w), dtype=bool)
    mask[8:16, 9:15] = True
    return h, w, fg, mask


def test_part_naming_accepts_localized_dominant_part():
    h, w, fg, mask = _fixture()
    fields = np.full((3, h, w), 0.12, dtype=np.float32)
    fields[0, fg] = 0.20
    fields[1, fg] = 0.18
    fields[2, fg] = 0.15
    fields[1, mask] = 0.88
    background = np.full((h, w), 0.08, dtype=np.float32)

    out = classify_part_region_v1(
        part_probabilities=fields,
        background_probability=background,
        mask=mask,
        foreground=fg,
        minimum_top_probability=0.34,
        minimum_class_margin=0.07,
        minimum_localization_margin=0.10,
        minimum_part_vs_background_margin=0.02,
    )
    assert out.accepted
    assert out.label_index == 1
    assert out.failure_code is None
    assert out.margin > 0.07


def test_part_naming_abstains_when_two_parts_are_ambiguous():
    h, w, fg, mask = _fixture()
    fields = np.full((3, h, w), 0.20, dtype=np.float32)
    fields[0, mask] = 0.780
    fields[1, mask] = 0.779
    background = np.full((h, w), 0.05, dtype=np.float32)

    out = classify_part_region_v1(
        part_probabilities=fields,
        background_probability=background,
        mask=mask,
        foreground=fg,
        minimum_class_margin=0.07,
    )
    assert not out.accepted
    assert out.label_index == -1
    assert out.failure_code == "PART_NAMING_AMBIGUOUS"


def test_part_naming_abstains_when_background_dominates():
    h, w, fg, mask = _fixture()
    fields = np.full((3, h, w), 0.10, dtype=np.float32)
    fields[2, mask] = 0.62
    background = np.full((h, w), 0.10, dtype=np.float32)
    background[mask] = 0.72

    out = classify_part_region_v1(
        part_probabilities=fields,
        background_probability=background,
        mask=mask,
        foreground=fg,
        minimum_top_probability=0.20,
        minimum_class_margin=0.02,
        minimum_localization_margin=0.05,
        minimum_part_vs_background_margin=0.02,
    )
    assert not out.accepted
    assert out.label_index == -1
    assert out.failure_code == "PART_NAMING_BACKGROUND_DOMINANT"
