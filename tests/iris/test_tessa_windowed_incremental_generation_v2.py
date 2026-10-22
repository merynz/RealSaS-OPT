from models.tessa.v2.windowed_incremental_generation_v2 import (
    assert_windowed_incremental_reference_parity_v2,
)


def test_windowed_incremental_generation_matches_reference_across_boundaries():
    receipt = assert_windowed_incremental_reference_parity_v2(device="cpu")
    assert receipt["status"] == "PASS__WINDOWED_INCREMENTAL_REFERENCE_PARITY"
    assert receipt["prefix_count"] == 17
    assert receipt["post_window_prefixes"] > 0
