"""The pose intervention cannot invalidate S37 or change its measurement gate."""
from tools.run_scoped_presentation_attempt import research_plan
from tools.platform_release_snapshot import snapshot


def test_matched_releases_change_only_stage42_and_share_relational_gate():
    manifest = {"presentation_research": {"exact_input_identity": "same"}, "runtime": {"native_player_identity": "same"}}
    releases = [snapshot(research_plan(variant, True), manifest,
        name=variant, purpose="RESEARCH", created_by="test")["manifest"] for variant in ("v3", "v4")]
    before, after = releases
    changed = [x["stage_id"] for x, y in zip(before["stages"], after["stages"]) if x != y]
    assert changed == ["42_RUNTIME_PROJECTION_AND_CAA_BINDING"]
    a, b = before["dag"]["stages"], after["dag"]["stages"]
    assert a[0] == b[0]
    assert a[-1] == b[-1]
    assert "presentation_research_v4:prove_presentation_stage" in a[-1]["adapter"]
    assert all(s["policy"]["product_pass_authority"] is False for s in b)
