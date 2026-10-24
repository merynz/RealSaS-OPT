from tools.platform_deployment_smoke import smoke_plans
from tools.platform_release_snapshot import snapshot


def _assert_smoke_release_projection(plan, name):
    request = snapshot(plan, name=name, purpose="RESEARCH", created_by="ci")
    assert set(request["capability_versions"]) == {"product.input", "product.shape"}
    requirements = request["manifest"]["model_requirements"]
    assert [row["family"] for row in requirements] == ["IRIS", "TESSA"]
    assert all(row["channel"] == "RESEARCH" for row in requirements)
    assert "AXIS" not in {row["family"] for row in requirements}


def test_deployment_smoke_partial_releases_do_not_advertise_mesh_or_rig():
    baseline, candidate = smoke_plans()
    _assert_smoke_release_projection(baseline, "partial-smoke-baseline")
    _assert_smoke_release_projection(candidate, "partial-smoke-candidate")
