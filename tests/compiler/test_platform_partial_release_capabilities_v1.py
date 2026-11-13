"""Compiler-stage smoke is not equivalent to product release admission.

The source-owned product role catalog is a complete nine-role RSS.vNext graph;
a historical one/two-stage RESEARCH smoke may pin its implementations, but
does not execute or certify a puppet. Product authority remains closed.
"""
from tools.platform_deployment_smoke import smoke_plans
from tools.platform_release_snapshot import snapshot
from tools.platform_capability_versions import capability_specs


def _assert_smoke_release_projection(plan, name):
    request = snapshot(plan, name=name, purpose="RESEARCH", created_by="ci")
    roles = {
        "product.input", "product.shape", "product.mesh", "product.rig",
        "product.skin", "product.mechanics", "product.graphics",
        "product.motion", "product.proof",
    }
    assert set(request["capability_implementations"]) == roles
    assert all(
        set(row) == {"implementation_sha256"}
        for row in request["capability_implementations"].values()
    )
    requirements = request["manifest"]["model_requirements"]
    assert requirements == [{
        "family": "RSSVNEXT",
        "architecture_id": "RSS.vNext.UnifiedCharacterCompiler.v1",
        "channel": "RESEARCH",
    }]
    specs = capability_specs()
    # Proof is the only decision owner; ownership does not mean that the
    # current release is independently qualified to mint a ProductRevision.
    assert {role for role in roles
            if specs[role]["policy"]["product_pass_authority"]} == {"product.proof"}
    assert all(specs[role]["parameters"].get("can_mint_product_revision") is False
               for role in roles)
    assert specs["product.proof"]["parameters"]["product_revision_requires_independent_admission"] is True
    assert len(plan["stages"]) in (1, 2)
    assert request["manifest"]["contract_version"] == "RealSaS.EngineReleaseManifest.v2"


def test_deployment_smoke_research_does_not_claim_executable_product_proof():
    baseline, candidate = smoke_plans()
    _assert_smoke_release_projection(baseline, "partial-smoke-baseline")
    _assert_smoke_release_projection(candidate, "partial-smoke-candidate")
