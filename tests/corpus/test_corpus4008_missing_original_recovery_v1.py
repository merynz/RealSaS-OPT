"""Public-safe recovery routing tests; no original downloads or training."""
from tools.corpus.audit_corpus4008_missing_original_recovery_v1 import (
    retrieval_route, SCHEMA,
)


def test_public_recovery_route_does_not_mint_access_or_data_rights():
    assert retrieval_route(
        "https://github.com/owner/repo/blob/deadbeef/a.fbx", "github"
    ) == "PINNED_GITHUB_BLOB_OR_RAW__EXACT_BYTE_HASH_REQUIRED"
    assert retrieval_route(
        "https://sketchfab.com/3d-models/a", "sketchfab"
    ) == "SKETCHFAB_OBJAVERSE_OFFICIAL_FETCH__EXACT_BYTE_HASH_REQUIRED"
    assert retrieval_route("", "sketchfab") == "PROVIDER_IDENTIFIER_ONLY__RECONSTRUCT_OFFICIAL_URL"
    assert retrieval_route("", "") == "NO_RETRIEVAL_ROUTE_RESOLVED"
    assert SCHEMA.endswith("RecoveryPlan.v1")
