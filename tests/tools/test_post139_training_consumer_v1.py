from __future__ import annotations

import pytest

from tools.corpus.post139_training_consumer_v1 import build_training_manifest, validate_split_plan


def authority_fixture():
    root = {"authority_digest": "authority"}
    identities = [
        {
            "subject_id": "base-a", "source_id": "base", "source_family": "q", "pack": "base",
            "source_sha256": "base-sha", "relative_source_path": "a.gltf",
            "fingerprints": {
                "structural_subject": "s1", "geometry_topology": "g1", "rig_rest": "r1",
                "rig_hierarchy": "h1", "skin_weights": "w1", "appearance": "a1",
                "animation_inventory": "m1",
            },
        },
        {
            "subject_id": "post-a", "source_id": "post", "source_family": "q", "pack": "post",
            "source_sha256": "post-sha", "relative_source_path": "a.gltf",
            "fingerprints": {
                "structural_subject": "s2", "geometry_topology": "g2", "rig_rest": "r2",
                "rig_hierarchy": "h1", "skin_weights": "w2", "appearance": "a2",
                "animation_inventory": "m2",
            },
        },
        {
            "subject_id": "excluded", "source_id": "post", "source_family": "q", "pack": "post",
            "source_sha256": "excluded-sha", "relative_source_path": "x.gltf",
            "fingerprints": {"structural_subject": "sx"},
        },
    ]
    ledgers = {
        "identity": {"subjects": identities},
        "admission": {"subjects": [
            {"subject_id": "base-a", "status": "REFERENCE_BASE139_ROW"},
            {"subject_id": "post-a", "status": "ADMIT_PRIMARY"},
            {"subject_id": "excluded", "status": "EXCLUDED_BY_SEMANTIC_POLICY"},
        ]},
        "lineage": {"components": [
            {"component_id": "LC0001", "members": ["base-a", "post-a"]},
        ]},
        "materialization": {"primary_subjects": [
            {
                "subject_id": "post-a", "lineage_component_id": "LC0001",
                "model_sha256": "post-sha", "workspace_relative_model": "post/raw/a.gltf",
                "source_closure_ledger_sha256": "ledger", "publisher_tree_digest": "tree",
            },
        ]},
        "roster": {
            "base139_training_row_count": 1,
            "post139_primary_materialization_count": 1,
            "expanded_full_training_row_count": 2,
        },
    }
    return root, ledgers


def test_training_manifest_contains_only_base_and_primary_post_rows():
    root, ledgers = authority_fixture()
    manifest = build_training_manifest(root, ledgers)
    assert manifest["status"] == "READY"
    assert manifest["training_row_count"] == 2
    assert manifest["partition_counts"] == {"BASE139": 1, "POST139": 1}
    by_id = {row["subject_id"]: row for row in manifest["subjects"]}
    assert set(by_id) == {"base-a", "post-a"}
    assert by_id["base-a"]["materialization_authority"] == "EXISTING_IMMUTABLE_TRUTH139_AUTHORITY"
    assert by_id["post-a"]["materialization_authority"] == "POST139_MATERIALIZATION_LEDGER_V1"


def test_lineage_component_cannot_cross_fit_unseen_boundary():
    root, ledgers = authority_fixture()
    manifest = build_training_manifest(root, ledgers)
    plan = {
        "schema": "RealSaS.Post139.TrainingSplitPlan.v1",
        "assignments": [
            {"subject_id": "base-a", "split": "FIT"},
            {"subject_id": "post-a", "split": "UNSEEN"},
        ],
    }
    with pytest.raises(ValueError, match="POST139_TRAINING_SPLIT_LINEAGE_LEAKAGE"):
        validate_split_plan(manifest, plan)


def test_lineage_atomic_split_is_accepted_and_normalized():
    root, ledgers = authority_fixture()
    manifest = build_training_manifest(root, ledgers)
    plan = {
        "schema": "RealSaS.Post139.TrainingSplitPlan.v1",
        "assignments": [
            {"subject_id": "base-a", "split": "fit"},
            {"subject_id": "post-a", "split": "FIT"},
        ],
    }
    validated = validate_split_plan(manifest, plan)
    assert validated["status"] == "PASS_LINEAGE_ATOMIC_SPLIT"
    assert validated["split_counts"] == {"FIT": 2}
