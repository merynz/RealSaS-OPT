from __future__ import annotations

from tools.corpus.render_corpus370_authority_matrix_markdown_v1 import render


def test_report_surfaces_model_counts_sdf_and_row_masks():
    families = {
        name: {"status": "UNKNOWN", "claim_metadata": []}
        for name in (
            "SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF", "GLOBAL_SOLID_SDF",
            "SURFACE_NORMALS", "APPEARANCE_CAA", "VISIBILITY_OCCLUSION", "RAY_COMPOSITING",
            "MECHANICAL_SURFACE_TOPOLOGY", "CAUSAL_RESPONSE", "RIG_HIERARCHY", "SKIN_WEIGHTS",
            "ANIMATION_KINEMATICS", "TESSA_TEACHER",
        )
    }
    for name in ("SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF", "RIG_HIERARCHY", "SKIN_WEIGHTS"):
        families[name] = {"status": "PASS", "claim_metadata": []}
    families["SKIN_WEIGHTS"]["claim_metadata"] = [{
        "row_validity_mask_required": True,
        "valid_row_coverage_fraction": 0.75,
        "authority_scope": "PINNED_BLENDER_FBX_DECODED_SOURCE_SEMANTICS__NOT_RAW_FBX_PARSER_PROOF",
    }]
    families["GLOBAL_SOLID_SDF"] = {"status": "MASKED", "claim_metadata": []}
    row = {
        "subject_id": "subject_a",
        "source_format": ".fbx",
        "source_id": "pack/a",
        "families": families,
        "model_eligibility": {
            "IRIS": {"corpus_eligible": True, "required_pass": ["SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF"], "blocking_families": []},
            "TESSA": {"corpus_eligible": False, "required_pass": ["SOURCE_IMPORT", "MECHANICAL_SURFACE_TOPOLOGY", "TESSA_TEACHER"], "blocking_families": ["MECHANICAL_SURFACE_TOPOLOGY", "TESSA_TEACHER"]},
            "MIRA": {"corpus_eligible": True, "required_pass": ["SOURCE_IMPORT", "RIG_HIERARCHY", "SKIN_WEIGHTS"], "blocking_families": []},
            "AXIS": {"corpus_eligible": False, "required_pass": ["SOURCE_IMPORT", "RIG_HIERARCHY", "ANIMATION_KINEMATICS"], "blocking_families": ["ANIMATION_KINEMATICS"]},
        },
    }
    matrix = {
        "schema": "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1",
        "status": "OPEN",
        "training_authorized": False,
        "subject_count": 1,
        "subjects": [row],
        "rollup": {
            "unknown_cell_count": 8,
            "quarantined_cell_count": 0,
            "families": {name: {cell["status"]: 1} for name, cell in families.items()},
            "models": {
                "IRIS": {"eligible": 1},
                "TESSA": {"blocked": 1},
                "MIRA": {"eligible": 1},
                "AXIS": {"blocked": 1},
            },
        },
        "next_required_authority": ["seal_global_lineage_and_dedup"],
    }
    text = render(matrix)
    assert "IRIS eligible subjects (1)" in text
    assert "MIRA eligible subjects (1)" in text
    assert "SKIN_WEIGHTS=PASS (0.750)" in text
    assert "validity-mask 0.750" in text
    assert "Whole-family MASKED subjects: **1**" in text
    assert "Training remains closed" in text
