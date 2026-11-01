import importlib.util
from pathlib import Path

MODULE = Path(__file__).parents[1] / "tools" / "corpus" / "build_corpus370_subject_family_authority_matrix_v1.py"
spec = importlib.util.spec_from_file_location("matrix_builder", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def subject(subject_id, **statuses):
    return {
        "subject_id": subject_id,
        "source_sha256": "a" * 64,
        "families": {name: {"status": status, "evidence": []} for name, status in statuses.items()},
    }


def test_missing_evidence_is_unknown_not_pass():
    out = mod.build({"subjects": [subject("s0")]})
    row = out["rows"][0]
    assert row["families"]["GLOBAL_SOLID_SDF"]["status"] == "UNKNOWN"
    assert row["models"]["IRIS"]["disposition"] == "PENDING"
    assert out["training_authorized"] is False


def test_iris_allows_explicit_sdf_mask_only_after_resolution():
    row = subject(
        "s0",
        SOURCE_IMPORT="PASS",
        CAMERA_OBSERVATION="PASS",
        SURFACE_GEOMETRY_UDF="PASS",
        GLOBAL_SOLID_SDF="MASKED",
    )
    out = mod.build({"subjects": [row]})
    assert out["rows"][0]["models"]["IRIS"]["disposition"] == "ELIGIBLE"


def test_tessa_source_missing_is_not_silently_eligible():
    row = subject(
        "s0",
        SOURCE_IMPORT="PASS",
        MECHANICAL_SURFACE_TOPOLOGY="PASS",
    )
    out = mod.build({"subjects": [row]})
    assert out["rows"][0]["models"]["TESSA"]["disposition"] == "PENDING"


def test_fail_blocks_model():
    row = subject(
        "s0",
        SOURCE_IMPORT="PASS",
        RIG_HIERARCHY="PASS",
        SKIN_WEIGHTS="FAIL",
    )
    out = mod.build({"subjects": [row]})
    assert out["rows"][0]["models"]["MIRA"]["disposition"] == "INELIGIBLE"
