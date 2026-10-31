from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from tools.corpus.build_presentation_family_from_stage48_witness_v2 import build
from tools.corpus.validate_presentation_family_from_stage48_witness_v2 import validate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def fixture(root: Path) -> Path:
    arrays = root / "STAGE48_RENDERER_WITNESS_ARRAYS.npz"
    np.savez_compressed(arrays, dummy=np.asarray([1], np.int32))
    hit = {"face_index": 7, "depth": 1.0, "barycentric": [0.2, 0.3, 0.5]}
    packed = {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "subject_id": "s",
        "teacher_scene_sha256": "a" * 64,
        "geometry_npz_sha256": "b" * 64,
        "candidate_sha256": "c" * 64,
        "camera_bundle_sha256": "d" * 64,
        "packed_source_scene_blend_sha256": "e" * 64,
        "sampling_contract": {"x": 1},
        "rgb_carrier_calibration": {"x": 1},
        "material_policy_replay": {"x": 1},
        "selected_ray_count": 1,
        "pass_ray_count": 1,
        "fail_ray_count": 0,
        "abstain_ray_count": 0,
        "rays": [{
            "ray_index": 0, "view_index": 0, "pixel_x": 10, "pixel_y": 20,
            "canonical_hits": [hit],
            "measurement": {
                "status": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
                "valid": True,
                "effective_alpha": [0.8],
                "straight_linear_rgb": [[0.25, 0.5, 0.75]],
                "incoming_transmittance": [1.0],
                "contribution": [0.8],
                "predicted_full_alpha": 0.8,
                "predicted_full_premultiplied_linear_rgb": [0.2, 0.4, 0.6],
            },
        }],
        "authority": "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS",
        "court_status": "CANDIDATE",
    }
    pa = root / "PACKED_A.json"; pb = root / "PACKED_B.json"; write(pa, packed); write(pb, packed)
    source = {
        "schema": "RealSaS.SourceRayAppearanceReference.v1",
        "subject_id": "s",
        "source_sha256": "f" * 64,
        "candidate_sha256": "c" * 64,
        "camera_bundle_sha256": "d" * 64,
        "appearance_sha256": "1" * 64,
        "geometry_npz_sha256": "b" * 64,
        "source_material_transform": {"x": 1},
        "rgb_carrier_calibration": {"x": 1},
        "equivalence_scope": "TEST",
        "hit_count": 1,
        "hits": [{
            "ray_index": 0, "view_index": 0, "pixel_x": 10, "pixel_y": 20,
            "hit_index": 0, "face_index": 7, "effective_alpha": 0.8,
            "premultiplied_linear_rgb": [0.2, 0.4, 0.6],
            "footprint_convergence_delta": 0.001,
        }],
        "authority": "SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY",
    }
    sa = root / "SOURCE_REFERENCE_A.json"; sb = root / "SOURCE_REFERENCE_B.json"; write(sa, source); write(sb, source)
    qualification = {
        "schema": "RealSaS.PresentationRaySubjectQualificationCandidate.v1",
        "status": "PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE",
        "subject_id": "s",
        "source_equivalent": True,
        "packed_replay_exact": True,
        "source_reference_replay_exact": True,
        "student_supervision_authorized": False,
        "local_qualification_decision": {
            "oracle_eligible": True,
            "deterministic_student_supervision_eligible": False,
            "conditional_student_supervision_eligible": False,
            "reasons": ["STUDENT_INPUT_LEAKAGE_NOT_CLEARED"],
        },
        "family_qualification_input": {
            "family": "RAY_COMPOSITING", "measurement_status": "MEASURED",
            "replay_deterministic": True, "source_equivalent": True,
        },
    }
    witness = root / "STAGE48_RENDERER_WITNESS_MANIFEST.json"
    write(witness, {
        "schema": "RealSaS.Truth139Stage48RendererWitness.v1",
        "status": "PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN",
        "subject_id": "s",
        "arrays": {"path": str(arrays), "sha256": sha(arrays)},
        "replays": {
            "packed_a": {"path": str(pa), "sha256": sha(pa)},
            "packed_b": {"path": str(pb), "sha256": sha(pb)},
            "source_reference_a": {"path": str(sa), "sha256": sha(sa)},
            "source_reference_b": {"path": str(sb), "sha256": sha(sb)},
        },
        "policy": {"student_observable_contribution_floor": 1e-6},
        "pass_domain": {"packed_a_pass_ray_count": 1, "packed_b_pass_ray_count": 1},
        "counts": {"measured_composite_hit_count": 1, "measured_composite_pair_count": 0},
        "renderer_subject_qualification": qualification,
        "stage48_identity_sha256": "9" * 64,
        "qualification_minted": False,
        "training_performed": False,
    })
    return witness


def test_stage48_witness_is_canonical_graphics_parent(tmp_path: Path):
    witness = fixture(tmp_path); out = tmp_path / "out"
    manifest = build(witness, out)
    court = validate(out / "PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V2.json")
    assert manifest["stage48"]["pass_ray_count"] == 1
    assert manifest["policy"]["path_discovery"] == "ONLY_HASH_BOUND_STAGE48_WITNESS_REFS"
    assert court["status"] == "PASS_PRESENTATION_FAMILY_STAGE48_BOUND_INDEPENDENT_COURT__STUDENT_ADMISSION_SEPARATE"
    assert court["counts"]["appearance_valid_rows"] == 1
    assert court["authority"]["training_authorized"] is False


def test_stage48_witness_replay_hash_drift_fails_closed(tmp_path: Path):
    witness = fixture(tmp_path)
    value = json.loads(witness.read_text())
    value["replays"]["packed_a"]["sha256"] = "0" * 64
    write(witness, value)
    with pytest.raises(ValueError, match="PRESENTATION_V2_PACKED_A_DRIFT"):
        build(witness, tmp_path / "out")


def test_stage48_bound_court_rejects_student_authority_escalation(tmp_path: Path):
    witness = fixture(tmp_path); out = tmp_path / "out"; build(witness, out)
    manifest_path = out / "PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V2.json"
    value = json.loads(manifest_path.read_text()); value["authority"]["training_authorized"] = True; write(manifest_path, value)
    with pytest.raises(ValueError, match="PRESENTATION_V2_COURT_PREMATURE_TRAINING_AUTHORITY"):
        validate(manifest_path)
