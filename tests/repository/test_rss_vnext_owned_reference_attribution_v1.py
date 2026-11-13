"""Ensure independent first-party code does not impersonate a published paper."""

from pathlib import Path

from models.rss_vnext.realsas_upstream_backbone_v1 import (
    CANONICAL, MULTIVIEW, RECONSTRUCTION, reference_ancestry_v1,
)

ROOT = Path(__file__).resolve().parents[2]


def test_first_party_neural_files_have_real_sas_ownership_not_paper_brand():
    root = ROOT / "models/rss_vnext"
    owned_files = sorted(root.glob("realsas_*.py"))
    assert len(owned_files) >= 15
    for file in owned_files:
        text = file.read_text(encoding="utf-8")
        assert "StdGEN" not in text, str(file)
        assert "RealSaS CVPR" not in text, str(file)
        assert "Published RealSaS" not in text, str(file)
        assert "source-released RealSaS" not in text, str(file)


def test_reconstruction_reference_has_verifiable_external_publication_identity():
    refs = [reference_ancestry_v1(x) for x in (CANONICAL, MULTIVIEW, RECONSTRUCTION)]
    assert refs[2].evidence == "https://arxiv.org/abs/2411.05738"
    assert refs[2].original_training_provenance == "CVPR2025_HE_ET_AL_SECTION_3_2_DIRECT"
    assert all(x.released_task_checkpoint_not_used for x in refs)
    assert all(x.released_era3d_weights_not_used for x in refs)


def test_uncalibrated_intake_does_not_claim_inherited_camera_authority():
    content = (ROOT / "compiler/realsas_compiler_core/rss_vnext_uncalibrated_upload_v1.py").read_text()
    assert "source_camera_calibrated" in content
    assert "source_cameras_known" in content
    assert "source_pose_canonicalized" in content
    assert "StdGEN" not in content
