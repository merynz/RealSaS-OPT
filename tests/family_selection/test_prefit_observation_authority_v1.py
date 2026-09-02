from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from experiments.family_selection_v1.prefit_observation_authority_v1 import (
    ObservationViewEvidenceV1,
    verify_prefit_observation_authority_v1,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, forbidden_name: bool = False):
    rows = []
    for i, yaw in enumerate(range(0, 360, 45)):
        stem = f"V{i}"
        rgba_name = "cel_clean.png" if forbidden_name and i == 0 else "RGBA.png"
        d = tmp_path / stem
        d.mkdir()
        rgba = d / rgba_name
        Image.new("RGBA", (1024, 1024), (i, 10, 20, 255)).save(rgba)
        camera = d / "camera.json"
        camera.write_text(json.dumps({"yaw_deg": yaw}), encoding="utf-8")
        rows.append(ObservationViewEvidenceV1(i, yaw, str(rgba), _sha(rgba), str(camera), _sha(camera)))
    return rows


def test_exact_textured_eight_view_authority_passes(tmp_path: Path):
    rows = _fixture(tmp_path)
    out = verify_prefit_observation_authority_v1(
        asset_id="asset_fixture",
        raster_authority="TEXTURED_RGBA_WITH_SOURCE_APPEARANCE",
        views=rows,
    )
    assert out.explicit_source_textured_rgba is True
    assert out.exact_camera_raster_authority is True
    assert out.native_resolution == 1024
    assert len(out.views) == 8
    assert len(out.observation_authority_sha256) == 64


def test_cel_clean_is_hard_rejected_even_if_mislabeled_textured(tmp_path: Path):
    rows = _fixture(tmp_path, forbidden_name=True)
    with pytest.raises(ValueError, match="NON_PRODUCT_RASTER_FORBIDDEN"):
        verify_prefit_observation_authority_v1(
            asset_id="asset_fixture",
            raster_authority="TEXTURED_RGBA_WITH_SOURCE_APPEARANCE",
            views=rows,
        )


def test_non_textured_authority_is_rejected(tmp_path: Path):
    rows = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SOURCE_TEXTURED_RGBA_AUTHORITY_REQUIRED"):
        verify_prefit_observation_authority_v1(
            asset_id="asset_fixture",
            raster_authority="GEOMETRY_ISOLATION_CONTROL",
            views=rows,
        )
