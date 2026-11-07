from __future__ import annotations
import json

import pytest
from PIL import Image, ImageDraw

from tools.corpus.audit_corpus4008_lookdev_pixels_v1 import audit


MODES = ("SOURCE_MATERIAL_DIAGNOSTIC",
         "STYLIZED_FLAT_DIAGNOSTIC",
         "STYLIZED_CEL_3BAND_DIAGNOSTIC")


def scene(tmp_path, *, scale=0.55, cel_diff=True):
    names = []
    size = 128
    width = round(size * scale)
    lo = (size - width) // 2
    for i, name in enumerate(MODES):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        color = [(100, 160, 180, 255), (130, 155, 175, 255),
                 (35 if cel_diff else 130, 65 if cel_diff else 155,
                  80 if cel_diff else 175, 255)][i]
        ImageDraw.Draw(img).rectangle(
            (lo, lo, lo + width - 1, lo + width - 1), fill=color)
        filename = name.lower() + ".png"
        img.save(tmp_path / filename)
        names.append({"mode": name, "path_private": filename})
    (tmp_path / "LOOKDEV_STUDY_V1.json").write_text(json.dumps({
        "asset_id": "x", "source_sha256": "a"*64,
        "training_authorized": False,
        "source_rgba_truth_authorized": False,
        "outputs": names,
    }))


def test_visual_pilot_requires_adequate_composition_and_distinct_style(tmp_path):
    scene(tmp_path)
    result = audit(tmp_path)
    assert result["numeric_pilot_gate_pass"] is True
    assert result["training_authorized"] is False
    assert result["human_visual_review_courted"] is False


def test_tiny_source_render_must_fail_without_guessing_quality(tmp_path):
    scene(tmp_path, scale=0.22)
    with pytest.raises(ValueError, match="COMPOSITION_OR_STYLE_UNDER_THRESHOLD"):
        audit(tmp_path)


def test_identical_flat_and_cel_must_fail(tmp_path):
    scene(tmp_path, cel_diff=False)
    with pytest.raises(ValueError, match="COMPOSITION_OR_STYLE_UNDER_THRESHOLD"):
        audit(tmp_path)


def test_no_training_authority_even_when_pixel_gate_passes(tmp_path):
    scene(tmp_path)
    d = json.loads((tmp_path / "LOOKDEV_STUDY_V1.json").read_text())
    d["training_authorized"] = True
    (tmp_path / "LOOKDEV_STUDY_V1.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="RENDER_AUTHORITY"):
        audit(tmp_path)
