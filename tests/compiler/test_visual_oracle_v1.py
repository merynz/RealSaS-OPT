"""Independent, hand-authored fixture; no ArtistIntent data in CI."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_core.visual_oracle_reference_v1 import compose, produce
from compiler.realsas_compiler_core.visual_oracle_package_v1 import package_frame
from compiler.realsas_compiler_core.visual_oracle_io_v1 import sha
from compiler.realsas_compiler_core.visual_oracle_measure_v1 import cached_source, independent_reference, metrics, initially_occluded, canonical_visible
from tools.platform_visual_oracle_v1 import oracle_plan
from tools.platform_release_snapshot import snapshot

BUDGETS = {"minimum_alpha_iou": .999, "maximum_pm_mean": 1/255, "maximum_pm_p99": 2/255,
           "maximum_robust_owner_mismatch_pixels": 0}


def fixture(tmp_path):
    def png(colour):
        output = BytesIO(); Image.new("RGBA", (8, 8), colour).save(output, format="PNG"); return output.getvalue()
    scml = b'''<spriter_data><folder id="0"><file id="0" name="cape.png" width="8" height="8"/><file id="1" name="head.png" width="8" height="8"/></folder><entity id="0"><animation id="0" name="move" length="200">
      <mainline><key id="0"><object_ref id="0" timeline="0" key="0" z_index="0" folder="0" file="0" abs_x="0" abs_y="0" abs_angle="0" abs_scale_x="1" abs_scale_y="1" abs_a="1"/><object_ref id="1" timeline="1" key="0" z_index="1" folder="0" file="1" abs_x="0" abs_y="0" abs_angle="0" abs_scale_x="1" abs_scale_y="1" abs_a="1"/></key>
      <key id="1" time="100"><object_ref id="0" timeline="0" key="0" z_index="0" folder="0" file="0" abs_x="0" abs_y="0" abs_angle="0" abs_scale_x="1" abs_scale_y="1" abs_a="1"/><object_ref id="1" timeline="1" key="1" z_index="1" folder="0" file="1" abs_x="4" abs_y="0" abs_angle="0" abs_scale_x="1" abs_scale_y="1" abs_a="1"/></key></mainline>
      <timeline id="0" obj="0" name="cape"><key id="0"><object folder="0" file="0"/></key></timeline>
      <timeline id="1" obj="1" name="head"><key id="0"><object folder="0" file="1"/></key><key id="1" time="100"><object folder="0" file="1" x="4"/></key></timeline>
    </animation></entity></spriter_data>'''
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("fixture/reference.scml", scml)
        z.writestr("fixture/cape.png", png((180, 30, 40, 255)))
        z.writestr("fixture/head.png", png((20, 100, 220, 255)))
    return {"archive_path": str(archive), "archive_sha256": sha(archive.read_bytes()),
            "scml_member": "fixture/reference.scml", "scml_sha256": sha(scml),
            "entity_id": 0, "clips": ["move"], "resolution": 64, "margin_pixels": 2}


def test_reflection_aware_spriter_component_composition():
    actual = compose(np.array([2, 3, 40, 4, 5, .8]), np.array([10, 20, 90, -2, 3, .5]))
    np.testing.assert_allclose(actual, [1, 16, 50, -8, 15, .4], atol=1e-12)


def test_producer_does_not_copy_independent_cached_pose(tmp_path):
    section = fixture(tmp_path)
    with zipfile.ZipFile(section["archive_path"]) as z:
        data = {n: z.read(n) for n in z.namelist()}
    data[section["scml_member"]] = data[section["scml_member"]].replace(b'abs_x="4"', b'abs_x="9"')
    with zipfile.ZipFile(section["archive_path"], "w") as z:
        for name, raw in data.items(): z.writestr(name, raw)
    section["archive_sha256"] = sha(Path(section["archive_path"]).read_bytes())
    section["scml_sha256"] = sha(data[section["scml_member"]])
    produced = produce(section, tmp_path / "evidence")
    _, frames, _ = cached_source(section)
    assert produced["frames"][1]["objects"][1]["pose"][0] == 4
    assert frames[1]["objects"][1]["pose"][0] == 9


def test_hash_drift_and_viewport_crop_fail_closed(tmp_path):
    section = fixture(tmp_path)
    with pytest.raises(ValueError, match="ARCHIVE_DRIFT"):
        produce({**section, "archive_sha256": "0"*64}, tmp_path / "bad")
    with pytest.raises(ValueError, match="CROPS_SOURCE"):
        produce({**section, "resolution": 8, "margin_pixels": 2}, tmp_path / "crop")


def test_research_graph_is_reconstructable_and_never_product():
    plan = oracle_plan()
    release = snapshot(plan, {"oracle_source": {}, "oracle_render": {}, "oracle_measure": {}},
        name="fixture-oracle", purpose="RESEARCH", created_by="pytest")
    assert release["manifest"]["dag"]["stage_count"] == 3
    assert all(not s["policy"]["product_pass_authority"] for s in plan["stages"])
    assert plan["stages"][-1]["manifest_keys"] == ["oracle_source", "oracle_measure"]


def test_actual_native_consumer_reference_mutations_and_reveal(tmp_path):
    player = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not player:
        pytest.skip("native player required by mainline CI")
    section = fixture(tmp_path)
    evidence = produce(section, tmp_path / "evidence")
    images, frames, owners = cached_source(section)
    refs = [independent_reference(f, images, owners, 64, evidence["offset"]) for f in frames]
    hidden = initially_occluded(frames[0], images, owners, refs[0][1], refs[0][2], evidence["offset"])
    visible = canonical_visible(frames[1], owners, refs[1][1], refs[1][3], refs[1][2])
    assert len(hidden[("entity:0/object:0", "fixture/cape.png")] & visible[("entity:0/object:0", "fixture/cape.png")]) > 0
    for variant in ("baseline", "wrong_order", "missing_layer", "wrong_owner", "wrong_pose"):
        path = tmp_path / (variant+".rss")
        face_owners = package_frame(evidence, evidence["frames"][1], path, variant)
        rgba, owner = tmp_path / (variant+".rgba"), tmp_path / (variant+".owner")
        subprocess.run([player, str(path), "--clip", "oracle", "--view", "V0", "--frame", "0", "--out-rgba", str(rgba), "--out-owner", str(owner)], check=True, capture_output=True)
        actual = np.fromfile(rgba, np.uint8).reshape(64, 64, 4)
        native_faces = np.fromfile(owner, "<i4").reshape(64, 64)
        actual_owner = np.full((64, 64), -1)
        lookup = np.array([owners.index(o) for o in face_owners])
        mask = native_faces >= 0; actual_owner[mask] = lookup[native_faces[mask]]
        expected, expected_owner, robust, _ = refs[1]
        measured = metrics(expected, actual, expected_owner, actual_owner, robust, BUDGETS)
        assert measured["passed"] == (variant == "baseline"), (variant, measured)
        if variant == "wrong_owner":
            assert measured["checks"]["alpha_iou"] and measured["checks"]["pm_mean"]
            assert not measured["checks"]["ownership"]


def test_native_rotated_mirrored_translucent_layer_matches_independent_sampler(tmp_path):
    player = os.environ.get("REALSAS_RUNTIME_V2_PLAYER")
    if not player:
        pytest.skip("native player required by mainline CI")
    section = fixture(tmp_path)
    with zipfile.ZipFile(section["archive_path"]) as z:
        data = {n: z.read(n) for n in z.namelist()}
    data[section["scml_member"]] = data[section["scml_member"]].replace(
        b'abs_x="4" abs_y="0" abs_angle="0" abs_scale_x="1"',
        b'abs_x="4" abs_y="3" abs_angle="33" abs_scale_x="-1"').replace(
        b'file="1" x="4"', b'file="1" x="4" y="3" angle="33" scale_x="-1"')
    alpha = np.arange(64, dtype=np.uint8).reshape(8, 8)*4
    pixels = np.zeros((8, 8, 4), np.uint8); pixels[:] = [20, 100, 220, 255]; pixels[..., 3] = alpha
    buffer = BytesIO(); Image.fromarray(pixels).save(buffer, format="PNG"); data["fixture/head.png"] = buffer.getvalue()
    with zipfile.ZipFile(section["archive_path"], "w") as z:
        for name, raw in data.items(): z.writestr(name, raw)
    section["archive_sha256"] = sha(Path(section["archive_path"]).read_bytes()); section["scml_sha256"] = sha(data[section["scml_member"]])
    evidence = produce(section, tmp_path / "evidence")
    images, frames, owners = cached_source(section)
    expected, _, _, _ = independent_reference(frames[1], images, owners, 64, evidence["offset"])
    package = tmp_path / "sample.rss"; package_frame(evidence, evidence["frames"][1], package)
    rgba = tmp_path / "sample.rgba"
    subprocess.run([player, str(package), "--clip", "oracle", "--view", "V0", "--frame", "0", "--out-rgba", str(rgba)], check=True, capture_output=True)
    actual = np.fromfile(rgba, np.uint8).reshape(64, 64, 4)
    assert np.max(np.abs(expected.astype(int)-actual.astype(int))) <= 1
