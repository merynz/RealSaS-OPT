from __future__ import annotations

"""User-edited M*/Rig/Skin -> real native 2D pixels, never source authority."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
)
from compiler.rss_vnext.workers.rss_vnext_canonical_authoring_revision_v1 import (
    compile_rss_vnext_canonical_user_edit_revision_v1,
)
from compiler.rss_vnext.workers.rss_vnext_edited_graphics_preview_v1 import (
    AUTHORITY,SCHEMA,compile_rss_vnext_edited_graphics_preview_v1,
    reopen_rss_vnext_edited_graphics_preview_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)
from tests.compiler.test_rss_vnext_canonical_authoring_revision_v1 import (
    _args,_multijoint_preview_fixture,
)
from tests.compiler.test_rss_vnext_native_preview_package_v1 import (
    _build_player,_compiled_preview_fixture,
)


def _files(tmp_path: Path):
    base,_,mesh,rig,skin,_ = _multijoint_preview_fixture(tmp_path)
    unchanged = base.read_bytes()
    revision,edits = compile_rss_vnext_canonical_user_edit_revision_v1(
        source_rss_path=base,**_args(mesh,rig,skin),
    )
    edit_path = tmp_path/"user-mesh-rest-weights.rsedit"
    edit_path.write_bytes(revision)
    preview,receipt = compile_rss_vnext_edited_graphics_preview_v1(
        source_rss_path=base,revision_path=edit_path,
    )
    preview_path = tmp_path/"user-edited-native-preview.rss"
    preview_path.write_bytes(preview)
    return base,edit_path,preview_path,receipt,edits,unchanged


def test_edited_geometry_rig_skin_rebinds_face_uv_without_source_authority(
    tmp_path,
):
    base,edit_path,preview_path,receipt,edit_receipt,unchanged = _files(tmp_path)
    state = reopen_rss_vnext_edited_graphics_preview_v1(
        preview_path=preview_path,source_rss_path=base,revision_path=edit_path,
    )
    original = reopen_rss_vnext_native_preview_package_v1(base)
    edited = state["preview"]
    assert receipt["schema"] == SCHEMA
    assert receipt["authority"] == AUTHORITY
    assert receipt["authoring_revision_id"] == edit_receipt["authoring_revision_id"]
    assert receipt["source_color_or_visibility_qualified"] is False
    assert receipt["original_source_camera_registered"] is False
    assert receipt["product_release_admission"] is False
    assert edited["mesh"].mesh_lineage_hash != original["mesh"].mesh_lineage_hash
    assert edited["rig"].rig_lineage_hash != original["rig"].rig_lineage_hash
    assert edited["skin"].skin_lineage_hash != original["skin"].skin_lineage_hash
    assert edited["manifest"]["graphics_uv_binding_qualified"] == "0"
    assert edited["manifest"]["source_uv_requalification_pass"] == "0"
    assert edited["manifest"]["user_edited_source_color_layers_erased"] == "1"
    assert edited["manifest"]["product_revision_minted"] == "0"
    assert edited["pose_bytes"] != original["pose_bytes"]
    assert base.read_bytes() == unchanged

    original_entries = read_rss_v2(base)
    derived_entries = read_rss_v2(preview_path)
    assert original_entries["texture.bin"] != derived_entries["texture.bin"]
    assert original_entries["graphics.bin"] != derived_entries["graphics.bin"]
    assert derived_entries["texture.bin"][:8] == original_entries["texture.bin"][:8]
    # Explicitly remove observed RGB taps; keep predicted H(x) candidate
    # and the face-local UV coordinates, without counterfeiting source pixels.
    version,width,height,faces,channels = __import__("struct").unpack_from(
        "<IIIII",derived_entries["texture.bin"],8,
    )
    start = 232 + faces*24
    assert set(derived_entries["texture.bin"][start:start+10*width*height]) == {0}
    inferred_start = start+10*width*height
    assert (
        derived_entries["texture.bin"][inferred_start:inferred_start+4*width*height]
        == original_entries["texture.bin"][inferred_start:inferred_start+4*width*height]
    )
    assert version in (1,2) and channels == 3

    exe = _build_player(tmp_path)
    rgba = tmp_path/"edited.rgba"
    owner = tmp_path/"edited.owner"
    authority = tmp_path/"edited.authority"
    unknown = tmp_path/"edited.unknown"
    result = subprocess.run([
        str(exe),"--rss",str(preview_path),"1","0","-1",
        str(rgba),str(owner),str(authority),
        "--unknown-mask="+str(unknown),
    ],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" in result.stdout
    assert "source_layer=-1" in result.stdout
    assert len(rgba.read_bytes()) == 96*96*4
    owners = np.frombuffer(owner.read_bytes(),dtype="<i4")
    kinds = np.frombuffer(authority.read_bytes(),dtype=np.uint8)
    assert np.all(owners >= -1)
    assert np.any(owners >= 0)
    assert set(np.unique(kinds)).issubset({0,2})
    assert unknown.stat().st_size == 96*96

    # Even an explicit source-view hint cannot recover observed authority=1.
    source_hint = subprocess.run([
        str(exe),"--rss",str(preview_path),"1","0","0",
        str(tmp_path/"hint.rgba"),str(tmp_path/"hint.owner"),
        str(tmp_path/"hint.authority"),
    ],capture_output=True,text=True)
    assert source_hint.returncode == 0,source_hint.stderr
    assert 1 not in np.frombuffer(
        (tmp_path/"hint.authority").read_bytes(),dtype=np.uint8,
    )


def test_eight_camera_user_mesh_edit_reselects_only_virtual_cameras(tmp_path):
    base,_,mesh,rig,skin,_ = _compiled_preview_fixture(
        tmp_path,eight_way=True,
    )
    vertex = mesh.vertices[0]
    blob,_ = compile_rss_vnext_canonical_user_edit_revision_v1(
        source_rss_path=base,
        mesh_vertex_position_edits={
            vertex.canonical_mesh_vertex_id:(
                vertex.P[0],vertex.P[1],vertex.P[2]+0.02,
            ),
        },
    )
    edit_path = tmp_path/"eight-view-mesh.rsedit"
    edit_path.write_bytes(blob)
    rebound,receipt = compile_rss_vnext_edited_graphics_preview_v1(
        source_rss_path=base,revision_path=edit_path,
    )
    path = tmp_path/"eight-view-user-edited-preview.rss"
    path.write_bytes(rebound)
    state = reopen_rss_vnext_edited_graphics_preview_v1(
        preview_path=path,source_rss_path=base,revision_path=edit_path,
    )["preview"]
    assert state["manifest"]["native_view_count"] == "8"
    assert len(state["virtual_camera"]) == 8
    assert state["manifest"]["original_source_camera_observed"] == "0"
    assert receipt["product_release_admission"] is False
    exe = _build_player(tmp_path)
    result = subprocess.run([
        str(exe),"--rss",str(path),"1","2","-1",str(tmp_path/"eight.rgba"),
    ],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" in result.stdout


def test_mutated_preview_even_rehashed_entries_cannot_replay(tmp_path):
    base,edit_path,preview_path,_,_,_ = _files(tmp_path)
    entries = read_rss_v2(preview_path)
    data = bytearray(entries["texture.bin"])
    data[-4] ^= 1
    entries["texture.bin"] = bytes(data)
    manifest = dict(
        line.split("=",1)
        for line in entries["manifest.txt"].decode().splitlines()
    )
    manifest["sha256:texture.bin"] = sha256(entries["texture.bin"]).hexdigest()
    entries["manifest.txt"] = (
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode()
    preview_path.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="REPLAY_DRIFT"):
        reopen_rss_vnext_edited_graphics_preview_v1(
            preview_path=preview_path,source_rss_path=base,
            revision_path=edit_path,
        )


def test_mutated_product_authority_flag_fails_even_with_rehashed_manifest(tmp_path):
    base,edit_path,preview_path,_,_,_ = _files(tmp_path)
    entries = read_rss_v2(preview_path)
    manifest = dict(
        line.split("=",1)
        for line in entries["manifest.txt"].decode().splitlines()
    )
    manifest["product_revision_minted"] = "1"
    manifest["source_rgb_surface_ownership_qualified"] = "1"
    entries["manifest.txt"] = (
        "\n".join(f"{k}={v}" for k,v in sorted(manifest.items()))+"\n"
    ).encode()
    preview_path.write_bytes(_entry_bytes(entries))
    with pytest.raises(RuntimeError,match="REPLAY_DRIFT"):
        reopen_rss_vnext_edited_graphics_preview_v1(
            preview_path=preview_path,source_rss_path=base,
            revision_path=edit_path,
        )


def test_original_source_edit_pair_is_immutable(tmp_path):
    base,edit_path,preview_path,_,_,_ = _files(tmp_path)
    another = tmp_path/"edited-original-archive.rss"
    modified = bytearray(base.read_bytes())
    modified[-12] ^= 1
    another.write_bytes(modified)
    with pytest.raises((RuntimeError,ValueError)):
        compile_rss_vnext_edited_graphics_preview_v1(
            source_rss_path=another,revision_path=edit_path,
        )
