"""Synthetic-only source-scene contract regression tests (NOT 4008 PASS)."""
import hashlib
import json
import struct

import pytest

from tools.corpus.probe_corpus4008_source_scene_v1 import build, scene_from_glb


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glb(path, doc):
    data = json.dumps(doc, sort_keys=True).encode()
    data += b" " * ((-len(data)) % 4)
    size = 12 + 8 + len(data)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, size)
                     + struct.pack("<I4s",len(data),b"JSON") + data)


def test_one_rig_with_joint_parented_unskinned_prop_and_action(tmp_path):
    path = tmp_path / "source.glb"
    _glb(path,{
      "asset":{"version":"2.0"},
      "nodes":[
        {"name":"Rig","children":[1]},
        {"name":"hand","children":[2,3]},
        {"name":"Sword","mesh":0},
        {"name":"body","mesh":1,"skin":0},
      ],
      "meshes":[{"primitives":[{}]},{"primitives":[{}]}],
      "skins":[{"joints":[1]}],
      "animations":[{"name":"Wave","channels":[{"target":{"node":1,"path":"rotation"}}]}],
      "materials":[], "images":[]
    })
    sha = _sha(path)
    result = build(path, source_format="glb", expected_sha256=sha,
                   canonical_asset_id="asset_synthetic",
                   candidate_id="synthetic:"+sha)
    s = result["source_scene"]
    assert s["native_animation_count"] == 1
    assert s["source_skin_joint_counts"] == [1]
    assert s["bone_parented_unskinned_mesh_count"] == 1
    assert s["bone_parented_unskinned_meshes"][0]["source_node_name_teacher_only"] == "Sword"
    assert s["bone_parented_unskinned_meshes"][0]["motion_consequence_courted"] is False
    assert result["character_vs_prop"] == "UNKNOWN"
    assert result["source_anatomical_up"] == "UNKNOWN"
    assert result["eight_view_camera_courted"] is False
    assert result["semantic_training_authorized"] is False
    assert result["training_authorized"] is False
    assert result["source_names_student_features"] is False


def test_geometry_only_obj_does_not_mint_rig_or_closed_sdf(tmp_path):
    p = tmp_path / "source.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\no FlatCape\nf 1 2 3\n")
    sha = _sha(p)
    result = build(p, source_format="obj", expected_sha256=sha,
                   canonical_asset_id="asset_synthetic", candidate_id="synthetic:"+sha)
    s = result["source_scene"]
    assert s["source_triangle_count"] == 1
    assert s["source_domains"][0]["boundary_edge_count"] == 3
    assert s["source_domains"][0]["watertight_by_shared_source_index"] is False
    assert s["source_group_is_semantic_component"] is False
    assert s["rig_in_native_obj"] is False
    assert result["graphics_training_authorized"] is False


def test_hash_drift_and_malformed_graph_refused(tmp_path):
    p = tmp_path / "source.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError,match="SOURCE_SHA_MISMATCH"):
        build(p, source_format="obj", expected_sha256="f"*64,
              canonical_asset_id="a",candidate_id="q:"+"f"*64)
    with pytest.raises(ValueError,match="BAD_CHILD"):
        scene_from_glb({"nodes":[{"children":[3]}]})


def test_verified_fbx_cannot_claim_structure_without_pinned_blender(tmp_path):
    p=tmp_path/"fake.fbx";p.write_bytes(b"source-fbx-bytes")
    sha=_sha(p)
    out=build(p,source_format="fbx",expected_sha256=sha,
              canonical_asset_id="a",candidate_id="q:"+sha)
    assert out["source_scene"] is None
    assert out["source_scene_status"]=="PINNED_BLENDER_SOURCE_EVALUATOR_REQUIRED"
    assert out["training_authorized"] is False


def test_source_ancestor_cycles_refused():
    with pytest.raises(ValueError,match="NODE_CYCLE"):
        scene_from_glb({
          "nodes":[{"children":[1]}, {"children":[0],"mesh":0}],
          "meshes":[{"primitives":[]}], "skins":[],
        })
