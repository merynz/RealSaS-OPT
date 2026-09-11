from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import struct
import tempfile
import zlib
import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.surface import rigging_surface_from_d2_arrays
from compiler.realsas_compiler_core.types import SurfaceRelation, RiggingSurfaceIR, SkeletonProposalJoint, SkeletonProposalEdge, SkeletonProposalIR, SkinInfluenceProposal, SkinProposalIR
from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.v4 import build_mechanical_state, build_renderable_component, build_directional_renderable, build_directional_renderable_set, make_single_family_e2e_capability_contract, assemble_product_v3
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion
from compiler.realsas_compiler_core.proof_engine import evaluate_product_proof
from compiler.realsas_compiler_core.deformation import mesh_skin_dense_weights
from compiler.realsas_compiler_core.directional_binding import DirectionalBindingPolicyV1, qualify_directional_joint_view_binding
from compiler.realsas_compiler_services.proof.directional_motion_provider import make_qualified_directional_motion_provider
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
    materialize_current_v4_proof_bakes_runtime_v2,
)
from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1
from experiments.single_family_e2e_v1.export_bundle_v1 import export_product_bundle_v1
from runtime.reference_v4.consumer import consume_product_v4_reference


NATIVE_RESOLUTION = 1024


def _grid_to_pixel_center_xy(x, y):
    return (
        (float(x) + 1.0) * 0.5 * NATIVE_RESOLUTION - 0.5,
        (float(y) + 1.0) * 0.5 * NATIVE_RESOLUTION - 0.5,
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff)


def _solid_png_rgba(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    if width <= 0 or height <= 0 or len(rgba) != 4 or any(int(v) < 0 or int(v) > 255 for v in rgba):
        raise ValueError('invalid synthetic PNG request')
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', int(width), int(height), 8, 6, 0, 0, 0)
    scanline = b'\x00' + bytes(map(int, rgba)) * int(width)
    raw = scanline * int(height)
    return signature + _png_chunk(b'IHDR', ihdr) + _png_chunk(b'IDAT', zlib.compress(raw, level=9)) + _png_chunk(b'IEND', b'')


def _write_mock_textures(texture_root: Path):
    rows = []
    for view in range(8):
        rel = f'textures/view_{view}.png'
        path = texture_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _solid_png_rgba(NATIVE_RESOLUTION, NATIVE_RESOLUTION, ((37 * view) % 256, (97 + 23 * view) % 256, (181 + 11 * view) % 256, 255))
        path.write_bytes(payload)
        image_sha = sha256(payload).hexdigest()
        atlas_hash = content_sha256({'image_sha256': image_sha, 'image_relpath': rel})
        rows.append(RuntimeTexturePayloadV1(
            view,
            rel,
            image_sha,
            zlib.crc32(payload) & 0xffffffff,
            NATIVE_RESOLUTION,
            NATIVE_RESOLUTION,
            atlas_hash,
        ))
    return tuple(rows)


def _mock_surface():
    P = np.asarray([[-.4, -.4, 0.], [.4, -.4, 0.], [.4, .4, 0.], [-.4, .4, 0.]], np.float64)
    support = np.ones((4, 8), np.uint8)
    raster = np.zeros((4, 8, 2), np.float64)
    for i, (x, y, _z) in enumerate(P):
        for v in range(8):
            raster[i, v] = _grid_to_pixel_center_xy(x, y)
    base = rigging_surface_from_d2_arrays(P, support, raster, authority_label='MOCK_LEARNED_IRIS_DEPTH_V1')
    ids = [n.surface_id for n in base.surface_nodes]
    pairs = ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3))
    rel = tuple(SurfaceRelation(f'R:{i}', ids[a], ids[b], 'LOCAL_NEIGHBOR', 1.0, {'mock_learned_relation': True, 'crosses_unknown': False}) for i, (a, b) in enumerate(pairs))
    lineage = content_sha256({'base': base.geometry_lineage_hash, 'relations': [r.to_dict() for r in rel]})
    return RiggingSurfaceIR(
        base.surface_nodes,
        rel,
        lineage,
        base.builder_id,
        base.schema_version,
        {
            **base.metadata,
            'mock_learned_input': True,
            'raster_coordinate_system': 'PIXEL_CENTER_XY',
            'resolution': NATIVE_RESOLUTION,
        },
    )


def _mock_skeleton(surface):
    sids = tuple(n.surface_id for n in surface.surface_nodes)
    joints = (SkeletonProposalJoint('P:ROOT', (-.2, 0., 0.), 1.0, .99, sids), SkeletonProposalJoint('P:CHILD', (.2, 0., 0.), .05, .99, sids))
    edges = (SkeletonProposalEdge('E:ROOT_CHILD', 'P:ROOT', 'P:CHILD', 1.0, .99, False, False, 'mock learned directed relation'),)
    return SkeletonProposalIR(joints, edges, surface.geometry_lineage_hash, 'MOCK_GEPPETTO_V2', metadata={'teacher_truth_used': False})


def _mock_skin(surface, skeleton):
    jids = tuple(j.canonical_joint_id for j in skeleton.joints); influences = []
    for node in surface.surface_nodes:
        for jid in jids:
            influences.append(SkinInfluenceProposal(node.surface_id, jid, 1.0 / len(jids)))
    return SkinProposalIR(tuple(influences), surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, 'MOCK_ARACHNE_V2', metadata={'teacher_truth_used': False})


def _visuals(mechanical, texture_bindings):
    texture_by_view = {int(row.view_index): row for row in texture_bindings}
    dirs = []
    obs = {v: content_sha256({'mock_observation_view': v, 'image_sha256': texture_by_view[v].image_sha256}) for v in range(8)}
    for v in range(8):
        texture = texture_by_view[v]
        cam = content_sha256({'mock_exact_camera': v})
        cand = build_mwb2_candidate(mechanical.surface, view_index=v, camera_binding_hash=cam)
        mesh = qualify_mwb2_mesh(mechanical.surface, cand)
        mskin = bind_mwb2_mesh_skin(mechanical.surface, mechanical.skeleton, mechanical.skin, mesh)
        app = build_observed_appearance_binding(
            surface=mechanical.surface,
            mesh=mesh,
            target_view_index=v,
            camera_binding_hash=cam,
            observation_hash_by_view=obs,
            atlas_payload_hash=texture.atlas_payload_hash,
        )
        comp = build_renderable_component(component_id='BODY', view_index=v, mesh=mesh, mesh_skin=mskin, appearance=app, setup_order=0, coverage_classification='OBSERVED_SAFE_LOCAL_RELATION_COMPLEX', metadata={'mock_learned_e2e': True})
        dirs.append(build_directional_renderable(view_index=v, camera_binding_hash=cam, components=(comp,), metadata={'mock_exact_camera': True}))
    return build_directional_renderable_set(tuple(dirs), metadata={'mock_learned_e2e': True})


def _deformation_fixture(product):
    comp = product.directional_renderables.directions[0].components[0]
    p, w, jids = mesh_skin_dense_weights(comp.mesh, comp.mesh_skin, product.mechanical_state.skeleton)
    T = np.tile(np.eye(4, dtype=np.float64), (1, len(jids), 1, 1))
    if len(jids) > 1:
        T[0, 1, 0, 3] = 0.1
    expected = apply_verified_lbs_v1(p, w, T)
    return {'view_index': 0, 'component_index': 0, 'transforms': T, 'expected': expected, 'rms_threshold': 1e-7, 'p95_threshold': 1e-7}


def run_complete_e2e_v1(output_root=None):
    """Synthetic architecture gate through current proof-owned bake and native-v2 archive.

    The 4-point planar binding policy is explicitly synthetic-only. Production
    DirectionalBindingPolicyV1 keeps its 8-correspondence default; this gate proves
    typed wiring and rank-3 affine-hull semantics without weakening production policy.
    """
    root = Path(output_root) if output_root is not None else Path(tempfile.mkdtemp(prefix='realsas_e2e_'))
    root.mkdir(parents=True, exist_ok=True)
    texture_root = root / 'native_texture_source'
    texture_bindings = _write_mock_textures(texture_root)

    surface = _mock_surface()
    skeleton = qualify_skeleton_v2(surface, _mock_skeleton(surface))
    skin = qualify_skin(surface, skeleton, _mock_skin(surface, skeleton))
    mechanical = build_mechanical_state(surface, skeleton, skin)
    visuals = _visuals(mechanical, texture_bindings)
    motion = build_deterministic_preset_motion(mechanical)
    capability = make_single_family_e2e_capability_contract(
        base_lbs_hash=content_sha256({'impl': 'verified_lbs_v1'}),
        mesh_hash=content_sha256({'mesh': tuple(d.components[0].mesh.mesh_lineage_hash for d in visuals.directions)}),
        skinning_hash=skin.skin_lineage_hash,
        visual_hash=visuals.directional_visual_state_hash,
        preset_motion_hash=motion.motion_state_hash,
        runtime_hash=content_sha256({'runtime': 'native_v2_plus_reference_v4'}),
        policy_hash=content_sha256({'policy': 'single_family_e2e_v1'}),
    )
    product = assemble_product_v3(mechanical, visuals, capability, motion, editable_metadata={'synthetic_source_gate': True}, runtime_policy={'backend': 'REFERENCE_V4_AND_NATIVE_V2'})

    synthetic_binding_policy = DirectionalBindingPolicyV1(
        min_correspondences=4,
        required_affine_rank=3,
        max_p95_residual01=0.015,
        max_residual01=0.05,
        max_joint_affine_hull_residual01=0.02,
        min_raster_span=8.0,
    )
    directional_binding = qualify_directional_joint_view_binding(product, policy=synthetic_binding_policy)
    motion_provider = make_qualified_directional_motion_provider(directional_binding)
    proof_artifacts = {}
    proof = evaluate_product_proof(
        product,
        deformation_fixture=_deformation_fixture(product),
        motion_bake_provider=motion_provider,
        artifacts_out=proof_artifacts,
    )
    if proof.overall_status != 'PASS':
        raise RuntimeError(f'SYNTHETIC_E2E_PROOF_NOT_PASS:{proof.overall_status}')
    motion_bakes = tuple(proof_artifacts.get('motion_bakes', {}).values())
    if {bake.clip_id for bake in motion_bakes} != {motion.clips[0].clip_id}:
        raise RuntimeError('SYNTHETIC_E2E_MOTION_BAKE_NOT_CAPTURED')

    runtime, manifest = export_product_bundle_v1(
        root,
        product=product,
        proof_bundle=proof,
        directional_binding=directional_binding,
        motion_bakes=motion_bakes,
    )
    expected_bake_files = {f'proof/motion_bakes/{bake.clip_id}.json' for bake in motion_bakes}
    if set(manifest.get('motion_bake_files') or ()) != expected_bake_files:
        raise RuntimeError('SYNTHETIC_E2E_EXPORTED_MOTION_BAKE_SET_MISMATCH')
    if not all((root / rel).is_file() for rel in expected_bake_files):
        raise RuntimeError('SYNTHETIC_E2E_EXPORTED_MOTION_BAKE_MISSING')
    if manifest.get('directional_binding_set_hash') != directional_binding.binding_set_hash:
        raise RuntimeError('SYNTHETIC_E2E_EXPORTED_DIRECTIONAL_BINDING_MISMATCH')
    if not (root / str(manifest.get('directional_binding_file') or '')).is_file():
        raise RuntimeError('SYNTHETIC_E2E_EXPORTED_DIRECTIONAL_BINDING_MISSING')
    consume = consume_product_v4_reference(product, proof, runtime)

    native_archive = root / 'native' / 'realsas_runtime.rss'
    native_projection, native_result = materialize_current_v4_proof_bakes_runtime_v2(
        out_path=native_archive,
        texture_root=texture_root,
        product=product,
        proof_bundle=proof,
        motion_bakes=motion_bakes,
        texture_bindings=texture_bindings,
    )

    return {
        'status': 'PASS_COMPLETE_SYNTHETIC_E2E_V1',
        'product': product,
        'proof': proof,
        'runtime': runtime,
        'runtime_report': consume,
        'bundle_manifest': manifest,
        'output_root': str(root),
        'truth_paths_consumed': False,
        'generalization_claim': False,
        'scientific_fit_steps': 0,
        'directional_binding_hash': directional_binding.binding_set_hash,
        'qualified_motion_provider_hash': motion_provider.provider_hash,
        'motion_bake_hash': motion_bakes[0].bake_hash,
        'motion_bakes': motion_bakes,
        'synthetic_binding_policy_only': True,
        'texture_bindings': texture_bindings,
        'native_projection': native_projection,
        'native_runtime_result': native_result,
        'native_runtime_archive': str(native_archive),
    }
