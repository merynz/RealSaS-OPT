from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
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
from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1
from experiments.single_family_e2e_v1.export_bundle_v1 import export_product_bundle_v1
from runtime.reference_v4.consumer import consume_product_v4_reference


NATIVE_RESOLUTION = 1024


def _grid_to_pixel_center_xy(x, y):
    return (
        (float(x) + 1.0) * 0.5 * NATIVE_RESOLUTION - 0.5,
        (float(y) + 1.0) * 0.5 * NATIVE_RESOLUTION - 0.5,
    )


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
        for ji, jid in enumerate(jids):
            influences.append(SkinInfluenceProposal(node.surface_id, jid, 1.0 / len(jids)))
    return SkinProposalIR(tuple(influences), surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, 'MOCK_ARACHNE_V2', metadata={'teacher_truth_used': False})


def _visuals(mechanical):
    dirs = []; obs = {v: content_sha256({'mock_observation_view': v}) for v in range(8)}
    for v in range(8):
        cam = content_sha256({'mock_exact_camera': v}); cand = build_mwb2_candidate(mechanical.surface, view_index=v, camera_binding_hash=cam); mesh = qualify_mwb2_mesh(mechanical.surface, cand); mskin = bind_mwb2_mesh_skin(mechanical.surface, mechanical.skeleton, mechanical.skin, mesh)
        app = build_observed_appearance_binding(surface=mechanical.surface, mesh=mesh, target_view_index=v, camera_binding_hash=cam, observation_hash_by_view=obs, atlas_payload_hash=content_sha256({'mock_rgba_view': v}))
        comp = build_renderable_component(component_id='BODY', view_index=v, mesh=mesh, mesh_skin=mskin, appearance=app, setup_order=0, coverage_classification='OBSERVED_SAFE_LOCAL_RELATION_COMPLEX', metadata={'mock_learned_e2e': True})
        dirs.append(build_directional_renderable(view_index=v, camera_binding_hash=cam, components=(comp,), metadata={'mock_exact_camera': True}))
    return build_directional_renderable_set(tuple(dirs), metadata={'mock_learned_e2e': True})


def _deformation_fixture(product):
    comp = product.directional_renderables.directions[0].components[0]; p, w, jids = mesh_skin_dense_weights(comp.mesh, comp.mesh_skin, product.mechanical_state.skeleton); T = np.tile(np.eye(4, dtype=np.float64), (1, len(jids), 1, 1))
    if len(jids) > 1:
        T[0, 1, 0, 3] = 0.1
    expected = apply_verified_lbs_v1(p, w, T)
    return {'view_index': 0, 'component_index': 0, 'transforms': T, 'expected': expected, 'rms_threshold': 1e-7, 'p95_threshold': 1e-7}


def run_complete_e2e_v1(output_root=None):
    """Synthetic architecture gate: mock learned proposals through real Compiler/proof/runtime."""
    surface = _mock_surface(); skeleton = qualify_skeleton_v2(surface, _mock_skeleton(surface)); skin = qualify_skin(surface, skeleton, _mock_skin(surface, skeleton)); mechanical = build_mechanical_state(surface, skeleton, skin); visuals = _visuals(mechanical); motion = build_deterministic_preset_motion(mechanical)
    capability = make_single_family_e2e_capability_contract(base_lbs_hash=content_sha256({'impl': 'verified_lbs_v1'}), mesh_hash=content_sha256({'mesh': tuple(d.components[0].mesh.mesh_lineage_hash for d in visuals.directions)}), skinning_hash=skin.skin_lineage_hash, visual_hash=visuals.directional_visual_state_hash, preset_motion_hash=motion.motion_state_hash, runtime_hash=content_sha256({'runtime': 'reference_v4'}), policy_hash=content_sha256({'policy': 'single_family_e2e_v1'}))
    product = assemble_product_v3(mechanical, visuals, capability, motion, editable_metadata={'synthetic_source_gate': True}, runtime_policy={'backend': 'REFERENCE_V4'})
    proof = evaluate_product_proof(product, deformation_fixture=_deformation_fixture(product))
    if proof.overall_status != 'PASS':
        raise RuntimeError(f'SYNTHETIC_E2E_PROOF_NOT_PASS:{proof.overall_status}')
    root = Path(output_root) if output_root is not None else Path(tempfile.mkdtemp(prefix='realsas_e2e_'))
    runtime, manifest = export_product_bundle_v1(root, product=product, proof_bundle=proof); consume = consume_product_v4_reference(product, proof, runtime)
    return {'status': 'PASS_COMPLETE_SYNTHETIC_E2E_V1', 'product': product, 'proof': proof, 'runtime': runtime, 'runtime_report': consume, 'bundle_manifest': manifest, 'output_root': str(root), 'truth_paths_consumed': False, 'generalization_claim': False, 'scientific_fit_steps': 0}
