from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

from realsas_compiler_core.types import (
    SkeletonProposalJoint, SkeletonProposalIR,
    SkinInfluenceProposal, SkinProposalIR,
)
from realsas_compiler_core.rig import qualify_skeleton
from realsas_compiler_core.skin import qualify_skin
from realsas_compiler_core.product import assemble_product, bind_proof

from run_exact_compiler_consumer_interlock_v0 import (
    load_fixture, reconstruct, a0, rot_axis_angle,
)

HERE = Path(__file__).resolve().parent
COLLAPSE_SCALE = 0.25
ROTATION_DEG = 15.0
MIN_CORRUPTION_RMS_NORM = 0.05
MIN_POST_COMP_RMSE_NORM = 0.005
MIN_POST_COMP_P95_NORM = 0.010


def collapse_g0(proposal):
    ranked = sorted(proposal.joints, key=lambda j: (-float(j.root_score), j.proposal_id))
    if len(ranked) < 2 or float(ranked[0].root_score) <= float(ranked[1].root_score):
        raise RuntimeError('G0 root is not uniquely determined by root_score')
    root = ranked[0]
    r = np.asarray(root.position, dtype=float)
    joints = []
    for j in proposal.joints:
        p = np.asarray(j.position, dtype=float)
        q = p if j.proposal_id == root.proposal_id else r + COLLAPSE_SCALE * (p - r)
        md = dict(j.metadata)
        md['coupling_probe_corruption'] = 'G0_BAD_COLLAPSE75_V1'
        joints.append(SkeletonProposalJoint(
            j.proposal_id, tuple(map(float, q)), float(j.root_score), float(j.confidence),
            tuple(j.support_surface_ids), md,
        ))
    return SkeletonProposalIR(
        tuple(joints), tuple(proposal.edges), proposal.surface_binding_hash,
        model_provenance=proposal.model_provenance + '+G0_BAD_COLLAPSE75_V1',
        schema_version=proposal.schema_version,
        metadata={**dict(proposal.metadata), 'coupling_probe_corruption':'G0_BAD_COLLAPSE75_V1', 'collapse_scale':COLLAPSE_SCALE},
    )


def posed_geometry(surface, skeleton, skin, rotation_deg=ROTATION_DEG):
    SP = np.asarray([n.P for n in surface.surface_nodes], dtype=float)
    sid_to_i = {n.surface_id:i for i,n in enumerate(surface.surface_nodes)}
    joints = {j.canonical_joint_id:j for j in skeleton.joints}
    root = skeleton.root_id
    transforms = {root:(np.eye(3), np.zeros(3))}
    for jid, j in sorted(joints.items()):
        if jid == root or j.parent_canonical_id is None:
            continue
        parent = joints[j.parent_canonical_id]
        pivot = np.asarray(parent.position, dtype=float)
        v = np.asarray(j.position, dtype=float) - pivot
        axis = np.cross(v, np.array([0., 0., 1.]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.array([1., 0., 0.])
        R = rot_axis_angle(axis, math.radians(rotation_deg))
        transforms[jid] = (R, pivot - R @ pivot)
    out = np.zeros_like(SP)
    mass = np.zeros(len(SP), dtype=float)
    for row in skin.rows:
        i = sid_to_i[row.surface_id]
        for jid, w in row.influences:
            R, t = transforms.get(jid, (np.eye(3), np.zeros(3)))
            out[i] += float(w) * (R @ SP[i] + t)
            mass[i] += float(w)
    if not np.allclose(mass, 1.0, atol=1e-6):
        raise RuntimeError('skin simplex drift in coupling pose')
    return out


def source_maps(skeleton):
    canonical_to_source = {j.canonical_joint_id:j.source_proposal_id for j in skeleton.joints}
    source_to_canonical = {j.source_proposal_id:j.canonical_joint_id for j in skeleton.joints}
    if len(canonical_to_source) != len(source_to_canonical):
        raise RuntimeError('source proposal mapping is not bijective')
    return canonical_to_source, source_to_canonical


def frozen_clean_weights_on_bad(surface, clean_skeleton, clean_skin, bad_skeleton):
    clean_c2s, _ = source_maps(clean_skeleton)
    _, bad_s2c = source_maps(bad_skeleton)
    influences = []
    for row in clean_skin.rows:
        for clean_jid, w in row.influences:
            source = clean_c2s[clean_jid]
            if source not in bad_s2c:
                raise RuntimeError(f'bad skeleton missing source proposal id: {source}')
            influences.append(SkinInfluenceProposal(row.surface_id, bad_s2c[source], float(w)))
    return SkinProposalIR(
        tuple(influences), surface.geometry_lineage_hash, bad_skeleton.skeleton_lineage_hash,
        model_provenance='SACRIFICIAL_A0_FROZEN_CLEAN_WEIGHTS_REMAPPED_V1',
        metadata={'canonical_authority':False, 'diagnostic_only':True, 'mapping':'source_proposal_id'},
    )


def skin_mean_row_l1(clean_skeleton, clean_skin, bad_skeleton, bad_skin):
    clean_c2s, _ = source_maps(clean_skeleton)
    bad_c2s, _ = source_maps(bad_skeleton)
    clean_rows = {}
    for row in clean_skin.rows:
        clean_rows[row.surface_id] = {clean_c2s[jid]:float(w) for jid,w in row.influences}
    bad_rows = {}
    for row in bad_skin.rows:
        bad_rows[row.surface_id] = {bad_c2s[jid]:float(w) for jid,w in row.influences}
    vals = []
    for sid in sorted(clean_rows):
        a = clean_rows[sid]
        b = bad_rows[sid]
        keys = set(a) | set(b)
        vals.append(sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in keys))
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


def skeleton_corruption_rms_norm(surface, clean_skeleton, bad_skeleton):
    clean = {j.source_proposal_id:np.asarray(j.position,float) for j in clean_skeleton.joints}
    bad = {j.source_proposal_id:np.asarray(j.position,float) for j in bad_skeleton.joints}
    if set(clean) != set(bad):
        raise RuntimeError('clean/bad skeleton source sets differ')
    delta = np.asarray([bad[k]-clean[k] for k in sorted(clean)])
    rms = float(np.sqrt(np.mean(np.sum(delta*delta, axis=1))))
    SP = np.asarray([n.P for n in surface.surface_nodes], dtype=float)
    diag = float(np.linalg.norm(SP.max(0)-SP.min(0)))
    return rms, rms/max(diag,1e-12), diag


def response_delta(a, b, diag):
    d = np.linalg.norm(a-b, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))
    return {
        'rmse':rmse,
        'rmse_norm':rmse/max(diag,1e-12),
        'mean_point_delta':float(d.mean()),
        'p95_point_delta':float(np.quantile(d,0.95)),
        'p95_point_delta_norm':float(np.quantile(d,0.95))/max(diag,1e-12),
        'max_point_delta':float(d.max()),
        'finite':bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))),
    }


def main():
    x, fixture_transport_sha = load_fixture()
    surface, clean_g0 = reconstruct(x)
    bad_g0 = collapse_g0(clean_g0)

    clean_qsk = qualify_skeleton(surface, clean_g0, run_ilp_shadow=False)
    bad_qsk = qualify_skeleton(surface, bad_g0, run_ilp_shadow=False)
    if len(clean_qsk.joints) != len(bad_qsk.joints):
        raise RuntimeError('clean/bad qualified joint count differs')

    clean_qskin = qualify_skin(surface, clean_qsk, a0(surface, clean_qsk), max_influences=4)
    bad_recomputed_qskin = qualify_skin(surface, bad_qsk, a0(surface, bad_qsk), max_influences=4)
    frozen_prop = frozen_clean_weights_on_bad(surface, clean_qsk, clean_qskin, bad_qsk)
    bad_frozen_qskin = qualify_skin(surface, bad_qsk, frozen_prop, max_influences=4)

    if not (len(clean_qskin.rows) == len(bad_recomputed_qskin.rows) == len(bad_frozen_qskin.rows) == 512):
        raise RuntimeError('coupling skin routes did not qualify 512/512 rows')

    corruption_rms, corruption_rms_norm, diag = skeleton_corruption_rms_norm(surface, clean_qsk, bad_qsk)
    clean_posed = posed_geometry(surface, clean_qsk, clean_qskin)
    bad_recomputed_posed = posed_geometry(surface, bad_qsk, bad_recomputed_qskin)
    bad_frozen_posed = posed_geometry(surface, bad_qsk, bad_frozen_qskin)

    post = response_delta(bad_recomputed_posed, clean_posed, diag)
    frozen = response_delta(bad_frozen_posed, clean_posed, diag)
    mean_skin_l1, p95_skin_l1 = skin_mean_row_l1(clean_qsk, clean_qskin, bad_qsk, bad_recomputed_qskin)
    compensation_ratio = None if frozen['rmse_norm'] <= 1e-15 else post['rmse_norm']/frozen['rmse_norm']

    hard_checks = {
        'clean_skeleton_qualified':len(clean_qsk.joints) == len(clean_g0.joints),
        'bad_skeleton_qualified':len(bad_qsk.joints) == len(bad_g0.joints),
        'clean_skin_512':len(clean_qskin.rows) == 512,
        'bad_recomputed_skin_512':len(bad_recomputed_qskin.rows) == 512,
        'bad_frozen_skin_512':len(bad_frozen_qskin.rows) == 512,
        'corruption_rms_norm_ge_0p05':corruption_rms_norm >= MIN_CORRUPTION_RMS_NORM,
        'all_posed_finite':bool(np.all(np.isfinite(clean_posed)) and np.all(np.isfinite(bad_recomputed_posed)) and np.all(np.isfinite(bad_frozen_posed))),
        'post_comp_rmse_norm_ge_0p005':post['rmse_norm'] >= MIN_POST_COMP_RMSE_NORM,
        'post_comp_p95_norm_ge_0p010':post['p95_point_delta_norm'] >= MIN_POST_COMP_P95_NORM,
    }
    passed = all(hard_checks.values())

    bad_product = assemble_product(
        surface, bad_qsk, bad_recomputed_qskin,
        deformation_state={'consumer_coupling_probe':{'post_compensation':post,'frozen_weight':frozen}},
        editable_metadata={'consumer_profile':'G0_BAD_COLLAPSE75_A0_V1','sacrificial':True},
    )
    proof = bind_proof(
        bad_product,
        probe_plan={'name':'CONSUMER_COUPLING_COLLAPSE75_ROTATE15_V1','collapse_scale':COLLAPSE_SCALE,'rotation_deg':ROTATION_DEG},
        measurements={'hard_checks':hard_checks,'post_compensation':post,'frozen_weight':frozen},
        passed=passed,
        proof_payload={'consumer_coupling_probe':True,'not_product_quality_claim':True},
    )
    proof_bound = proof.product_state_hash == bad_product.product_state_hash
    if not proof_bound:
        raise RuntimeError('coupling proof is not bound to exact corrupted product state')
    passed = passed and proof_bound

    out = {
        'schema':'RealSaS.ConsumerInterlock.CouplingProbe.v1',
        'asset_id':x['asset_id'],
        'fixture_transport_sha256':fixture_transport_sha,
        'corruption':{
            'name':'G0_BAD_COLLAPSE75_V1',
            'collapse_scale':COLLAPSE_SCALE,
            'joint_position_rms':corruption_rms,
            'joint_position_rms_norm':corruption_rms_norm,
        },
        'rotation_deg':ROTATION_DEG,
        'qualified_joint_count_clean':len(clean_qsk.joints),
        'qualified_joint_count_bad':len(bad_qsk.joints),
        'qualified_skin_rows_clean':len(clean_qskin.rows),
        'qualified_skin_rows_bad_recomputed':len(bad_recomputed_qskin.rows),
        'qualified_skin_rows_bad_frozen_clean_weights':len(bad_frozen_qskin.rows),
        'post_compensation_response_delta':post,
        'frozen_clean_weight_response_delta':frozen,
        'compensation_ratio_diagnostic':compensation_ratio,
        'a0_weight_change_mean_row_l1':mean_skin_l1,
        'a0_weight_change_p95_row_l1':p95_skin_l1,
        'hard_checks':hard_checks,
        'coupling_proof_bound_to_exact_bad_product_state':proof_bound,
        'bad_product_state_hash':bad_product.product_state_hash,
        'clean_skeleton_lineage_hash':clean_qsk.skeleton_lineage_hash,
        'bad_skeleton_lineage_hash':bad_qsk.skeleton_lineage_hash,
        'clean_skin_lineage_hash':clean_qskin.skin_lineage_hash,
        'bad_recomputed_skin_lineage_hash':bad_recomputed_qskin.skin_lineage_hash,
        'status':'PASS' if passed else 'FAIL',
        'interpretation':'PASS means severe G0 geometry corruption remains detectably coupled to final posed surface after A0 is allowed to recompute; compensation_ratio is diagnostic only.',
    }
    op = HERE/'CONSUMER_INTERLOCK_COUPLING_PROBE_RESULT_V1.json'
    op.write_text(json.dumps(out, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(out, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
