from __future__ import annotations

import argparse, io, json, zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash, mesh_lineage_hash, qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1, evaluate_mesh_quality, mesh_raster_quality_report,
)
import experiments.mage_demo_fit1_v5_p1.materialize_p1_b2_g10_v1 as mat
import experiments.mage_full_subject_reclosure_v1.run_fit2_baseline_preserving_adaptive_patch_cdt_v1 as patch_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2

SCHEMA = 'RealSaS.MageDemo.P1B2G10CurrentAuthorityMaterialization.v1'


def _sha(path: Path) -> str:
    h=sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, separators=(',', ': '))+'\n', encoding='utf-8')


def _npy_bytes(a):
    s=io.BytesIO(); np.lib.format.write_array(s,np.asarray(a),allow_pickle=False); return s.getvalue()


def _write_npz(path: Path, arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            info=zipfile.ZipInfo(f'{k}.npy',date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.create_system=3; info.external_attr=0o644<<16
            z.writestr(info,_npy_bytes(np.asarray(arrays[k])))


def _project_candidate(candidate, historical_surface_lineage: str):
    md=dict(candidate.metadata or {})
    if md.get('surface_lineage_hash') != candidate.surface_binding_hash:
        raise RuntimeError('CURRENT_CANDIDATE_SURFACE_METADATA_INCONSISTENT')
    md['surface_lineage_hash']=historical_surface_lineage
    p=replace(candidate, surface_binding_hash=historical_surface_lineage, candidate_lineage_hash='', metadata=md)
    return replace(p,candidate_lineage_hash=mesh_candidate_lineage_hash(p))


def _project_mesh(mesh, projected_candidate, historical_surface_lineage: str):
    qr=dict(mesh.qualification_report or {})
    md=dict(mesh.metadata or {})
    qr['candidate_lineage_hash']=projected_candidate.candidate_lineage_hash
    md['source_candidate_lineage_hash']=projected_candidate.candidate_lineage_hash
    p=replace(mesh, surface_binding_hash=historical_surface_lineage, qualification_report=qr, metadata=md, mesh_lineage_hash='')
    return replace(p,mesh_lineage_hash=mesh_lineage_hash(p))


def run(args):
    historical=json.load(open(args.historical_result,encoding='utf-8'))
    old_gsa=historical['gsa_replay']['actual_gsa_lineage_hash']
    old_sig=historical['gsa_replay']['cdt_relevant_mechanical_raster_signature']
    old_p1=historical['treatments'][0]
    if old_p1['name'] != 'P1_B2_G10': raise RuntimeError('HISTORICAL_P1_NOT_FIRST_TREATMENT')
    old_by_view={int(r['view']):r for r in old_p1['views']}

    surface, tensor, replay=ceiling_v2._preflight_surface(args)
    print('CURRENT_GSA_REPLAY='+json.dumps(replay,sort_keys=True),flush=True)
    if not replay['gsa_lineage_exact_match']:
        raise RuntimeError('CURRENT_CANONICAL_GSA_LINEAGE_NOT_EXACT')
    global_signature_exact = replay['cdt_relevant_mechanical_raster_signature'] == old_sig
    print('GLOBAL_CDT_SIGNATURE_EXACT='+str(global_signature_exact), flush=True)
    if old_gsa == replay['actual_gsa_lineage_hash']:
        raise RuntimeError('HISTORICAL_PROJECTION_NOT_NEEDED_UNEXPECTEDLY')

    obs=tuple(Path(p) for p in args.observations); cams=tuple(Path(p) for p in args.cameras)
    baselines,domains=mat._baseline_inputs(surface,obs,cams)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; artifacts=[]
    for view in range(8):
        candidate=patch_v1._build_hybrid_candidate(surface,baselines[view],view=view,camera_hash=ceiling_v1.CAMERA_SHA256[view],domain=domains[view],treatment=mat.TREATMENT)
        mesh=qualify_supported_mesh(surface,candidate)
        projected_candidate=_project_candidate(candidate,old_gsa)
        projected_mesh=_project_mesh(mesh,projected_candidate,old_gsa)
        old=old_by_view[view]
        candidate_match=projected_candidate.candidate_lineage_hash == old['candidate_lineage_hash']
        mesh_match=projected_mesh.mesh_lineage_hash == old['mesh_lineage_hash'] == mat.EXPECTED_MESH_LINEAGE[view]
        if not candidate_match or not mesh_match:
            raise RuntimeError(f'HISTORICAL_LINEAGE_PROJECTION_FAILED_V{view}:candidate={projected_candidate.candidate_lineage_hash}:{old["candidate_lineage_hash"]}:mesh={projected_mesh.mesh_lineage_hash}:{old["mesh_lineage_hash"]}')
        quality=evaluate_mesh_quality(coverage=dict(candidate.residual_report),raster_report=mesh_raster_quality_report(mesh,surface=surface,view_index=view),policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1)
        checks={
            'vertex_count': len(mesh.vertices)==int(old['vertex_count']),
            'face_count': len(mesh.faces)==int(old['face_count']),
            'edge_count': len(mesh.edges)==int(old['edge_count']),
            'source_alpha_recall': float(quality['source_alpha_recall'])==float(old['source_alpha_recall']),
            'precision_inside_alpha': float(quality['precision_inside_alpha'])==float(old['precision_inside_alpha']),
            'alpha_iou': float(quality['alpha_iou'])==float(old['alpha_iou']),
            'min_raster_triangle_angle_deg': float(quality['min_raster_triangle_angle_deg'])==float(old['min_raster_triangle_angle_deg']),
            'max_raster_triangle_aspect_ratio': float(quality['max_raster_triangle_aspect_ratio'])==float(old['max_raster_triangle_aspect_ratio']),
        }
        if not all(checks.values()): raise RuntimeError(f'HISTORICAL_GEOMETRY_RASTER_EQUIVALENCE_FAILED_V{view}:{checks}')

        jname=f'P1_B2_G10_CURRENT_V{view}_QUALIFIED_MESH_IR.json'; nname=f'P1_B2_G10_CURRENT_V{view}_GEOMETRY.npz'
        jp=out/jname; npz=out/nname
        _write_json(jp,mesh.to_dict()); _write_npz(npz,mat._mesh_npz_arrays(mesh))
        for p in (jp,npz): artifacts.append({'path':p.name,'sha256':_sha(p),'bytes':p.stat().st_size})
        row={
            'view':view,
            'current_surface_lineage_hash':surface.geometry_lineage_hash,
            'historical_surface_lineage_hash':old_gsa,
            'current_candidate_lineage_hash':candidate.candidate_lineage_hash,
            'historical_projected_candidate_lineage_hash':projected_candidate.candidate_lineage_hash,
            'historical_expected_candidate_lineage_hash':old['candidate_lineage_hash'],
            'current_mesh_lineage_hash':mesh.mesh_lineage_hash,
            'historical_projected_mesh_lineage_hash':projected_mesh.mesh_lineage_hash,
            'historical_expected_mesh_lineage_hash':old['mesh_lineage_hash'],
            'historical_projection_exact':True,
            'geometry_raster_equivalence_checks':checks,
            'vertex_count':len(mesh.vertices),'face_count':len(mesh.faces),'edge_count':len(mesh.edges),
            'source_alpha_recall':float(quality['source_alpha_recall']),'precision_inside_alpha':float(quality['precision_inside_alpha']),
            'alpha_iou':float(quality['alpha_iou']),'min_raster_triangle_angle_deg':float(quality['min_raster_triangle_angle_deg']),
            'max_raster_triangle_aspect_ratio':float(quality['max_raster_triangle_aspect_ratio']),
            'full_frozen_policy_pass':bool(quality['passed']),'failure_invariants':list(quality['failure_invariants']),
            'qualified_mesh_json':jname,'geometry_npz':nname,
        }
        rows.append(row)
        print('P1_CURRENT_V'+str(view)+'='+json.dumps(row,sort_keys=True),flush=True)

    manifest={
      'schema':SCHEMA,
      'status':'PASS__CURRENT_CANONICAL_GSA_P1_B2_G10_MATERIALIZED__HISTORICAL_HASH_PROJECTION_EXACT',
      'source_snapshot_commit':'2800a22df41471f00e6f9cf2406e8f4accb757d1',
      'treatment':dict(mat.TREATMENT),
      'current_canonical_gsa_lineage_hash':replay['actual_gsa_lineage_hash'],
      'historical_numeric_run_gsa_lineage_hash':old_gsa,
      'cdt_relevant_mechanical_raster_signature':replay['cdt_relevant_mechanical_raster_signature'],
      'historical_cdt_relevant_mechanical_raster_signature':old_sig,
      'global_cdt_mechanical_raster_signature_exact':global_signature_exact,
      'product_equivalence_authority':'PER_VIEW_EXACT_CANDIDATE_AND_MESH_HASH_PROJECTION',
      'historical_lineage_relabelled_as_current':False,
      'scientific_mechanical_surface_relabelled':False,
      'historical_hashes_used_as_provenance_only':True,
      'current_mesh_hashes_are_product_materialization_authority':True,
      'mesh_scientific_pass_claimed':False,
      'product_pass_claimed':False,
      'views':rows,'artifacts':artifacts,
    }
    mp=out/'P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json'; _write_json(mp,manifest)
    seal={'schema':SCHEMA+'.Seal.v1','status':'SEALED__CURRENT_AUTHORITY_P1_B2_G10_V0_V7__HISTORICAL_PROJECTION_PROVEN','manifest':mp.name,'manifest_sha256':_sha(mp),'artifacts':artifacts,'artifact_count':len(artifacts),'product_pass_claimed':False}
    sp=out/'P1_B2_G10_CURRENT_AUTHORITY_MATERIALIZATION_SEAL.json'; _write_json(sp,seal)
    print('P1_CURRENT_AUTHORITY_SEAL='+json.dumps(seal,sort_keys=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--zero-surface',required=True); p.add_argument('--cameras',nargs=8,required=True); p.add_argument('--observations',nargs=8,required=True); p.add_argument('--historical-result',required=True); p.add_argument('--output-dir',required=True); run(p.parse_args())
