#!/usr/bin/env python3
"""Execute sealed real-morphology -> V3.1 extraction-bound selection court."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from tools.ops.inspect_fit8_morphology_scale_census_v1 import (
    _safe_authority_path, file_sha256, mask_internal_gap_runs,
    mask_local_thickness_px, nearest_rank_quantiles, tessa_units_per_pixel,
)
ROOT=Path(__file__).resolve().parents[2]
PREREG_PATH=ROOT/'canonical/V31_EXTRACTION_RESOLUTION_SELECTION_PREREG_V1.json'
REQUEST_PATH=ROOT/'canonical/NATIVE_TEACHER_REPLAY_REQUEST_V1.json'
OBS_POLICY_PATH=ROOT/'canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json'
SCHEMA='RealSaS.V31ExtractionResolutionSelectionCourt.v1'

def q05(v):
    x=nearest_rank_quantiles(v)['p05']
    if x is None: raise RuntimeError('V31_EXTRACTION_RESOLUTION_RESOLVED_SCALE_EMPTY')
    return float(x)

def choose_resolution(thickness_q05,gap_q05,candidates):
    for r in candidates:
        if math.sqrt(2.0)/r <= thickness_q05 and 1.0/r <= gap_q05: return int(r)
    return None

def resolved(values_px,scale,floor_px):
    raw=np.asarray(values_px,dtype=np.float64).reshape(-1)
    if not len(raw): return np.asarray([],dtype=np.float64),0
    keep=raw>float(floor_px)
    return raw[keep]*float(scale),int((~keep).sum())

def run(authority_root:Path,expected_main_sha:str):
    prereg=json.loads(PREREG_PATH.read_text())
    if prereg.get('schema')!='RealSaS.V31ExtractionResolutionSelectionPrereg.v1' or prereg.get('status')!='SEALED_BEFORE_SELECTION_EXECUTION':
        raise RuntimeError('V31_EXTRACTION_RESOLUTION_PREREG_INVALID')
    parent=prereg['parent_census']; census_path=_safe_authority_path(authority_root,parent['authority_relative_path'])
    if not census_path.is_file() or file_sha256(census_path)!=parent['sha256']:
        raise RuntimeError('V31_EXTRACTION_RESOLUTION_PARENT_CENSUS_BYTES_DRIFT')
    census=json.loads(census_path.read_text())
    if (census.get('schema')!=parent['schema'] or census.get('status')!=parent['status'] or
        census.get('source_code_sha')!=parent['source_code_sha'] or
        census.get('selection',{}).get('production_resolution_selected') is not False):
        raise RuntimeError('V31_EXTRACTION_RESOLUTION_PARENT_CENSUS_CONTRACT_DRIFT')
    obs=json.loads(OBS_POLICY_PATH.read_text()); sp=prereg['source_observation_policy']; floor=float(sp['source_uncertainty_floor_px'])
    if int(obs.get('resolution',0))!=int(sp['resolution_px']) or float(obs.get(sp['source_uncertainty_floor_field'],-1))!=floor:
        raise RuntimeError('V31_EXTRACTION_RESOLUTION_SOURCE_FLOOR_DRIFT')
    subjects=list(json.loads(REQUEST_PATH.read_text()).get('subjects') or ())
    if len(subjects)!=8: raise RuntimeError('V31_EXTRACTION_RESOLUTION_SUBJECT_COUNT_DRIFT')
    candidates=[int(x) for x in prereg['candidate_uniform_resolutions']]
    if candidates!=sorted(set(candidates)) or candidates[-1]!=int(prereg['selection_rule']['uniform_resolution_ceiling']):
        raise RuntimeError('V31_EXTRACTION_RESOLUTION_CANDIDATE_DRIFT')
    rows=[]
    for subject in subjects:
        sid=str(subject['subject_id']); ar=subject['observation_admission']; proof_path=_safe_authority_path(authority_root,ar['relative'])
        if not proof_path.is_file() or file_sha256(proof_path)!=ar['sha256']: raise RuntimeError(f'V31_EXTRACTION_RESOLUTION_ADMISSION_DRIFT:{sid}')
        proof=json.loads(proof_path.read_text())
        if proof.get('status')!='PASS_IRIS_BIND_OBSERVATIONS' or proof.get('subject_id')!=sid: raise RuntimeError(f'V31_EXTRACTION_RESOLUTION_ADMISSION_CONTRACT:{sid}')
        folder=proof_path.parent; camera_path=folder/'camera_bundle.json'
        if file_sha256(camera_path)!=proof['camera_sha256']: raise RuntimeError(f'V31_EXTRACTION_RESOLUTION_CAMERA_DRIFT:{sid}')
        camera=json.loads(camera_path.read_text()); common=float(camera['common_half_extent']); cameras=list(camera['cameras']); pv={int(v['view_index']):v for v in proof['views']}
        ts=[]; gs=[]; tt=tu=gt=gu=0
        for i,cam in enumerate(cameras):
            mp=folder/f'V{i}.mask.bin'
            if file_sha256(mp)!=pv[i]['mask_sha256']: raise RuntimeError(f'V31_EXTRACTION_RESOLUTION_MASK_DRIFT:{sid}:{i}')
            mask=np.frombuffer(mp.read_bytes(),dtype=np.uint8).reshape(1024,1024)>0
            tp=mask_local_thickness_px(mask); gp=mask_internal_gap_runs(mask); scale=tessa_units_per_pixel(cam['half_extent'],common,1024)
            tr,tun=resolved(tp,scale,floor); gr,gun=resolved(gp,scale,floor)
            tt+=len(tp); tu+=tun; gt+=len(gp); gu+=gun
            if len(tr): ts.append(tr)
            if len(gr): gs.append(gr)
        if not ts or not gs: raise RuntimeError(f'V31_EXTRACTION_RESOLUTION_NO_RESOLVED_SCALE:{sid}')
        tq=q05(np.concatenate(ts)); gq=q05(np.concatenate(gs)); selected=choose_resolution(tq,gq,candidates)
        rows.append({'subject_id':sid,'resolved_local_thickness_q05_tessa':tq,'resolved_internal_gap_q05_tessa':gq,
          'local_thickness_total_samples':tt,'local_thickness_source_unresolved_samples':tu,'local_thickness_source_unresolved_fraction':float(tu/tt),
          'internal_gap_total_samples':gt,'internal_gap_source_unresolved_samples':gu,'internal_gap_source_unresolved_fraction':float(gu/gt) if gt else 0.0,
          'required_continuous_resolution_from_thickness':int(math.ceil(math.sqrt(2.0)/tq)),
          'required_continuous_resolution_from_gap':int(math.ceil(1.0/gq)),
          'selected_uniform_resolution':selected,'bound_sufficient':selected is not None})
    failed=[r['subject_id'] for r in rows if not r['bound_sufficient']]
    if failed:
        status='V31_UNIFORM_BOUND_INSUFFICIENT__LOCAL_SPARSE_REFINEMENT_REQUIRED_V1'; bound=None; next_action='DESIGN_AND_PREREGISTER_LOCAL_SPARSE_OR_HIERARCHICAL_EXTRACTION_REFINEMENT'
    else:
        status='V31_BOUNDED_EXTRACTION_RESOLUTION_POLICY_SELECTED_V1'; bound=max(int(r['selected_uniform_resolution']) for r in rows); next_action='IMPLEMENT_V31_OBSERVABILITY_GATED_EXTRACTION_BRIDGE_WITH_SELECTED_BOUND'
    return {'schema':SCHEMA,'status':status,'source_code_sha':str(expected_main_sha).lower(),'prereg_sha256':file_sha256(PREREG_PATH),
      'parent_census_sha256':parent['sha256'],'source_uncertainty_floor_px':floor,'resolved_lower_quantile':float(prereg['measurement_binding']['resolved_lower_quantile']),
      'candidate_uniform_resolutions':candidates,'policy_form':prereg['selection_rule']['policy_form'],'subjects':rows,
      'selection':{'selected_uniform_resolution_bound':bound,'uniform_bound_sufficient_for_all_fit8_subjects':not bool(failed),'subjects_exceeding_uniform_ceiling':failed,'maximum_candidate_resolution':candidates[-1],'next_action':next_action},
      'training_executed':False,'model_release_minted':False,'model_promotion_performed':False,'product_shape_activated':False,'shape_proposal_minted':False,'canonical_mesh_minted':False,'product_revision_minted':False,'fit8_pass_claimed':False,'genericity_claimed':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--authority-root',type=Path,required=True); p.add_argument('--expected-main-sha',required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    result=run(a.authority_root.resolve(),a.expected_main_sha); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'selection':result['selection'],'subjects':result['subjects'],'out':str(a.out)},sort_keys=True))
if __name__=='__main__': main()
