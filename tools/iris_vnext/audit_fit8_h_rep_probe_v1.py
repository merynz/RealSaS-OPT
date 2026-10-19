#!/usr/bin/env python3
from __future__ import annotations

"""Independent held-out H-REP-P0 representation court."""

import argparse, json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.h_rep_probe_v1 import PROBE_ARMS_V1, HRepresentationProbeV1, build_probe_input_v1, h_rep_probe_metrics_v1
from tools.iris_vnext.h_rep_probe_runtime_v1 import cache_frozen_v8_probe_features_v1, load_h_rep_binding_v1
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head, load_json, sha256_file

SCHEMA="RealSaS.HRepP0HeldoutCourt.v1"

def _mean(rows:list[dict[str,float]],key:str)->float: return float(np.mean([r[key] for r in rows]))
def _max(rows:list[dict[str,float]],key:str)->float: return float(max(r[key] for r in rows))
def _min(rows:list[dict[str,float]],key:str)->float: return float(min(r[key] for r in rows))

def audit(args:argparse.Namespace)->dict[str,Any]:
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower()
    if git_head(repo)!=source_sha: raise RuntimeError('H_REP_P0_COURT_SOURCE_SHA_DRIFT')
    binding=load_h_rep_binding_v1(Path(args.binding).resolve()); prereg=load_json(Path(args.preregistration).resolve())
    if prereg.get('schema')!='RealSaS.HRepP0Preregistration.v1': raise ValueError('H_REP_P0_COURT_PREREG_INVALID')
    training=load_json(Path(args.training_result).resolve())
    if training.get('status')!='COMPLETED_H_REP_P0_EQUAL_CAPACITY_PROBES__AWAITING_INDEPENDENT_COURT': raise RuntimeError('H_REP_P0_COURT_TRAINING_NOT_COMPLETE')
    checkpoint=Path(args.checkpoint).resolve()
    if sha256_file(checkpoint)!=training.get('checkpoint_sha256'): raise ValueError('H_REP_P0_COURT_CHECKPOINT_SHA_DRIFT')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if cp.get('schema')!='RealSaS.HRepP0Checkpoint.v1' or int(cp.get('completed_steps',-1))!=1200: raise ValueError('H_REP_P0_COURT_CHECKPOINT_INVALID')
    device=torch.device('cuda'); cache,scene_hash=cache_frozen_v8_probe_features_v1(v8_checkpoint=Path(args.v8_checkpoint),prepared_root=Path(args.prepared_root),prepared_manifest=Path(args.prepared_manifest),dataset_root=Path(args.dataset_root),dataset_receipt=Path(args.dataset_receipt),binding=binding,split='holdout',device=device,query_chunk=int(args.query_chunk))
    if scene_hash!=training.get('v8_scene_core_sha256'): raise RuntimeError('H_REP_P0_COURT_SCENE_HASH_DRIFT')
    models={arm:HRepresentationProbeV1(arm).to(device).eval() for arm in PROBE_ARMS_V1}
    for arm in PROBE_ARMS_V1: models[arm].load_state_dict(cp['models'][arm],strict=True)
    subject_results=[]
    with torch.no_grad():
        for subject_id in SUBJECT_IDS:
            item=cache[subject_id]; n=len(item['xyz_iris']); arm_metrics={}
            for arm in PROBE_ARMS_V1:
                chunks=[]
                for start in range(0,n,int(args.eval_chunk)):
                    sl=slice(start,min(n,start+int(args.eval_chunk))); h=item['H'][sl].to(device); xyz=item['xyz_iris'][sl].to(device); field=item['raw_field'][sl].to(device)
                    shuffled=item['H'][item['shuffle_index'][sl]].to(device) if arm=='SHUFFLED_H_XYZ' else None
                    x=build_probe_input_v1(arm,h=h,xyz_iris=xyz,raw_field=field,shuffled_h=shuffled); chunks.append(models[arm](x))
                pred={k:torch.cat([c[k] for c in chunks],dim=0) for k in chunks[0]}
                arm_metrics[arm]=h_rep_probe_metrics_v1(pred,xyz_iris=item['xyz_iris'].to(device),nearest_reference_iris=item['nearest_reference_iris'].to(device),distance_iris=item['distance_iris'].to(device),normal_unit=item['normal_unit'].to(device),normal_valid=item['normal_valid'].to(device),near_label=item['near_label'].to(device))
            subject_results.append({'subject_id':subject_id,'arms':arm_metrics})
            print('H_REP_P0_COURT_SUBJECT',subject_id,json.dumps(arm_metrics,sort_keys=True),flush=True)
    aggregate={}
    for arm in PROBE_ARMS_V1:
        rows=[s['arms'][arm] for s in subject_results]
        aggregate[arm]={
            'mean_distance_mae_iris':_mean(rows,'distance_mae_iris'),
            'maximum_subject_distance_mae_iris':_max(rows,'distance_mae_iris'),
            'mean_distance_abs_error_p95_iris':_mean(rows,'distance_abs_error_p95_iris'),
            'maximum_subject_endpoint_error_p95_iris':_max(rows,'endpoint_error_p95_iris'),
            'minimum_subject_normal_absolute_dot_mean':_min(rows,'normal_absolute_dot_mean'),
            'minimum_subject_near_balanced_accuracy':_min(rows,'near_balanced_accuracy'),
            'mean_near_balanced_accuracy':_mean(rows,'near_balanced_accuracy'),
        }
    t=prereg['heldout_court']; h=aggregate['H_XYZ']; raw=aggregate['RAW_FIELD_XYZ']; xyz=aggregate['XYZ_ONLY']; shuf=aggregate['SHUFFLED_H_XYZ']
    def reduction(a:float,b:float)->float: return 1.0-a/max(b,1e-12)
    h_absolute={
        'mean_distance_mae':h['mean_distance_mae_iris']<=float(t['maximum_h_mean_distance_mae_iris']),
        'max_subject_endpoint_p95':h['maximum_subject_endpoint_error_p95_iris']<=float(t['maximum_h_subject_endpoint_p95_iris']),
        'min_subject_normal':h['minimum_subject_normal_absolute_dot_mean']>=float(t['minimum_h_subject_normal_alignment']),
        'min_subject_near_ba':h['minimum_subject_near_balanced_accuracy']>=float(t['minimum_h_subject_near_balanced_accuracy']),
    }
    h_alignment={
        'distance_reduction_vs_xyz':reduction(h['mean_distance_mae_iris'],xyz['mean_distance_mae_iris'])>=float(t['minimum_h_distance_mae_reduction_vs_xyz']),
        'distance_reduction_vs_shuffled':reduction(h['mean_distance_mae_iris'],shuf['mean_distance_mae_iris'])>=float(t['minimum_h_distance_mae_reduction_vs_shuffled']),
        'near_ba_gain_vs_xyz':h['mean_near_balanced_accuracy']-xyz['mean_near_balanced_accuracy']>=float(t['minimum_h_near_ba_gain_vs_xyz']),
        'near_ba_gain_vs_shuffled':h['mean_near_balanced_accuracy']-shuf['mean_near_balanced_accuracy']>=float(t['minimum_h_near_ba_gain_vs_shuffled']),
    }
    raw_absolute={
        'mean_distance_mae':raw['mean_distance_mae_iris']<=float(t['maximum_raw_mean_distance_mae_iris']),
        'max_subject_endpoint_p95':raw['maximum_subject_endpoint_error_p95_iris']<=float(t['maximum_raw_subject_endpoint_p95_iris']),
        'min_subject_normal':raw['minimum_subject_normal_absolute_dot_mean']>=float(t['minimum_raw_subject_normal_alignment']),
        'min_subject_near_ba':raw['minimum_subject_near_balanced_accuracy']>=float(t['minimum_raw_subject_near_balanced_accuracy']),
        'distance_reduction_vs_xyz':reduction(raw['mean_distance_mae_iris'],xyz['mean_distance_mae_iris'])>=float(t['minimum_raw_distance_mae_reduction_vs_xyz']),
        'near_ba_gain_vs_xyz':raw['mean_near_balanced_accuracy']-xyz['mean_near_balanced_accuracy']>=float(t['minimum_raw_near_ba_gain_vs_xyz']),
    }
    h_pass=all(h_absolute.values()) and all(h_alignment.values()); raw_pass=all(raw_absolute.values()); carrier_decodable=h_pass or raw_pass
    status='SCIENTIFIC_PASS_H_REP_P0_LOCAL_GEOMETRY_DECODABLE' if carrier_decodable else 'SCIENTIFIC_FAIL_H_REP_P0_LOCAL_GEOMETRY_DECODABILITY'
    next_action='DESIGN_EXTRACTION_TRANSITION_ALTERNATIVE__DO_NOT_RETRAIN_H_YET__DO_NOT_TRAIN_TESSA' if carrier_decodable else 'PREREGISTER_CARRIER_REPRESENTATION_RETRAINING_EXPERIMENT__DO_NOT_TRAIN_TESSA'
    result={'schema':SCHEMA,'status':status,'execution_source_sha':source_sha,'training_checkpoint_sha256':sha256_file(checkpoint),'v8_scene_core_sha256':scene_hash,'subjects':subject_results,'aggregate':aggregate,'gates':{'h_absolute':h_absolute,'h_local_alignment':h_alignment,'raw_field_absolute_and_xyz_control':raw_absolute},'derived_comparisons':{'h_distance_mae_reduction_vs_xyz':reduction(h['mean_distance_mae_iris'],xyz['mean_distance_mae_iris']),'h_distance_mae_reduction_vs_shuffled':reduction(h['mean_distance_mae_iris'],shuf['mean_distance_mae_iris']),'h_near_ba_gain_vs_xyz':h['mean_near_balanced_accuracy']-xyz['mean_near_balanced_accuracy'],'h_near_ba_gain_vs_shuffled':h['mean_near_balanced_accuracy']-shuf['mean_near_balanced_accuracy'],'raw_distance_mae_reduction_vs_xyz':reduction(raw['mean_distance_mae_iris'],xyz['mean_distance_mae_iris']),'raw_near_ba_gain_vs_xyz':raw['mean_near_balanced_accuracy']-xyz['mean_near_balanced_accuracy']},'h_local_geometry_decodable':h_pass,'raw_field_local_geometry_decodable':raw_pass,'carrier_local_geometry_decodable':carrier_decodable,'point_cloud_proxy_only':True,'mesh_extraction_invoked':False,'tessa_invoked':False,'optimizer_steps_executed_by_court':0,'scientific_pass_claimed':carrier_decodable,'next_action':next_action}
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print('H_REP_P0_COURT',status,json.dumps(aggregate,sort_keys=True),flush=True); return result

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo-root',required=True); p.add_argument('--expected-source-sha',required=True); p.add_argument('--binding',default='canonical/H_REP_P0_INPUT_BINDING_V1.json'); p.add_argument('--preregistration',default='canonical/H_REP_P0_PREREGISTRATION_V1.json'); p.add_argument('--training-result',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset-root',required=True); p.add_argument('--dataset-receipt',required=True); p.add_argument('--prepared-root',required=True); p.add_argument('--prepared-manifest',required=True); p.add_argument('--v8-checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--query-chunk',type=int,default=8192); p.add_argument('--eval-chunk',type=int,default=4096); audit(p.parse_args()); return 0
if __name__=='__main__': raise SystemExit(main())
