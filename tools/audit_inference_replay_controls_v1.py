"""Separate inference-apparatus drift from refined-surface changes."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload,sha256_file
from tools.demo.render_knight_motion_preview_v1 import _ctx


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
def read(path):return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority-root',type=Path,required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--fit-run',type=Path,required=True);ap.add_argument('--arm-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True);ctx=_ctx(args.authority_root,args.run_id)
    surface=stage_output_payload(ctx,'15_RIGGING_SURFACE_QUALIFIED','RealSaS.RiggingSurfaceIR.v1')
    skeleton=stage_output_payload(ctx,'28_SKELETON_QUALIFIED','RealSaS.QualifiedSkeletonIR.v1')
    skin=stage_output_payload(ctx,'32_SKIN_QUALIFIED','RealSaS.QualifiedSkinIR.v1')
    candidate=stage_output_payload(ctx,'18_CANONICAL_MESH_ADDRESSING_BUILD','RealSaS.CanonicalMeshCandidateIR.v1')
    write(out/'original_surface.json',surface);write(out/'original_candidate.json',candidate);write(out/'original_skeleton.json',skeleton)
    write(out/'original_skin.json',skin)
    results={}
    for arm in ('EXISTING_RIG_FRESH_SKIN','FRESH_RIG_FRESH_SKIN'):
        dst=out/arm;dst.mkdir(parents=True,exist_ok=True)
        common=['--surface-json',str(out/'original_surface.json'),'--fit-run',str(args.fit_run),'--out-dir',str(dst)]
        if arm=='EXISTING_RIG_FRESH_SKIN':
            write(dst/'fresh_qualified_skeleton.json',skeleton)
            write(dst/'rig_REPORT.json',{'mode':'EXISTING_QUALIFIED_RIG_CONTROL_NO_RIG_INFERENCE',
                'skeleton_lineage_hash':skeleton['skeleton_lineage_hash']})
        else:
            subprocess.run([sys.executable,'tools/inference/refined_surface_rig_skin_v1.py','--lane','rig',*common],check=True)
        subprocess.run([sys.executable,'tools/inference/refined_surface_rig_skin_v1.py','--lane','skin',*common,'--arm-dir',str(args.arm_dir)],check=True)
        newskin=read(dst/'fresh_qualified_skin.json')
        if arm=='EXISTING_RIG_FRESH_SKIN':
            old={r['surface_id']:dict(r['influences']) for r in skin['rows']};new={r['surface_id']:dict(r['influences']) for r in newskin['rows']}
            assert set(old)==set(new)
            diff=np.array([sum(abs(old[s].get(j,0.)-new[s].get(j,0.)) for j in set(old[s])|set(new[s])) for s in sorted(old)])
            write(dst/'WEIGHT_REPLAY_DELTA.json',{'old_skin_hash':skin['skin_lineage_hash'],'new_skin_hash':newskin['skin_lineage_hash'],
                'row_count':len(diff),'row_l1_max':float(diff.max()),'row_l1_p95':float(np.quantile(diff,.95)),
                'row_l1_mean':float(diff.mean()),'changed_dominant_joint_count':sum(max(old[s],key=old[s].get)!=max(new[s],key=new[s].get) for s in old),
                'comparison_scope':'SAME_ORIGINAL_SURFACE_AND_QUALIFIED_RIG__ARCHIVED_A100_OUTPUT_VS_CPU_FP32_REPLAY'})
        proc=subprocess.run([sys.executable,'tools/audit_fresh_inference_dynamics_v1.py','--authority-root',str(args.authority_root),'--run-id',args.run_id,
            '--surface-json',str(out/'original_surface.json'),'--candidate-json',str(out/'original_candidate.json'),'--inference-dir',str(dst)])
        if proc.returncode not in (0,2):raise RuntimeError('CONTROL_COURT_APPARATUS_ERROR')
        report=read(dst/'DYNAMIC_REPORT.json')
        results[arm]={k:report[k] for k in ['status','g3_passed','failed_motion_frame_count','maximum_motion_edge_ratio']}
        results[arm]['edge_extension_gate_passed']=report['maximum_motion_edge_ratio']<=4.
        if arm=='EXISTING_RIG_FRESH_SKIN':results[arm]['weight_replay_delta']=read(dst/'WEIGHT_REPLAY_DELTA.json')
    write(out/'CONTROL_REPORT.json',{'scope':'ORIGINAL_SURFACE_CPU_FP32_INFERENCE_CONTROLS','arms':results,'product_authority_minted':False})
    print('INFERENCE_CONTROL_RESULTS='+json.dumps(results),flush=True)
    if any(r['failed_motion_frame_count'] or not r['g3_passed'] or not r['edge_extension_gate_passed'] for r in results.values()):raise SystemExit(2)


if __name__=='__main__':main()
