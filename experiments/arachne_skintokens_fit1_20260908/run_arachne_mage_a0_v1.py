from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch

from compiler.realsas_compiler_core.substrate.scene_first_signed import rigging_surface_from_scene_first_zero_mesh_v1
from compiler.realsas_compiler_core.types import QualifiedJoint, SkinInfluenceProposal, SkinProposalIR
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from compiler.realsas_compiler_core.skin import qualify_skin
from models.skin_field_codec.v1.skin_field_codec_v1 import SkinFieldCodecV1, skin_field_codec_loss_v1
from models.skin_field_codec.v1.codec_deformation_loss_v1 import codec_deformation_loss_v1, torch_verified_lbs_v1

SCHEMA='RealSaS.ArachneMageA0Run.v1'
SEED=20260908
MAX_STEPS=1536
CHECK_EVERY=32
CHECKPOINT_EVERY=128
REQUIRED_STABLE=3
LR=1e-3
WEIGHT_DECAY=1e-4
EXPECTED_CODEC_HASH='24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715'
EXPECTED_CACHE_SHA='12484afc23d5c03cbad8020266ed5b39c3201d979e78f96380e748902152be6e'
EXPECTED_SURFACE_HASH='67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb'
EXPECTED_SKELETON_HASH='738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306'
EXPECTED_BINDING_HASH='cb41eb7055b8e2646628daecdd0e31dfc079d163d5f5adaaa1a92f1ca1dfb994'
ZERO_SURFACE_SHA='987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b'
SOURCE_RUN_ID='20260904T220929Z'
IRIS_CHECKPOINT_SHA='766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2'
CAMERA_SHA=(
'73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541',
'bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4',
'3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c',
'24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d',
'5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39',
'7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe',
'daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec',
'68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f')


def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p:Path,obj): p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=float)+'\n',encoding='utf-8')

def load_surface(zero:Path,camera_dir:Path):
    if sha(zero)!=ZERO_SURFACE_SHA: raise RuntimeError('ZERO_SURFACE_SHA_DRIFT')
    cams=[]
    for i,exp in enumerate(CAMERA_SHA):
        p=camera_dir/f'V{i}.camera.json'
        if sha(p)!=exp: raise RuntimeError(f'CAMERA_SHA_DRIFT_V{i}')
        cams.append(json.loads(p.read_text()))
    with np.load(zero,allow_pickle=False) as z:
        world=np.asarray(z['vertices'],np.float64); faces=np.asarray(z['faces'],np.int64); hints=np.asarray(z['normals'],np.float64)
    center=np.asarray(cams[0]['center'],np.float64); half=float(cams[0]['half_extent'])
    surface=rigging_surface_from_scene_first_zero_mesh_v1((world-center[None,:])/half,faces,hints,tuple(cams),
        normalization_center=center,normalization_half_extent=half,authority_label='IRIS_SCENE_FIRST_SIGNED_V3_PROMOTED_MAGE_FIT',
        source_run_id=SOURCE_RUN_ID,source_checkpoint_sha256=IRIS_CHECKPOINT_SHA,source_zero_surface_sha256=ZERO_SURFACE_SHA,
        target_nodes=1024,normal_k=64,visibility_depth_tolerance_norm=0.02,
        metadata={'camera_contract':'CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG','teacher_truth_used':False,'geppetto_reference_strength_fit1':True})
    if surface.geometry_lineage_hash!=EXPECTED_SURFACE_HASH: raise RuntimeError('SURFACE_LINEAGE_DRIFT')
    if len(surface.surface_nodes)!=950 or len(surface.local_relations)!=2813: raise RuntimeError('SURFACE_SHAPE_DRIFT')
    return surface

def load_skeleton(path:Path):
    d=json.loads(path.read_text()); joints=tuple(QualifiedJoint(x['canonical_joint_id'],tuple(map(float,x['position'])),x['parent_canonical_id'],tuple(x.get('support_surface_ids',())),x.get('source_proposal_id','')) for x in d['joints'])
    sk=QualifiedSkeletonIRV2(joints,tuple(d['deform_root_ids']),dict(d['assembly_root_binding']),dict(d['qualification_report']),d['skeleton_lineage_hash'])
    if sk.skeleton_lineage_hash!=EXPECTED_SKELETON_HASH or len(sk.joints)!=22: raise RuntimeError('SKELETON_LINEAGE_OR_COUNT_DRIFT')
    return sk

def load_cache(path:Path):
    if sha(path)!=EXPECTED_CACHE_SHA: raise RuntimeError('CONDITIONING_CACHE_SHA_DRIFT')
    with np.load(path,allow_pickle=False) as z: data={k:np.asarray(z[k]) for k in z.files}
    required=('surface_ids','joint_ids','surface_features','joint_features','teacher_weights','teacher_supervision_mask','rest_points_world')
    if any(k not in data for k in required): raise RuntimeError('CACHE_SCHEMA_MISSING')
    if data['surface_features'].shape!=(950,20) or data['joint_features'].shape!=(22,8) or data['teacher_weights'].shape!=(950,22): raise RuntimeError('CACHE_SHAPE_DRIFT')
    if int(data['teacher_supervision_mask'].sum())!=931: raise RuntimeError('SUPERVISION_MASK_DRIFT')
    return data

def validate_binding(surface,sk,cache):
    sids=tuple(map(str,cache['surface_ids'])); jids=tuple(map(str,cache['joint_ids']))
    if set(sids)!={n.surface_id for n in surface.surface_nodes}: raise RuntimeError('CACHE_SURFACE_ID_SET_DRIFT')
    if set(jids)!={j.canonical_joint_id for j in sk.joints}: raise RuntimeError('CACHE_JOINT_ID_SET_DRIFT')
    if sids!=tuple(sorted(sids)) or jids!=tuple(sorted(jids)): raise RuntimeError('CACHE_IDS_NOT_CANONICAL_SORTED')
    return sids,jids

def probe_transforms(j,device):
    poses=4; t=torch.eye(4,dtype=torch.float32,device=device)[None,None].repeat(1,poses,j,1,1)
    for ji in range(j):
        u=float(ji+1)/float(j)
        t[0,1,ji,0,3]=0.10*u; t[0,1,ji,1,3]=0.035*(-1.0 if ji%2 else 1.0)
        t[0,2,ji,1,3]=0.085*u; t[0,2,ji,2,3]=0.030*(ji-(j-1)/2.0)
        t[0,3,ji,0,3]=-0.055*(ji-(j-1)/2.0); t[0,3,ji,2,3]=0.070*u
    return t

def tensors(cache,device):
    sf=torch.tensor(cache['surface_features'][None],dtype=torch.float32,device=device)
    jf=torch.tensor(cache['joint_features'][None],dtype=torch.float32,device=device)
    w=torch.tensor(cache['teacher_weights'][None],dtype=torch.float32,device=device)
    full_sm=torch.ones((1,950),dtype=torch.bool,device=device)
    teacher_sm=torch.tensor(cache['teacher_supervision_mask'][None].astype(bool),dtype=torch.bool,device=device)
    jm=torch.ones((1,22),dtype=torch.bool,device=device)
    rest=torch.tensor(cache['rest_points_world'][None],dtype=torch.float32,device=device)
    return sf,jf,w,full_sm,teacher_sm,jm,rest

def decode(codec,T):
    sf,jf,w,full_sm,teacher_sm,jm,rest=T
    lat=codec.encode_teacher_weights(sf,jf,w,teacher_sm,jm)
    pred,_=codec.decode_from_latents(lat,sf,jf,full_sm,jm)
    return pred

def train_step(codec,opt,T,transforms):
    sf,jf,w,full_sm,teacher_sm,jm,rest=T
    codec.train(); opt.zero_grad(set_to_none=True)
    lat=codec.encode_teacher_weights(sf,jf,w,teacher_sm,jm)
    pred,_=codec.decode_from_latents(lat,sf,jf,full_sm,jm)
    rec=skin_field_codec_loss_v1(pred,w,teacher_sm,jm)
    deform=codec_deformation_loss_v1(pred,w,rest,transforms,teacher_sm,jm)
    total=rec['total']+deform['deformation_mse']; total.backward(); opt.step()
    return {'total':float(total.detach().cpu()),'reconstruction':float(rec['total'].detach().cpu()),'deformation_mse':float(deform['deformation_mse'].detach().cpu())}

def proposal_from_matrix(mat,sids,jids,surface,sk):
    inf=[]
    a=np.asarray(mat,np.float64)
    for i,sid in enumerate(sids):
        for j,jid in enumerate(jids): inf.append(SkinInfluenceProposal(sid,jid,float(a[i,j])))
    return SkinProposalIR(tuple(inf),surface.geometry_lineage_hash,sk.skeleton_lineage_hash,model_provenance='ARACHNE_MAGE_A0_CODEC_V1')

def qualified_matrix(qskin,sids,jids):
    smap={sid:i for i,sid in enumerate(sids)}; jmap={jid:j for j,jid in enumerate(jids)}; out=np.zeros((len(sids),len(jids)),np.float64)
    for row in qskin.rows:
        i=smap[row.surface_id]
        for jid,w in row.influences: out[i,jmap[jid]]=float(w)
    return out

def deformation_ratio(rest,wtruth,wpred,transforms,mask):
    td=torch_verified_lbs_v1(rest,wtruth,transforms); pd=torch_verified_lbs_v1(rest,wpred,transforms); rp=rest[:,None].expand_as(td)
    m=mask[:,None,:,None].to(td.dtype); denom=(m.sum()*td.shape[1]*td.shape[-1]).clamp_min(1.0)
    motion=torch.sqrt((((td-rp)*m).square().sum()/denom).clamp_min(1e-12)); err=torch.sqrt((((pd-td)*m).square().sum()/denom).clamp_min(1e-12))
    return float((err/motion.clamp_min(1e-6)).cpu()),float(motion.cpu()),float(err.cpu())

def neighbor_telemetry(surface,pred,sids,lowmask):
    idx={s:i for i,s in enumerate(sids)}; vals=[]
    for r in surface.local_relations:
        a=idx[r.a_surface_id]; b=idx[r.b_surface_id]
        if lowmask[a] or lowmask[b]: vals.append(float(np.abs(pred[a]-pred[b]).sum()))
    if not vals:return {'count':0,'mean':0.,'p95':0.,'max':0.}
    x=np.asarray(vals); return {'count':len(vals),'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())}

def metrics(codec,T,transforms,surface,sk,sids,jids,cache):
    codec.eval()
    with torch.no_grad(): pred=decode(codec,T)
    sf,jf,w,full_sm,teacher_sm,jm,rest=T
    raw=pred[0].detach().cpu().numpy().astype(np.float64); truth=w[0].detach().cpu().numpy().astype(np.float64); auth=cache['teacher_supervision_mask'].astype(bool); low=~auth
    row_l1=np.abs(raw-truth).sum(1)
    prop=proposal_from_matrix(raw,sids,jids,surface,sk)
    q=qualify_skin(surface,sk,prop,max_simplex_repair_l1=1e-6,max_total_correction_l1=1e-4,negative_tolerance=1e-8,max_influences=None)
    qw=qualified_matrix(q,sids,jids); qrow=np.abs(qw-truth).sum(1)
    qwt=torch.tensor(qw[None],dtype=torch.float32,device=rest.device)
    rr,rm,re=deformation_ratio(rest,w,pred,transforms,teacher_sm); qr,qm,qe=deformation_ratio(rest,w,qwt,transforms,teacher_sm)
    low_rr,_,_=deformation_ratio(rest,w,pred,transforms,torch.tensor(low[None],dtype=torch.bool,device=rest.device)) if low.any() else (0.,0.,0.)
    report=q.qualification_report
    corrvals=[float(r.correction_l1) for r in q.rows]
    corrmax=max(corrvals,default=0.)
    corrmean=float(np.mean(corrvals)) if corrvals else 0.0
    raw_simplex=float(np.max(np.abs(raw.sum(1)-1.0))); q_simplex=float(np.max(np.abs(qw.sum(1)-1.0)))
    dom=np.argmax(raw[low],axis=1) if low.any() else np.array([],int); dom_hist={jids[int(k)]:int((dom==k).sum()) for k in np.unique(dom)}
    return {
      'raw_row_l1_mean_auth':float(row_l1[auth].mean()),'raw_row_l1_p95_auth':float(np.quantile(row_l1[auth],.95)),
      'qualified_row_l1_mean_auth':float(qrow[auth].mean()),'qualified_row_l1_p95_auth':float(np.quantile(qrow[auth],.95)),
      'raw_deformation_error_ratio_auth':rr,'qualified_deformation_error_ratio_auth':qr,'teacher_motion_rms_auth':rm,
      'raw_deformation_error_rms_auth':re,'qualified_deformation_error_rms_auth':qe,
      'raw_simplex_max_abs_residual':raw_simplex,'qualified_simplex_max_abs_residual':q_simplex,
      'negative_weight_count':int((raw < -1e-8).sum()),'finite':bool(np.isfinite(raw).all()),'qualified_row_count':len(q.rows),
      'compiler_total_correction_l1':float(report['total_correction_l1']),'compiler_mean_row_correction_l1':corrmean,'compiler_max_row_correction_l1':corrmax,
      'compiler_corrected_row_count':int(report['corrected_row_count']),'compiler_total_sparsification_discarded_mass':float(report['total_sparsification_discarded_mass']),
      'low_row_l1_mean_telemetry':float(row_l1[low].mean()),'low_row_l1_p95_telemetry':float(np.quantile(row_l1[low],.95)),
      'low_deformation_error_ratio_telemetry':low_rr,'low_dominant_joint_histogram':dom_hist,'low_neighbor_l1_telemetry':neighbor_telemetry(surface,raw,sids,low),
    }

def is_pass(m):
    return bool(m['finite'] and m['negative_weight_count']==0 and m['qualified_row_count']==950 and
        m['raw_row_l1_p95_auth']<=0.05 and m['qualified_row_l1_p95_auth']<=0.05 and
        m['raw_deformation_error_ratio_auth']<=0.05 and m['qualified_deformation_error_ratio_auth']<=0.05 and
        m['raw_simplex_max_abs_residual']<=1e-6 and m['qualified_simplex_max_abs_residual']<=1e-6 and
        m['compiler_total_correction_l1']<=1e-4 and m['compiler_mean_row_correction_l1']<=1e-7 and m['compiler_max_row_correction_l1']<=1e-6 and
        m['compiler_total_sparsification_discarded_mass']==0.0)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().cpu(),'cuda':[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def set_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if torch.cuda.is_available() and s.get('cuda'): torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])

def save_progress(path,codec,opt,sched,step,streak,trace,elapsed):
    torch.save({'schema':SCHEMA+'.Progress','step':step,'streak':streak,'trace':trace,'elapsed_seconds':elapsed,'model':codec.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng_state(),'codec_config_hash':codec.config.config_hash,'cache_sha256':EXPECTED_CACHE_SHA,'binding_sha256':EXPECTED_BINDING_HASH},path)

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--zero-surface',type=Path,required=True); ap.add_argument('--camera-dir',type=Path,required=True); ap.add_argument('--qualified-skeleton',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--require-cuda',action='store_true'); ap.add_argument('--preflight-only',action='store_true'); args=ap.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA_REQUIRED')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(min(4,os.cpu_count() or 1))
    surface=load_surface(args.zero_surface,args.camera_dir); sk=load_skeleton(args.qualified_skeleton); cache=load_cache(args.cache); sids,jids=validate_binding(surface,sk,cache)
    T=tensors(cache,device); transforms=probe_transforms(len(jids),device)
    codec=SkinFieldCodecV1().to(device)
    if codec.config.config_hash!=EXPECTED_CODEC_HASH: raise RuntimeError('CODEC_CONFIG_HASH_DRIFT')
    opt=torch.optim.AdamW(codec.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_STEPS,eta_min=0.0)
    pre={'schema':SCHEMA+'.Preflight','device':str(device),'surface_hash':surface.geometry_lineage_hash,'skeleton_hash':sk.skeleton_lineage_hash,'cache_sha256':sha(args.cache),'codec_config_hash':codec.config.config_hash,'supervised_rows':int(cache['teacher_supervision_mask'].sum()),'full_rows':950,'joint_count':22}
    if args.preflight_only:
        m0=metrics(codec,T,transforms,surface,sk,sids,jids,cache); loss=train_step(codec,opt,T,transforms); sched.step(); m1=metrics(codec,T,transforms,surface,sk,sids,jids,cache)
        pre.update({'status':'PASS_CPU_EXECUTABLE_PREFLIGHT','step0':m0,'step1_loss':loss,'step1':m1}); write_json(args.output_dir/'ARACHNE_MAGE_A0_PREFLIGHT.json',pre); print('A0_PREFLIGHT='+json.dumps(pre,sort_keys=True,default=float)); return 0
    progress=args.output_dir/'ARACHNE_MAGE_A0_PROGRESS.pt'; result_path=args.output_dir/'ARACHNE_MAGE_A0_RESULT.json'; final_ck=args.output_dir/'ARACHNE_MAGE_A0_CODEC_CHECKPOINT.pt'
    step0=0; streak=0; trace=[]; elapsed_before=0.0
    if progress.exists():
        ck=torch.load(progress,map_location='cpu',weights_only=False)
        if ck.get('codec_config_hash')!=EXPECTED_CODEC_HASH or ck.get('cache_sha256')!=EXPECTED_CACHE_SHA or ck.get('binding_sha256')!=EXPECTED_BINDING_HASH: raise RuntimeError('RESUME_FINGERPRINT_MISMATCH')
        codec.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); step0=int(ck['step']); streak=int(ck['streak']); trace=list(ck['trace']); elapsed_before=float(ck.get('elapsed_seconds',0.)); set_rng(ck['rng'])
        print(f'RESUME step={step0} next={step0+1} streak={streak}')
    start=time.time(); closure=None; last_metrics=None
    for step in range(step0+1,MAX_STEPS+1):
        loss=train_step(codec,opt,T,transforms); sched.step()
        if step==1 or step%CHECK_EVERY==0:
            last_metrics=metrics(codec,T,transforms,surface,sk,sids,jids,cache); ok=is_pass(last_metrics); streak=streak+1 if ok else 0
            rec={'step':step,'pass':ok,'streak':streak,'lr':float(opt.param_groups[0]['lr']),'loss':loss,**last_metrics}; trace.append(rec); print('A0_CHECK='+json.dumps(rec,sort_keys=True,default=float),flush=True)
            if streak>=REQUIRED_STABLE: closure=step
        elapsed=elapsed_before+(time.time()-start)
        if step%CHECKPOINT_EVERY==0 or closure is not None or step==MAX_STEPS: save_progress(progress,codec,opt,sched,step,streak,trace,elapsed)
        if closure is not None: break
    status='A0_TERMINAL_PASS' if closure is not None else 'NO_A0_TERMINAL_CLOSURE'
    final_step=closure if closure is not None else MAX_STEPS
    if last_metrics is None: last_metrics=metrics(codec,T,transforms,surface,sk,sids,jids,cache)
    result={'schema':SCHEMA,'status':status,'seed':SEED,'final_step':final_step,'closure_step':closure,'terminal_streak':streak,'required_terminal_streak':REQUIRED_STABLE,'surface_hash':EXPECTED_SURFACE_HASH,'skeleton_hash':EXPECTED_SKELETON_HASH,'cache_sha256':EXPECTED_CACHE_SHA,'binding_sha256':EXPECTED_BINDING_HASH,'codec_config_hash':EXPECTED_CODEC_HASH,'optimizer':{'type':'AdamW','lr':LR,'weight_decay':WEIGHT_DECAY,'scheduler':'CosineAnnealingLR','t_max':MAX_STEPS},'confidence_policy':{'supervised_rows':931,'low_rows':19,'low_in_encoder':False,'low_in_loss':False},'final_metrics':last_metrics,'trace':trace,'a1_optimizer_authorized':False,'a1_prereg_allowed':status=='A0_TERMINAL_PASS'}
    if closure is not None:
        torch.save({'schema':SCHEMA+'.QualifiedCodec','step':closure,'model':{k:v.detach().cpu() for k,v in codec.state_dict().items()},'codec_config_hash':EXPECTED_CODEC_HASH,'cache_sha256':EXPECTED_CACHE_SHA,'binding_sha256':EXPECTED_BINDING_HASH,'final_metrics':last_metrics},final_ck)
        result['qualified_codec_checkpoint_sha256']=sha(final_ck)
    write_json(result_path,result); print('A0_RESULT='+json.dumps({k:v for k,v in result.items() if k!='trace'},sort_keys=True,default=float)); return 0
if __name__=='__main__': raise SystemExit(main())
