from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.skin_field_codec.v1.skin_field_codec_v1 import SkinFieldCodecV1, skin_field_codec_loss_v1

SEED=20260908; STEPS=4096; CHECK_EVERY=128; LR=.01; BASE=0.11468168089528862
PROGRESS_SHA='233b2c003f523060a8cedc6999c13259764238cd60d46f2d347c806ae070de34'
CACHE_SHA='db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd'

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def probe_transforms(j,device):
    poses=4; t=torch.eye(4,dtype=torch.float32,device=device)[None,None].repeat(1,poses,j,1,1)
    for ji in range(j):
        u=float(ji+1)/float(j)
        t[0,1,ji,0,3]=0.10*u; t[0,1,ji,1,3]=0.035*(-1.0 if ji%2 else 1.0)
        t[0,2,ji,1,3]=0.085*u; t[0,2,ji,2,3]=0.030*(ji-(j-1)/2.0)
        t[0,3,ji,0,3]=-0.055*(ji-(j-1)/2.0); t[0,3,ji,2,3]=0.070*u
    return t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--progress',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    if sha(args.progress)!=PROGRESS_SHA: raise RuntimeError('PROGRESS_SHA_DRIFT')
    if sha(args.cache)!=CACHE_SHA: raise RuntimeError('CACHE_SHA_DRIFT')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
    ck=torch.load(args.progress,map_location='cpu',weights_only=False)
    if int(ck['step'])!=1536: raise RuntimeError('CHECKPOINT_STEP_DRIFT')
    with np.load(args.cache,allow_pickle=False) as z: c={k:np.asarray(z[k]) for k in z.files}
    codec=SkinFieldCodecV1(); codec.load_state_dict(ck['model']); codec.eval(); [p.requires_grad_(False) for p in codec.parameters()]
    if codec.config.config_hash!='24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715': raise RuntimeError('CODEC_CONFIG_DRIFT')
    auth=c['teacher_supervision_mask'].astype(bool); ix=np.where(auth)[0]
    if len(ix)!=934: raise RuntimeError('SUPERVISION_DRIFT')
    sf0=torch.tensor(c['surface_features'][None],dtype=torch.float32); jf=torch.tensor(c['joint_features'][None],dtype=torch.float32); wf=torch.tensor(c['teacher_weights'][None],dtype=torch.float32); tsm=torch.tensor(auth[None]); jm=torch.ones((1,22),dtype=torch.bool)
    with torch.no_grad():
        init=codec.encode_teacher_weights(sf0,jf,wf,tsm,jm).detach().clone(); sf=sf0[:,ix]; s=codec.surface_embed(sf); j=codec.joint_embed(jf)
        std,_=codec.decode_from_latents(init,sf,jf,torch.ones((1,len(ix)),dtype=torch.bool),jm)
    lat=nn.Parameter(init.clone()); opt=torch.optim.Adam([lat],lr=LR,weight_decay=0.0); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS,eta_min=0.0)
    w=wf[:,ix]; sm=torch.ones((1,len(ix)),dtype=torch.bool); rest=torch.tensor(c['rest_points_world'][None,ix],dtype=torch.float32); trans=probe_transforms(22,torch.device('cpu'))
    ones=torch.ones((*rest.shape[:2],1)); hom=torch.cat([rest,ones],-1); transformed=torch.einsum('bpjac,bnc->bpjna',trans,hom)[...,:3]; truth_def=torch.einsum('bnj,bpjna->bpna',w,transformed).detach()
    L1,L2,L3=codec.decoder.net[0],codec.decoder.net[2],codec.decoder.net[4]; W1=L1.weight; b1=L1.bias
    with torch.no_grad(): sproj=F.linear(s,W1[:,:192],None); jproj=F.linear(j,W1[:,192:384],None)
    def dec():
        lp=F.linear(lat,W1[:,384:],b1); h=sproj[:,:,None,:]+jproj[:,None,:,:]+lp[:,None,:,:]; h=F.gelu(h); h=F.gelu(L2(h)); logits=L3(h).squeeze(-1); temp=F.softplus(codec.log_temperature)+codec.config.temperature_floor; return torch.softmax(logits/temp,-1)
    with torch.no_grad(): transport_max_diff=float((dec()-std).abs().max())
    if transport_max_diff>2e-6: raise RuntimeError('OPTIMIZED_DECODER_NUMERIC_DRIFT')
    def dmse(p): return (torch.einsum('bnj,bpjna->bpna',p,transformed)-truth_def).square().mean()
    def meas():
        with torch.no_grad():
            p=dec(); row=np.abs(p[0].numpy().astype(float)-w[0].numpy().astype(float)).sum(1); pd=torch.einsum('bnj,bpjna->bpna',p,transformed); mse=(pd-truth_def).square().mean(); rms=torch.sqrt(mse+1e-12); motion=torch.sqrt(((truth_def-rest[:,None].expand_as(truth_def)).square().mean()).clamp_min(1e-12)); ratio=rms/motion.clamp_min(1e-6)
            return {'row_l1_mean_auth':float(row.mean()),'row_l1_p95_auth':float(np.quantile(row,.95)),'row_l1_max_auth':float(row.max()),'rows_gt_0p05':int((row>.05).sum()),'deformation_error_ratio_auth':float(ratio),'latent_l2':float(torch.linalg.vector_norm(lat)),'latent_delta_l2':float(torch.linalg.vector_norm(lat-init))}
    initial=meas(); trace=[]; print('ORACLE_INITIAL='+json.dumps(initial,sort_keys=True),flush=True)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True); p=dec(); rec=skin_field_codec_loss_v1(p,w,sm,jm); dm=dmse(p); total=rec['total']+dm; total.backward(); opt.step(); sched.step()
        if step==1 or step%CHECK_EVERY==0:
            mm=meas(); rr={'step':step,'lr':float(opt.param_groups[0]['lr']),'loss_total':float(total.detach()),**mm}; trace.append(rr); print('ORACLE_CHECK='+json.dumps(rr,sort_keys=True),flush=True)
    final=meas(); red=(BASE-final['row_l1_p95_auth'])/BASE
    if final['row_l1_p95_auth']<=.05 and final['deformation_error_ratio_auth']<=.05: cls='DECODER_EXPRESSIVE_ENCODER_LIMITING'
    elif red>=.25: cls='MATERIAL_GAIN_BUT_GATE_NOT_MET'
    else: cls='NO_MATERIAL_GAIN_DECODER_OR_FIELD_LIMIT_REMAINS'
    r={'schema':'RealSaS.ArachneMageA0FS1PostfailOracleLatentDiagnostic.v1','status':'PASS_DIAGNOSTIC_COMPLETED__NON_AUTHORIZING','prereg_commit':'57218de673e9c7bec85424e6ea987ae4fa33b000','a0_result_sha256':'a2b3a7f1d4a4b448a9e236d28efc9a92577d12de48b18e50c767448d6038f64f','a0_progress_checkpoint_sha256':PROGRESS_SHA,'intervention':{'optimized_object':'free_latent_[1,22,64]','codec_parameter_updates':0,'steps':4096,'optimizer':'Adam','lr':LR,'scheduler':'CosineAnnealingLR','device':'cpu','optimized_decoder_transport_max_abs_diff_at_init':transport_max_diff},'initial':initial,'final':final,'relative_p95_reduction':float(red),'classification':cls,'trace':trace,'policy':{'does_not_reopen_a0':True,'does_not_authorize_a1':True,'candidate_checkpoint_emitted':False}}
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('ORACLE_RESULT='+json.dumps({'classification':cls,'final_p95':final['row_l1_p95_auth'],'reduction':red,'final_deform':final['deformation_error_ratio_auth'],'rows_gt_0p05':final['rows_gt_0p05']},sort_keys=True),flush=True)
    print('ORACLE_RESULT_SHA='+sha(args.out),flush=True)

if __name__=='__main__': main()
