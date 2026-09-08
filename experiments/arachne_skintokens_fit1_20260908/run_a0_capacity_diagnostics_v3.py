from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from torch import nn

BUNDLE_SHA='e6841a0715e1a65e4502082ae29dfb87d1f12de668330f988fb8f5b35c42d243'
BASELINE_P95=0.11468168089528862
BASELINE_DEFORM=0.04871330037713051
SEED=20260908

class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,layers):
        super().__init__(); blocks=[]; d=in_dim
        for _ in range(layers-1): blocks += [nn.Linear(d,hidden_dim),nn.GELU()]; d=hidden_dim
        blocks.append(nn.Linear(d,out_dim)); self.net=nn.Sequential(*blocks)
    def forward(self,x): return self.net(x)

class FrozenCodec(nn.Module):
    def __init__(self):
        super().__init__(); h=192
        self.surface_embed=MLP(20,h,h,2); self.decoder=MLP(2*h+64,h,1,3)
        self.log_temperature=nn.Parameter(torch.tensor(0.0)); self.temperature_floor=0.35
    def decode(self,lat,sf,je,sm,jm):
        s=self.surface_embed(sf); j=je
        pair=torch.cat([s[:,:,None,:].expand(-1,-1,j.shape[1],-1),j[:,None,:,:].expand(-1,s.shape[1],-1,-1),lat[:,None,:,:].expand(-1,s.shape[1],-1,-1)],dim=-1)
        logits=self.decoder(pair).squeeze(-1)/(F.softplus(self.log_temperature)+self.temperature_floor)
        logits=logits.masked_fill(~jm[:,None,:].bool(),-1e4)
        return torch.softmax(logits,dim=-1)*sm[:,:,None].to(logits.dtype)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_bundle(path):
    got=sha(path)
    if got!=BUNDLE_SHA: raise RuntimeError(f'BUNDLE_SHA_DRIFT:{got}')
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}

def load_state(codec,data):
    state={}
    for k in codec.state_dict().keys():
        nk='log_temperature' if k=='log_temperature' else 'model__'+k.replace('.','__')
        if nk not in data: raise RuntimeError(f'MODEL_STATE_MISSING:{nk}')
        state[k]=torch.from_numpy(np.asarray(data[nk]))
    codec.load_state_dict(state,strict=True)

def probe_transforms(j,device):
    t=torch.eye(4,dtype=torch.float32,device=device)[None,None].repeat(1,4,j,1,1)
    for ji in range(j):
        u=float(ji+1)/float(j)
        t[0,1,ji,0,3]=0.10*u; t[0,1,ji,1,3]=0.035*(-1.0 if ji%2 else 1.0)
        t[0,2,ji,1,3]=0.085*u; t[0,2,ji,2,3]=0.030*(ji-(j-1)/2.0)
        t[0,3,ji,0,3]=-0.055*(ji-(j-1)/2.0); t[0,3,ji,2,3]=0.070*u
    return t

def lbs(rest,w,t):
    hom=torch.cat([rest,torch.ones((*rest.shape[:2],1),dtype=rest.dtype,device=rest.device)],dim=-1)
    tr=torch.einsum('bpjac,bnc->bpjna',t,hom)[...,:3]
    return torch.einsum('bnj,bpjna->bpna',w,tr)

def deformation_mse(pred_w,truth_w,rest,t,sm,jm):
    pair=sm[:,:,None].bool() & jm[:,None,:].bool(); pw=pred_w*pair.to(pred_w.dtype); tw=truth_w*pair.to(truth_w.dtype)
    pw=pw/pw.sum(-1,keepdim=True).clamp_min(1e-8); tw=tw/tw.sum(-1,keepdim=True).clamp_min(1e-8)
    pred=lbs(rest,pw,t); truth=lbs(rest,tw,t); valid=sm[:,None,:,None].to(pred.dtype)
    return (((pred-truth)*valid).square().sum())/(valid.sum().clamp_min(1.0)*pred.shape[1]*pred.shape[-1])

def deformation_ratio(pred_w,truth_w,rest,t,sm):
    pred=lbs(rest,pred_w,t); truth=lbs(rest,truth_w,t); rp=rest[:,None].expand_as(truth); valid=sm[:,None,:,None].to(truth.dtype)
    denom=valid.sum().clamp_min(1.0)*truth.shape[1]*truth.shape[-1]
    motion=torch.sqrt((((truth-rp)*valid).square().sum()/denom).clamp_min(1e-12)); err=torch.sqrt((((pred-truth)*valid).square().sum()/denom).clamp_min(1e-12))
    return float((err/motion.clamp_min(1e-6)).cpu()),float(motion.cpu()),float(err.cpu())

def reconstruction_loss(pred,truth,sm,jm):
    pair=sm[:,:,None].bool() & jm[:,None,:].bool(); t=truth.clamp_min(0.0)*pair.to(truth.dtype); p=pred.clamp_min(1e-8)
    ce=-(t*torch.log(p))*pair.to(p.dtype); active_mass=(t*(t>=1e-3).to(t.dtype)).sum(-1); weighted=ce*(1.0+2.0*active_mass)[...,None]
    ce_loss=weighted.sum()/pair.sum().clamp_min(1).to(weighted.dtype); l1=(torch.abs(pred-truth)*pair.to(pred.dtype)).sum()/pair.sum().clamp_min(1)
    return ce_loss+l1

def metrics(pred,truth,sm,rest,t):
    auth=sm[0].bool(); vals=torch.abs(pred[0]-truth[0]).sum(-1)[auth].detach().cpu().numpy().astype(np.float64)
    ratio,motion,err=deformation_ratio(pred,truth,rest,t,sm)
    return {'row_l1_mean_auth':float(vals.mean()),'row_l1_p95_auth':float(np.quantile(vals,.95)),'rows_gt_0p05':int((vals>.05).sum()),'rows_gt_0p10':int((vals>.10).sum()),'deformation_error_ratio_auth':ratio,'teacher_motion_rms_auth':motion,'deformation_error_rms_auth':err,'simplex_max_abs_residual':float(torch.abs(pred.sum(-1)-1).max().cpu()),'negative_weight_count':int((pred<-1e-8).sum().cpu()),'finite':bool(torch.isfinite(pred).all().item())}

def classify_original(m):
    rel=(BASELINE_P95-m['row_l1_p95_auth'])/BASELINE_P95
    if m['row_l1_p95_auth']<=.05 and m['deformation_error_ratio_auth']<=.05: return 'DECODER_EXPRESSIVE_ENCODER_LIMITING',rel
    return ('MATERIAL_GAIN_BUT_GATE_NOT_MET' if rel>=.25 else 'NO_MATERIAL_GAIN_DECODER_OR_FIELD_LIMIT_REMAINS'),rel

def classify_tail(m):
    rel=(BASELINE_P95-m['row_l1_p95_auth'])/BASELINE_P95
    if m['row_l1_p95_auth']<=.05 and m['deformation_error_ratio_auth']<=.05: return 'DECODER_LATENT_CAPACITY_SUFFICIENT_FOR_GATE',rel
    return ('TAIL_OBJECTIVE_MATERIAL_GAIN_BUT_GATE_NOT_MET' if rel>=.25 else 'DECODER_OR_FIELD_CAPACITY_LIMIT_REMAINS'),rel

def run_opt(name,codec,init_lat,sf,je,truth,full_sm,teacher_sm,jm,rest,t,steps,tail):
    lat=nn.Parameter(init_lat.detach().clone()); opt=torch.optim.Adam([lat],lr=.01,weight_decay=0.0); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=0.0); trace=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True); pred=codec.decode(lat,sf,je,full_sm,jm)
        if tail:
            row=torch.abs(pred-truth).sum(-1)[teacher_sm]; loss=row.mean()+torch.topk(row,k=94,largest=True,sorted=False).values.mean()+deformation_mse(pred,truth,rest,t,teacher_sm,jm)
        else: loss=reconstruction_loss(pred,truth,teacher_sm,jm)+deformation_mse(pred,truth,rest,t,teacher_sm,jm)
        loss.backward(); opt.step(); sch.step()
        if step==1 or step%128==0 or step==steps:
            with torch.no_grad(): mm=metrics(codec.decode(lat,sf,je,full_sm,jm),truth,teacher_sm,rest,t)
            rec={'step':step,'loss':float(loss.detach().cpu()),'lr':float(opt.param_groups[0]['lr']),**mm}; trace.append(rec)
            print(f"{name}_CHECK step={step}/{steps} p95={mm['row_l1_p95_auth']:.9f} mean={mm['row_l1_mean_auth']:.9f} deform={mm['deformation_error_ratio_auth']:.9f} bad05={mm['rows_gt_0p05']} lr={rec['lr']:.8g}",flush=True)
    with torch.no_grad(): final=metrics(codec.decode(lat,sf,je,full_sm,jm),truth,teacher_sm,rest,t)
    return final,trace

def main():
    bundle=Path(os.environ.get('ARACHNE_CAPACITY_BUNDLE','experiments/arachne_skintokens_fit1_20260908/ARACHNE_MAGE_A0_FS1_CAPACITY_BUNDLE_V3.npz')); out=Path(os.environ.get('ARACHNE_CAPACITY_OUT','/tmp/ARACHNE_MAGE_A0_FS1_CAPACITY_DIAGNOSTICS_V1.json')); data=load_bundle(bundle)
    torch.manual_seed(SEED); np.random.seed(SEED); torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('DEVICE='+str(device),flush=True)
    if device.type=='cuda': torch.cuda.manual_seed_all(SEED); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    codec=FrozenCodec(); load_state(codec,data); codec.to(device).eval(); [p.requires_grad_(False) for p in codec.parameters()]
    sf=torch.tensor(data['surface_features'][None],dtype=torch.float32,device=device); je=torch.tensor(data['joint_embedding'][None],dtype=torch.float32,device=device); truth=torch.tensor(data['teacher_weights'][None],dtype=torch.float32,device=device); teacher_sm=torch.tensor(data['teacher_supervision_mask'][None].astype(bool),dtype=torch.bool,device=device); full_sm=torch.ones((1,950),dtype=torch.bool,device=device); jm=torch.ones((1,22),dtype=torch.bool,device=device); rest=torch.tensor(data['rest_points_world'][None],dtype=torch.float32,device=device); init_lat=torch.tensor(data['initial_latents'][None],dtype=torch.float32,device=device); t=probe_transforms(22,device)
    with torch.no_grad(): base=metrics(codec.decode(init_lat,sf,je,full_sm,jm),truth,teacher_sm,rest,t)
    print('BASELINE='+json.dumps(base,sort_keys=True),flush=True)
    if abs(base['row_l1_p95_auth']-BASELINE_P95)>3e-6 or base['rows_gt_0p05']!=165 or abs(base['deformation_error_ratio_auth']-BASELINE_DEFORM)>3e-6: raise RuntimeError('BASELINE_REPLAY_DRIFT')
    if os.environ.get('ARACHNE_CAPACITY_SMOKE')=='1': print('SMOKE_PASS',flush=True); return
    orig,otr=run_opt('ORIGINAL_ORACLE',codec,init_lat,sf,je,truth,full_sm,teacher_sm,jm,rest,t,4096,False); oc,orel=classify_original(orig); print('ORIGINAL_ORACLE_CLASSIFICATION='+oc,flush=True)
    tail,ttr=run_opt('TAIL_CAPACITY',codec,init_lat,sf,je,truth,full_sm,teacher_sm,jm,rest,t,1024,True); tc,trel=classify_tail(tail); print('TAIL_CAPACITY_CLASSIFICATION='+tc,flush=True)
    if tc=='DECODER_LATENT_CAPACITY_SUFFICIENT_FOR_GATE': root='DECODER_64D_CAPACITY_SUFFICIENT__ORIGINAL_A0_FAILURE_PRIMARY_OBJECTIVE_ENCODER_PATH_UNDERFIT'
    elif oc=='DECODER_EXPRESSIVE_ENCODER_LIMITING': root='DECODER_CAPACITY_SUFFICIENT_UNDER_ORIGINAL_OBJECTIVE__TEACHER_ENCODER_LIMITING'
    elif trel>=.25: root='TAIL_OBJECTIVE_MATERIAL_GAIN__DECODER_CAPACITY_NOT_PROVEN_SUFFICIENT'
    else: root='DECODER_OR_FIELD_EXPRESSIVITY_LIMIT_REMAINS'
    report={'schema':'RealSaS.ArachneMageA0FS1CapacityDiagnostics.v1','status':'PASS_DIAGNOSTIC_EXECUTION__NON_AUTHORIZING','bundle_sha256':BUNDLE_SHA,'device':str(device),'codec_parameter_updates':0,'a0_verdict_unchanged':'NO_A0_TERMINAL_CLOSURE','a1_authorized':False,'baseline':base,'original_oracle':{'classification':oc,'relative_p95_reduction':orel,'final':orig,'trace':otr},'tail_capacity':{'classification':tc,'relative_p95_reduction':trel,'final':tail,'trace':ttr},'root_cause_classification':root,'policy':{'threshold_changed':False,'architecture_changed':False,'shipping_claim_allowed':False}}
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print('ROOT_CAUSE_CLASSIFICATION='+root,flush=True); print('RESULT_PATH='+str(out),flush=True)

if __name__=='__main__': main()
