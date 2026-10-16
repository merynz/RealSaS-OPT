from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,os,random,tempfile,time,uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .contracts_v1 import IRISVNextConfigV1
from .model_v1 import IRISVNextV1
from .training_v1 import IRISVNextOptimizationPolicyV1,build_vnext_optimizer_v1,mechanical_demand_loss_v1

SUBJECT_IDS=("Snake","Bat","T-Rex","Horse","Spider","GreenBlob","Shark","George")

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def _vh(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _source_identity(root:Path):
    paths=("models/iris/vnext/contracts_v1.py","models/iris/vnext/scene_core_v1.py","models/iris/vnext/heads_v1.py","models/iris/vnext/structure_v1.py","models/iris/vnext/model_v1.py","models/iris/vnext/training_v1.py","models/iris/vnext/cohort_v1.py","canonical/VNEXT_SINGLE_CARRIER_CONTRACT_V1.json")
    return {p:_sha(root/p) for p in paths}

@dataclass(frozen=True)
class VNextStageV1:
    stage_id:str; steps:int; scene_lr_scale:float=1.; evidence_lr_scale:float=1.; structure_lr_scale:float=0.; field_weight:float=1.; mechanical_weight:float=0.; minimum_lr_ratio:float=.1
    def validate(self):
        if self.steps<=0 or min(self.scene_lr_scale,self.evidence_lr_scale,self.structure_lr_scale,self.field_weight,self.mechanical_weight,self.minimum_lr_ratio)<0 or self.minimum_lr_ratio>1:raise ValueError("VNEXT_STAGE_POLICY_INVALID")
        if self.field_weight+self.mechanical_weight<=0:raise ValueError("VNEXT_STAGE_HAS_NO_OBJECTIVE")

@dataclass(frozen=True)
class VNextFit8PolicyV1:
    seed:int=8162026; geometry_samples_per_subject:int=4096; mechanical_samples_per_subject:int=2048; weight_decay:float=.01; scene_learning_rate:float=3e-5; evidence_learning_rate:float=1e-4; structure_learning_rate:float=5e-5; checkpoint_interval:int=100; eval_interval:int=100; expected_subjects:tuple[str,...]=SUBJECT_IDS; stages:tuple[VNextStageV1,...]=(VNextStageV1("G1_GEOMETRY",2400,field_weight=1.,mechanical_weight=0.),VNextStageV1("G2_MECHANICAL_EVIDENCE",2400,scene_lr_scale=.5,evidence_lr_scale=1.,field_weight=.5,mechanical_weight=1.))
    def validate(self):
        if tuple(self.expected_subjects)!=SUBJECT_IDS or self.geometry_samples_per_subject<=0 or self.mechanical_samples_per_subject<=0 or self.checkpoint_interval<=0 or self.eval_interval<=0:raise ValueError("VNEXT_FIT8_POLICY_INVALID")
        for s in self.stages:s.validate()
    @property
    def total_steps(self):return sum(s.steps for s in self.stages)

_REQUIRED_SOURCE=("image_tokens","token_xy","camera_features")
_REQUIRED_GEOMETRY=("geometry_points","geometry_targets")
_REQUIRED_MECH=("mechanical_points","mechanical_normals","mechanical_stretch_major","mechanical_stretch_minor","mechanical_tangent_direction_2theta","mechanical_bend_demand","mechanical_rigidity_probability")

class VNextFit8CohortV1:
    """Eight fixed shards resident as one batched cohort; no model-forward subject loop."""
    def __init__(self,data_root,manifest,device):
        self.root=Path(data_root).resolve();self.manifest=dict(manifest);self.device=torch.device(device);rows=tuple(self.manifest.get("subjects") or ())
        ids=tuple(str(r["subject_id"]) for r in rows)
        if ids!=SUBJECT_IDS:raise ValueError("VNEXT_FIT8_SUBJECT_ORDER_OR_ID_DRIFT")
        if self.manifest.get("teacher_inputs_in_inference_forward") is not False:raise ValueError("VNEXT_FIT8_TEACHER_FORWARD_FORBIDDEN")
        shards=[]
        for row in rows:
            path=(self.root/str(row["shard_file"])).resolve()
            if not path.is_relative_to(self.root) or _sha(path)!=str(row["shard_sha256"]):raise ValueError("VNEXT_FIT8_SHARD_HASH_OR_PATH_DRIFT")
            x=torch.load(path,map_location="cpu",weights_only=True)
            if str(x.get("subject_id"))!=str(row["subject_id"]):raise ValueError("VNEXT_FIT8_SHARD_SUBJECT_DRIFT")
            if x.get("teacher_inputs_in_inference_forward") is not False:raise ValueError("VNEXT_FIT8_SHARD_TEACHER_FORWARD_FORBIDDEN")
            for key in _REQUIRED_SOURCE+_REQUIRED_GEOMETRY+_REQUIRED_MECH:
                if key not in x or not isinstance(x[key],torch.Tensor):raise ValueError("VNEXT_FIT8_SHARD_FIELD_MISSING:"+key)
            shards.append(x)
        def stack(key):
            values=[]
            for x in shards:
                v=x[key]
                if key in _REQUIRED_SOURCE and v.ndim>0 and v.shape[0]==1:v=v.squeeze(0)
                values.append(v)
            shapes={tuple(v.shape) for v in values}
            if len(shapes)!=1:raise ValueError("VNEXT_FIT8_BATCH_SHAPE_DRIFT:"+key)
            out=torch.stack(values,dim=0).contiguous()
            if self.device.type=="cuda":out=out.pin_memory().to(self.device,non_blocking=True)
            else:out=out.to(self.device)
            return out
        self.tensors={k:stack(k) for k in _REQUIRED_SOURCE+_REQUIRED_GEOMETRY+_REQUIRED_MECH}
        if self.tensors["image_tokens"].shape[0]!=8:raise AssertionError("VNEXT_FIT8_NOT_BATCH8")
        if self.tensors["geometry_points"].shape[:1]!=(8,) or self.tensors["mechanical_points"].shape[:1]!=(8,):raise AssertionError("VNEXT_FIT8_BANK_BATCH_DRIFT")
        if self.tensors["geometry_targets"].shape!=self.tensors["geometry_points"].shape[:2]:raise ValueError("VNEXT_FIT8_GEOMETRY_TARGET_SHAPE")
        mshape=self.tensors["mechanical_points"].shape[:2]
        if self.tensors["mechanical_normals"].shape!=self.tensors["mechanical_points"].shape:raise ValueError("VNEXT_FIT8_MECHANICAL_NORMAL_SHAPE")
        for k in ("mechanical_stretch_major","mechanical_stretch_minor","mechanical_bend_demand","mechanical_rigidity_probability"):
            if self.tensors[k].shape!=mshape:raise ValueError("VNEXT_FIT8_MECHANICAL_TARGET_SHAPE:"+k)
        if self.tensors["mechanical_tangent_direction_2theta"].shape!=mshape+(2,):raise ValueError("VNEXT_FIT8_MECHANICAL_DIRECTION_SHAPE")
    @staticmethod
    def _indices(bank_count,sample_count,seed,step,device):
        if bank_count<sample_count:raise ValueError("VNEXT_FIT8_BANK_TOO_SMALL")
        g=torch.Generator(device="cpu").manual_seed(int(seed)+int(step)*1000003)
        idx=torch.stack([torch.randperm(bank_count,generator=g)[:sample_count] for _ in range(8)],dim=0)
        return idx.to(device)
    @staticmethod
    def _gather(bank,idx):
        tail=bank.shape[2:];view=idx.reshape(idx.shape+(1,)*len(tail)).expand(idx.shape+tail);return torch.gather(bank,1,view)
    def geometry_batch(self,policy,step):
        idx=self._indices(self.tensors["geometry_points"].shape[1],policy.geometry_samples_per_subject,policy.seed+11,step,self.device)
        return self._gather(self.tensors["geometry_points"],idx),self._gather(self.tensors["geometry_targets"],idx)
    def mechanical_batch(self,policy,step):
        idx=self._indices(self.tensors["mechanical_points"].shape[1],policy.mechanical_samples_per_subject,policy.seed+23,step,self.device)
        keys=_REQUIRED_MECH
        return {k:self._gather(self.tensors[k],idx) for k in keys}

class VNextCheckpointStoreV1:
    def __init__(self,root,binding_hash):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.binding_hash=binding_hash
    def save(self,payload):
        step=int(payload["step"]);name=f"step_{step:08d}_{uuid.uuid4().hex}.pt";tmp=self.root/(name+".staged")
        torch.save(payload,tmp);digest=_sha(tmp);torch.load(tmp,map_location="cpu",weights_only=True);target=self.root/name;os.replace(tmp,target);pointer={"schema":"RealSaS.IRISVNextFit8CheckpointPointer.v1","file":name,"sha256":digest,"step":step,"binding_sha256":self.binding_hash};ptmp=self.root/"LATEST.json.staged";ptmp.write_text(json.dumps(pointer,sort_keys=True,indent=2));os.replace(ptmp,self.root/"LATEST.json");return pointer
    def load(self):
        p=self.root/"LATEST.json"
        if not p.exists():return None
        row=json.loads(p.read_text())
        if row["binding_sha256"]!=self.binding_hash:raise ValueError("VNEXT_RESUME_BINDING_DRIFT")
        path=self.root/Path(row["file"]).name
        if _sha(path)!=row["sha256"]:raise ValueError("VNEXT_RESUME_PAYLOAD_HASH_DRIFT")
        value=torch.load(path,map_location="cpu",weights_only=True)
        if value["binding_sha256"]!=self.binding_hash or int(value["step"])!=int(row["step"]):raise ValueError("VNEXT_RESUME_PAYLOAD_BINDING_DRIFT")
        return value

class IRISVNextFit8TrainerV1:
    """Real B=8 scene encode/query with one backward and one optimizer update."""
    def __init__(self,*,data_root,manifest,checkpoint_root,device,model_config=IRISVNextConfigV1(),policy=VNextFit8PolicyV1(),repo_root=None):
        policy.validate();model_config.validate();self.device=torch.device(device);self.policy=policy;self.cohort=VNextFit8CohortV1(data_root,manifest,self.device);self.repo_root=Path(repo_root or Path(__file__).resolve().parents[3]);self.binding={"schema":"RealSaS.IRISVNextFit8Binding.v1","source":_source_identity(self.repo_root),"manifest_sha256":_vh(manifest),"policy":asdict(policy),"model_config":asdict(model_config),"optimizer_semantics":"ONE_BATCH8_FORWARD__ONE_BACKWARD__ONE_STEP","teacher_inputs_in_inference_forward":False};self.binding_hash=_vh(self.binding)
        random.seed(policy.seed);np.random.seed(policy.seed);torch.manual_seed(policy.seed)
        if self.device.type=="cuda":torch.cuda.manual_seed_all(policy.seed);torch.set_float32_matmul_precision("high");torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
        self.model=IRISVNextV1(model_config).to(self.device);op=IRISVNextOptimizationPolicyV1(scene_learning_rate=policy.scene_learning_rate,evidence_learning_rate=policy.evidence_learning_rate,structure_learning_rate=policy.structure_learning_rate,weight_decay=policy.weight_decay,expected_cohort_size=8);self.optimizer=build_vnext_optimizer_v1(self.model,op);self.store=VNextCheckpointStoreV1(checkpoint_root,self.binding_hash);self.step=0;self.stage_index=0;self.stage_step=0;self.started=time.monotonic();self._apply_stage_lrs();prev=self.store.load()
        if prev:self._restore(prev)
        else:self.save()
    @property
    def stage(self):return self.policy.stages[self.stage_index]
    def _apply_stage_lrs(self):
        scales={"scene_core":self.stage.scene_lr_scale,"evidence_heads":self.stage.evidence_lr_scale,"structure":self.stage.structure_lr_scale};base={"scene_core":self.policy.scene_learning_rate,"evidence_heads":self.policy.evidence_learning_rate,"structure":self.policy.structure_learning_rate}
        for g in self.optimizer.param_groups:g["lr"]=base[g["name"]]*scales[g["name"]]
    def _payload(self):return {"schema":"RealSaS.IRISVNextFit8Checkpoint.v1","binding":self.binding,"binding_sha256":self.binding_hash,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"step":self.step,"stage_index":self.stage_index,"stage_step":self.stage_step,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    def save(self):return self.store.save(self._payload())
    def _restore(self,p):
        self.step=int(p["step"]);self.stage_index=int(p["stage_index"]);self.stage_step=int(p["stage_step"]);self.model.load_state_dict(p["model"]);self.optimizer.load_state_dict(p["optimizer"]);torch.set_rng_state(p["torch_rng"])
        if p["cuda_rng"]:torch.cuda.set_rng_state_all(p["cuda_rng"])
        self._apply_stage_lrs()
    def _advance_stage_if_needed(self):
        if self.stage_step<self.stage.steps:return False
        if self.stage_index+1>=len(self.policy.stages):return True
        self.stage_index+=1;self.stage_step=0;self._apply_stage_lrs();return False
    def one_step(self):
        if self._advance_stage_if_needed():raise StopIteration("VNEXT_FIT8_COMPLETE")
        self.model.train();self.optimizer.zero_grad(set_to_none=True);gpoints,gtarget=self.cohort.geometry_batch(self.policy,self.step+1);mb=self.cohort.mechanical_batch(self.policy,self.step+1);source=self.cohort.tensors
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            planes=self.model.encode_scene(source["image_tokens"],source["token_xy"],source["camera_features"]);gpred=self.model.query_evidence(planes,gpoints);field=F.smooth_l1_loss(gpred["field"].float(),gtarget.float());mechanical=field*0.
            if self.stage.mechanical_weight>0:
                mpred=self.model.query_evidence(planes,mb["mechanical_points"])["mechanical_demand"];truth={k:mb[k] for k in ("mechanical_stretch_major","mechanical_stretch_minor","mechanical_tangent_direction_2theta","mechanical_bend_demand","mechanical_rigidity_probability")};mechanical=mechanical_demand_loss_v1(mpred,{"stretch_major":truth["mechanical_stretch_major"],"stretch_minor":truth["mechanical_stretch_minor"],"tangent_direction_2theta":truth["mechanical_tangent_direction_2theta"],"bend_demand":truth["mechanical_bend_demand"],"rigidity_probability":truth["mechanical_rigidity_probability"]});loss=self.stage.field_weight*field+self.stage.mechanical_weight*mechanical
        if not bool(torch.isfinite(loss)):raise RuntimeError("VNEXT_FIT8_NONFINITE_LOSS")
        loss.backward();self.optimizer.step();self.step+=1;self.stage_step+=1
        if self.step%self.policy.checkpoint_interval==0:self.save()
        return {"global_step":self.step,"stage":self.stage.stage_id,"stage_step":self.stage_step,"loss":float(loss.detach()),"field_loss":float(field.detach()),"mechanical_loss":float(mechanical.detach()),"batch_subjects":8,"optimizer_steps":1,"elapsed_seconds":time.monotonic()-self.started}
