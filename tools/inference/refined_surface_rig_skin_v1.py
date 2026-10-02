"""Fresh checkpoint inference on an explicit surface; never reuse weight rows."""
from __future__ import annotations
import argparse, ast, inspect, json, time
from pathlib import Path

import numpy as np
import torch
from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict, qualified_skeleton_from_dict
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import SkinInfluenceProposal, SkinProposalIR
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import sha256_file
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import GeppettoReferenceStrengthNoLearnedSlotV1
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import GeppettoReferenceStrengthConfigV1
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import tensorize_rigging_surface_v1
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1V4
from models.arachne.v6.readout_v6 import ArachneV6RawReadout


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')

def verified_checkpoint(path,digest):
    if sha256_file(path)!=digest:raise RuntimeError('INFERENCE_CHECKPOINT_SHA_DRIFT')
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def class_ast(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name),include_attributes=False)


def run(args):
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    surface=rigging_surface_from_dict(read(args.surface_json))
    # Same FP32 operations and all surface nodes; SDPA avoids materializing N*N
    # attention matrices in the CPU native MHA fast path. No downsampling/cast.
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(11);np.random.seed(11)
    report={'lane':args.lane,'surface_lineage_hash':surface.geometry_lineage_hash,
            'surface_file_sha256':sha256_file(args.surface_json),'node_count':len(surface.surface_nodes),
            'torch_version':str(torch.__version__),'device':'cpu','dtype':'float32',
            'teacher_inference_inputs_used':False,'model_training_used':False,
            'product_authority_minted':False,'seed':11}
    write(out/(args.lane+'_progress.json'),report)
    started=time.monotonic()
    if args.lane=='rig':
        execution=read(args.fit_run/'artifacts/27_GEPPETTO_FIT/model_fit_execution.json')
        checkpoint=Path(execution['checkpoint_path']);digest=execution['checkpoint_sha256']
        data=verified_checkpoint(checkpoint,digest)
        model=GeppettoReferenceStrengthNoLearnedSlotV1(GeppettoReferenceStrengthConfigV1(**data['config']))
        if model.config.config_hash!=data['config_hash']:raise RuntimeError('GEPPETTO_CONFIG_DRIFT')
        model.load_state_dict(data['model'],strict=True);model.eval();del data
        tensor=tensorize_rigging_surface_v1(surface)
        print('FRESH_RIG_INFERENCE_BEGIN',len(surface.surface_nodes),flush=True)
        with torch.inference_mode():
            proposal=model.propose(tensor,resource_step_limit=min(128,tensor.node_count),generator=torch.Generator(device='cpu').manual_seed(11))
        write(out/'fresh_skeleton_proposal.json',proposal.to_dict())
        skeleton=qualify_skeleton(surface,proposal,run_ilp_shadow=False)
        write(out/'fresh_qualified_skeleton.json',skeleton.to_dict())
        report.update(checkpoint_sha256=digest,skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
                      joint_count=len(skeleton.joints),tensorization_hash=tensor.tensorization_hash)
    else:
        prereg=read(args.fit_run/'artifacts/30_ARACHNE_FIT_PREREGISTERED/model_fit_preregistration.json')
        sealed_source=Path(prereg['model_source_path'])
        if sha256_file(sealed_source)!=prereg['model_source_sha256']:raise RuntimeError('ARACHNE_SOURCE_SHA_DRIFT')
        if class_ast(sealed_source.read_text(),'ArachneV6RawReadout')!=class_ast(inspect.getsource(ArachneV6RawReadout),'ArachneV6RawReadout'):
            raise RuntimeError('ARACHNE_READOUT_CLASS_SEMANTIC_DRIFT')
        result=read(args.arm_dir/'ARACHNE_KNIGHT_V6_RESULT.json')
        baseline=read('canonical/knight_arachne_v6_2x2_rebound_stage35_v1/UNIFORM_ALL_PROJECTED.REPORT.json')
        if result['canonical_weights_sha256']!=baseline['source_weights_npz_sha256']:
            raise RuntimeError('ARACHNE_CORRECTED_ARM_WEIGHT_RECEIPT_DRIFT')
        checkpoint=args.arm_dir/'ARACHNE_KNIGHT_V6_MODEL_FINAL_FP32.pt';digest=result['model_sha256']
        data=verified_checkpoint(checkpoint,digest)
        skeleton=qualified_skeleton_from_dict(read(out/'fresh_qualified_skeleton.json'))
        conditioning=ArachneRichConditioningAdapterV3(require_scene_first=True)([surface],[skeleton])
        model=ArachneA1V4();model.load_state_dict(data['backbone'],strict=True);model.eval()
        ci={key:torch.as_tensor(getattr(conditioning,'view_yaw_code' if key=='view_yaw_fourier' else key))
            for key in inspect.signature(model.forward).parameters}
        print('FRESH_SKIN_BACKBONE_BEGIN',len(surface.surface_nodes),len(skeleton.joints),flush=True)
        with torch.inference_mode():raw=model(**ci)
        geom=torch.as_tensor(conditioning.geometry7)[0].float()
        pair=ci['pair_geometry'][0].float();legal=(ci['pair_mask'].bool() & ci['surface_mask'][:,:,None].bool() & ci['joint_mask'][:,None,:].bool())[0]
        memory=raw.surface_memory[0].float();tokens=raw.field_tokens[0].float()
        decoder=ArachneV6RawReadout(memory.shape[-1],tokens.shape[-1],pair.shape[-1]);decoder.load_state_dict(data['decoder'],strict=True);decoder.eval()
        del model,data
        print('FRESH_SKIN_READOUT_BEGIN',flush=True)
        with torch.inference_mode():_,pred=decoder.decode_all(memory,geom,pair,tokens,legal,chunk=128)
        weights=pred.cpu().numpy().astype(np.float64)
        if not np.isfinite(weights).all() or np.any(weights<0):raise RuntimeError('FRESH_SKIN_NONFINITE_OR_NEGATIVE')
        # FP32 softmax roundoff correction only; recorded, fixed numeric budget.
        normalized=weights/weights.sum(axis=1,keepdims=True)
        corrections=np.abs(normalized-weights).sum(axis=1)
        if corrections.max()>1e-5:raise RuntimeError('FRESH_SKIN_SOFTMAX_NUMERIC_DRIFT')
        ids=conditioning.surface_ids[0];jids=conditioning.joint_ids[0]
        proposal=SkinProposalIR(tuple(SkinInfluenceProposal(sid,jid,float(normalized[i,j])) for i,sid in enumerate(ids) for j,jid in enumerate(jids)),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,model_provenance=digest,
            metadata={'fresh_model_inference':True,'teacher_input_used':False,'architecture_id':result['architecture_id']})
        write(out/'fresh_skin_proposal.json',proposal.to_dict())
        skin=qualify_skin(surface,skeleton,proposal,**prereg['qualification_policy'])
        write(out/'fresh_qualified_skin.json',skin.to_dict())
        np.savez_compressed(out/'fresh_skin_weights.npz',weights=normalized,surface_ids=np.asarray(ids),joint_ids=np.asarray(jids))
        report.update(checkpoint_sha256=digest,arm_result_sha256=sha256_file(args.arm_dir/'ARACHNE_KNIGHT_V6_RESULT.json'),
            skeleton_lineage_hash=skeleton.skeleton_lineage_hash,skin_lineage_hash=skin.skin_lineage_hash,
            joint_count=len(jids),conditioning_hash=conditioning.conditioning_hashes[0],
            softmax_normalization_max_row_l1=float(corrections.max()),softmax_normalization_total_l1=float(corrections.sum()))
    report.update(status='INFERENCE_AND_STRUCTURAL_QUALIFICATION_COMPLETE',seconds=time.monotonic()-started)
    write(out/(args.lane+'_REPORT.json'),report);print('FRESH_INFERENCE='+json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['rig','skin'],required=True)
    ap.add_argument('--surface-json',type=Path,required=True);ap.add_argument('--fit-run',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--arm-dir',type=Path)
    args=ap.parse_args()
    try:run(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        write(args.out_dir/(args.lane+'_ERROR.json'),{'error':type(exc).__name__,'message':str(exc),'inference_complete':False})
        raise
