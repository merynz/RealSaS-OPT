"""Compare CPU/CUDA Geppetto inference with archived rig, without training."""
from __future__ import annotations
import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict, qualified_skeleton_from_dict
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload, sha256_file
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import GeppettoReferenceStrengthNoLearnedSlotV1
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import GeppettoReferenceStrengthConfigV1
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import tensorize_rigging_surface_v1
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import read, write, verified_checkpoint


def compare(predicted, archived, scale):
    old = {j.source_proposal_id: j for j in archived.joints}
    new = {j.source_proposal_id: j for j in predicted.joints}
    report = {'same_generation_id_set': set(old) == set(new), 'joint_count': len(new)}
    if set(old) != set(new):
        return report
    old_source = {j.canonical_joint_id: j.source_proposal_id for j in archived.joints}
    new_source = {j.canonical_joint_id: j.source_proposal_id for j in predicted.joints}
    distances = np.asarray([np.linalg.norm(np.asarray(new[p].position) - old[p].position) for p in sorted(old)]) / scale
    report.update(
        normalized_position_rmse=float(np.sqrt(np.square(distances).mean())),
        normalized_position_max=float(distances.max()),
        changed_parent_count=sum(old_source.get(old[p].parent_canonical_id) != new_source.get(new[p].parent_canonical_id) for p in old),
        changed_support_set_count=sum(set(old[p].support_surface_ids) != set(new[p].support_surface_ids) for p in old),
        comparison_scope='SAME_GENERATION_ID__ARCHIVED_PREDICTION_REFERENCE_NOT_TEACHER',
    )
    return report


def main(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_REPLAY_DEVICE_NOT_AVAILABLE')
    ctx = _ctx(args.authority_root, args.run_id)
    surface = rigging_surface_from_dict(stage_output_payload(ctx, '15_RIGGING_SURFACE_QUALIFIED', 'RealSaS.RiggingSurfaceIR.v1'))
    execution = read(args.fit_run / 'artifacts/27_GEPPETTO_FIT/model_fit_execution.json')
    if sha256_file(Path(inspect.getfile(GeppettoReferenceStrengthNoLearnedSlotV1))) != execution['model_source_sha256']:
        raise RuntimeError('GEPPETTO_ENTRY_SOURCE_DRIFT')
    data = verified_checkpoint(Path(execution['checkpoint_path']), execution['checkpoint_sha256'])
    config = GeppettoReferenceStrengthConfigV1(**data['config'])
    if config.config_hash != data['config_hash']:
        raise RuntimeError('GEPPETTO_CONFIG_DRIFT')
    tensor = tensorize_rigging_surface_v1(surface)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictions = {}
    timings = {}
    for device in ('cpu', 'cuda'):
        torch.manual_seed(11)
        model = GeppettoReferenceStrengthNoLearnedSlotV1(config).to(device).eval()
        model.load_state_dict(data['model'], strict=True)
        started = time.monotonic()
        print('GEPPETTO_BACKEND_BEGIN=' + device, flush=True)
        with torch.inference_mode():
            proposal = model.propose(tensor, resource_step_limit=min(128, tensor.node_count),
                                     generator=torch.Generator(device=device).manual_seed(11))
        predicted = qualify_skeleton(surface, proposal, run_ilp_shadow=False)
        write(out / (device + '_proposal.json'), proposal.to_dict())
        write(out / (device + '_qualified_skeleton.json'), predicted.to_dict())
        predictions[device] = predicted
        timings[device] = time.monotonic() - started
        del model
        torch.cuda.empty_cache()
    # Archived output is an evaluation reference, never an inference input.
    archived = qualified_skeleton_from_dict(stage_output_payload(ctx, '28_SKELETON_QUALIFIED', 'RealSaS.QualifiedSkeletonIR.v1'))
    write(out / 'archived_qualified_skeleton.json', archived.to_dict())
    for name in ('result', 'execution_receipt'):
        path = Path(execution[name + '_path'])
        if sha256_file(path) != execution[name + '_sha256']:
            raise RuntimeError('ARCHIVED_GEPPETTO_RECEIPT_DRIFT')
        write(out / ('archived_' + name + '.json'), read(path))
    report = {
        'status': 'BACKEND_REPLAY_MEASURED_ONLY',
        'surface_lineage_hash': surface.geometry_lineage_hash,
        'archived_skeleton_lineage_hash': archived.skeleton_lineage_hash,
        'checkpoint_sha256': execution['checkpoint_sha256'],
        'torch_version': str(torch.__version__), 'cuda_version': str(torch.version.cuda),
        'gpu_name': torch.cuda.get_device_name(0), 'gpu_capability': list(torch.cuda.get_device_capability(0)),
        'dtype': 'float32', 'seed': 11, 'mha_fastpath_enabled': False,
        'normalization_scale': float(tensor.normalization_scale), 'seconds': timings,
        'comparisons': {device: compare(pred, archived, tensor.normalization_scale) for device, pred in predictions.items()},
        'training_used': False, 'teacher_predictor_input_used': False,
        'qualification_scope': 'STRUCTURAL_GRAPH_ONLY', 'product_authority_minted': False,
    }
    write(out / 'REPORT.json', report)
    print('GEPPETTO_BACKEND_RESULT=' + json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('authority-root', 'fit-run', 'out-dir'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / 'ERROR.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
