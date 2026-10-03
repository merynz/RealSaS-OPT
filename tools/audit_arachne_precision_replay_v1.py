"""Measure CPU precision sensitivity on the original surface and fixed rig.

Archived skin is evaluation-only. No fitting, qualification, or product promotion.
CPU BF16 is a diagnostic control, not a claim of CUDA BF16 replay equivalence.
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file, stage_output_payload,
)
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1V4
from models.arachne.v6.readout_v6 import ArachneV6RawReadout
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import class_ast, read, verified_checkpoint, write
from tools.inference.query_chunked_sdpa_v1 import query_chunked_cuda_bf16_sdpa


def delta(a, b):
    row_l1 = np.abs(a - b).sum(axis=1)
    return {
        "row_l1_mean": float(row_l1.mean()),
        "row_l1_p95": float(np.quantile(row_l1, .95)),
        "row_l1_max": float(row_l1.max()),
        "changed_dominant_joint_count": int(np.count_nonzero(a.argmax(1) != b.argmax(1))),
    }


def main(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(args.authority_root, args.run_id)
    surface = rigging_surface_from_dict(stage_output_payload(
        ctx, '15_RIGGING_SURFACE_QUALIFIED', 'RealSaS.RiggingSurfaceIR.v1'))
    skeleton = qualified_skeleton_from_dict(stage_output_payload(
        ctx, '28_SKELETON_QUALIFIED', 'RealSaS.QualifiedSkeletonIR.v1'))
    prereg = read(args.fit_run / 'artifacts/30_ARACHNE_FIT_PREREGISTERED/model_fit_preregistration.json')
    source = Path(prereg['model_source_path'])
    if sha256_file(source) != prereg['model_source_sha256']:
        raise RuntimeError('ARACHNE_SOURCE_SHA_DRIFT')
    if class_ast(source.read_text(), 'ArachneV6RawReadout') != class_ast(
            inspect.getsource(ArachneV6RawReadout), 'ArachneV6RawReadout'):
        raise RuntimeError('ARACHNE_READOUT_CLASS_SEMANTIC_DRIFT')
    result = read(args.arm_dir / 'ARACHNE_KNIGHT_V6_RESULT.json')
    baseline = read('canonical/knight_arachne_v6_2x2_rebound_stage35_v1/UNIFORM_ALL_PROJECTED.REPORT.json')
    if result['canonical_weights_sha256'] != baseline['source_weights_npz_sha256']:
        raise RuntimeError('ARACHNE_CORRECTED_ARM_WEIGHT_RECEIPT_DRIFT')
    checkpoint = args.arm_dir / 'ARACHNE_KNIGHT_V6_MODEL_FINAL_FP32.pt'
    data = verified_checkpoint(checkpoint, result['model_sha256'])

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(11)
    np.random.seed(11)
    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)([surface], [skeleton])
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA_DEVICE_NOT_AVAILABLE')
    model = ArachneA1V4().to(device).eval()
    model.load_state_dict(data['backbone'], strict=True)
    ci = {key: torch.as_tensor(getattr(conditioning, 'view_yaw_code' if key == 'view_yaw_fourier' else key), device=device)
          for key in inspect.signature(model.forward).parameters}
    geom = torch.as_tensor(conditioning.geometry7, device=device)[0].float()
    pair = ci['pair_geometry'][0].float()
    legal = (ci['pair_mask'].bool() & ci['surface_mask'][:, :, None].bool()
             & ci['joint_mask'][:, None, :].bool())[0]
    predictions = {}
    measurements = {}
    started = time.monotonic()
    for backbone_precision in ('fp32', 'bf16'):
        print('PRECISION_BACKBONE_BEGIN=' + backbone_precision, flush=True)
        attention_context = (query_chunked_cuda_bf16_sdpa(args.cuda_query_chunk)
                             if device == 'cuda' and args.cuda_query_chunk else nullcontext(None))
        with attention_context as telemetry, torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                                                                                    enabled=backbone_precision == 'bf16'):
            raw = model(**ci)
        memory = raw.surface_memory[0].float()
        tokens = raw.field_tokens[0].float()
        decoder = ArachneV6RawReadout(memory.shape[-1], tokens.shape[-1], pair.shape[-1]).to(device).eval()
        decoder.load_state_dict(data['decoder'], strict=True)
        for readout_precision in ('fp32', 'bf16'):
            arm = backbone_precision + '_backbone__' + readout_precision + '_readout'
            print('PRECISION_READOUT_BEGIN=' + arm, flush=True)
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                                                       enabled=readout_precision == 'bf16'):
                _, pred = decoder.decode_all(memory, geom, pair, tokens, legal, chunk=128)
            weights = pred.float().cpu().numpy().astype(np.float64)
            if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(1) <= 0):
                raise RuntimeError('INVALID_PRECISION_PROBE_WEIGHTS')
            normalized = weights / weights.sum(1, keepdims=True)
            correction = np.abs(normalized - weights).sum(1)
            predictions[arm] = normalized
            measurements[arm] = {
                'normalization_max_row_l1': float(correction.max()),
                'normalization_total_l1': float(correction.sum()),
                'normalization_is_diagnostic_only': True,
                'attention_query_chunk_telemetry': telemetry,
            }
            np.savez_compressed(out / (arm + '.npz'), raw_weights=weights, weights=normalized,
                                surface_ids=np.asarray(conditioning.surface_ids[0]),
                                joint_ids=np.asarray(conditioning.joint_ids[0]))
            write(out / 'PROGRESS.json', {'completed_arms': list(predictions), 'measurements': measurements})
        del raw, decoder, memory, tokens

    # This read occurs only after all model predictions. Archived weights never
    # become model inputs, and cannot influence parameter selection or execution.
    archived = stage_output_payload(ctx, '32_SKIN_QUALIFIED', 'RealSaS.QualifiedSkinIR.v1')
    if archived['surface_binding_hash'] != surface.geometry_lineage_hash or archived['skeleton_binding_hash'] != skeleton.skeleton_lineage_hash:
        raise RuntimeError('ARCHIVED_SKIN_BINDING_MISMATCH')
    rows = {r['surface_id']: dict(r['influences']) for r in archived['rows']}
    ids, jids = conditioning.surface_ids[0], conditioning.joint_ids[0]
    if set(rows) != set(ids) or any(set(row) - set(jids) for row in rows.values()):
        raise RuntimeError('ARCHIVED_SKIN_ID_MISMATCH')
    target = np.asarray([[rows[s].get(j, 0.) for j in jids] for s in ids], dtype=np.float64)
    for arm, pred in predictions.items():
        measurements[arm]['vs_archived'] = delta(pred, target)
        measurements[arm]['vs_' + device + '_fp32'] = delta(pred, predictions['fp32_backbone__fp32_readout'])
    report = {
        'status': 'PRECISION_SENSITIVITY_MEASURED_ONLY',
        'surface_lineage_hash': surface.geometry_lineage_hash,
        'skeleton_lineage_hash': skeleton.skeleton_lineage_hash,
        'archived_skin_lineage_hash': archived['skin_lineage_hash'],
        'checkpoint_sha256': result['model_sha256'],
        'arm_result_sha256': sha256_file(args.arm_dir / 'ARACHNE_KNIGHT_V6_RESULT.json'),
        'conditioning_hash': conditioning.conditioning_hashes[0],
        'node_count': len(ids), 'joint_count': len(jids),
        'device': device, 'torch_version': str(torch.__version__), 'seed': 11,
        'cuda_query_chunk': args.cuda_query_chunk,
        'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'readout_chunk': 128, 'measurements': measurements,
        'seconds': time.monotonic() - started,
        'training_used': False, 'teacher_predictor_input_used': False,
        'qualification_executed': False, 'product_authority_minted': False,
        'cuda_bf16_equivalence_claimed': False,
    }
    write(out / 'REPORT.json', report)
    print('PRECISION_REPLAY_RESULT=' + json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('authority-root', 'fit-run', 'arm-dir', 'out-dir'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--cuda-query-chunk', type=int, default=0)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / 'ERROR.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
