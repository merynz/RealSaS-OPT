from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

import run_arachne_mage_a0_historical_1cc449 as hist

SCHEMA = 'RealSaS.ArachneMageA0FS1Run.v2'
EXPECTED_HISTORICAL_RUNNER_GIT_BLOB = 'f32d2f75e6b9e6095e1d5455ed9227b32553a9d5'
EXPECTED_CACHE_SHA = 'db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd'
EXPECTED_CACHE_MANIFEST_SHA = '5b3e63d60fce76226e16aac5d754a5f9cde114ccd8327a3242fad8a714af7d6b'
EXPECTED_CACHE_BINDING_SHA = 'c7e3bf10fc8edf16f862b4ce58b3aabfeb2aabf14cc8e744e53e3afc0764ac9f'
EXPECTED_TARGET_BINDING_SHA = 'ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf'
EXPECTED_TEACHER_WEIGHTS_SHA = '7a09f276efc41f0febc7037900c2e954f7094cb5ae5e6bad70cb04f4507b586d'
EXPECTED_SUPERVISED_ROWS = 934
EXPECTED_LOW_ROWS = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode('ascii') + data).hexdigest()


def raw_array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + '\n', encoding='utf-8')


def patch_historical_runner(cache_manifest: Path) -> None:
    if git_blob_sha(Path(hist.__file__).resolve()) != EXPECTED_HISTORICAL_RUNNER_GIT_BLOB:
        raise RuntimeError('HISTORICAL_RUNNER_GIT_BLOB_DRIFT')
    if sha(cache_manifest) != EXPECTED_CACHE_MANIFEST_SHA:
        raise RuntimeError('CACHE_MANIFEST_SHA_DRIFT')
    manifest = json.loads(cache_manifest.read_text(encoding='utf-8'))
    if manifest.get('cache', {}).get('npz_sha256') != EXPECTED_CACHE_SHA:
        raise RuntimeError('CACHE_MANIFEST_NPZ_DRIFT')
    if manifest.get('cache', {}).get('binding_sha256') != EXPECTED_CACHE_BINDING_SHA:
        raise RuntimeError('CACHE_BINDING_DRIFT')
    if manifest.get('inputs', {}).get('fs1_target_binding_sha256') != EXPECTED_TARGET_BINDING_SHA:
        raise RuntimeError('TARGET_BINDING_DRIFT')
    if manifest.get('inputs', {}).get('teacher_weights_content_sha256') != EXPECTED_TEACHER_WEIGHTS_SHA:
        raise RuntimeError('TEACHER_WEIGHTS_MANIFEST_DRIFT')
    if manifest.get('cache', {}).get('supervised_rows') != EXPECTED_SUPERVISED_ROWS or manifest.get('cache', {}).get('low_rows') != EXPECTED_LOW_ROWS:
        raise RuntimeError('CACHE_SUPERVISION_MANIFEST_DRIFT')

    hist.SCHEMA = SCHEMA
    hist.EXPECTED_CACHE_SHA = EXPECTED_CACHE_SHA
    hist.EXPECTED_BINDING_HASH = EXPECTED_TARGET_BINDING_SHA

    def load_cache_fs1(path: Path):
        if sha(path) != EXPECTED_CACHE_SHA:
            raise RuntimeError('CONDITIONING_CACHE_SHA_DRIFT')
        with np.load(path, allow_pickle=False) as z:
            data = {k: np.asarray(z[k]) for k in z.files}
        required = ('surface_ids', 'joint_ids', 'surface_features', 'joint_features', 'teacher_weights', 'teacher_supervision_mask', 'rest_points_world')
        if any(k not in data for k in required):
            raise RuntimeError('CACHE_SCHEMA_MISSING')
        if data['surface_features'].shape != (950, 20) or data['joint_features'].shape != (22, 8) or data['teacher_weights'].shape != (950, 22):
            raise RuntimeError('CACHE_SHAPE_DRIFT')
        supervised = int(data['teacher_supervision_mask'].sum())
        low = int((~data['teacher_supervision_mask'].astype(bool)).sum())
        if supervised != EXPECTED_SUPERVISED_ROWS or low != EXPECTED_LOW_ROWS:
            raise RuntimeError('SUPERVISION_MASK_DRIFT')
        if raw_array_sha(np.asarray(data['teacher_weights'], dtype=np.float32)) != EXPECTED_TEACHER_WEIGHTS_SHA:
            raise RuntimeError('TEACHER_WEIGHT_CONTENT_DRIFT')
        return data

    hist.load_cache = load_cache_fs1


def _tree_equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b): return bool(torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)): return len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def _reset_seed() -> None:
    random.seed(hist.SEED); np.random.seed(hist.SEED); torch.manual_seed(hist.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(hist.SEED)


def _new_objects(device):
    codec = hist.SkinFieldCodecV1().to(device)
    if codec.config.config_hash != hist.EXPECTED_CODEC_HASH: raise RuntimeError('CODEC_CONFIG_HASH_DRIFT')
    opt = torch.optim.AdamW(codec.parameters(), lr=hist.LR, weight_decay=hist.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hist.MAX_STEPS, eta_min=0.0)
    return codec, opt, sched


def run_regression(cache: Path, zero_surface: Path, camera_dir: Path, qualified_skeleton: Path, output_dir: Path) -> dict:
    device = torch.device('cpu')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(min(4, __import__('os').cpu_count() or 1))
    surface = hist.load_surface(zero_surface, camera_dir); sk = hist.load_skeleton(qualified_skeleton); data = hist.load_cache(cache); sids, jids = hist.validate_binding(surface, sk, data)
    T = hist.tensors(data, device); transforms = hist.probe_transforms(len(jids), device)

    _reset_seed(); c1, o1, s1 = _new_objects(device); losses_ref = []
    for _ in range(3): losses_ref.append(hist.train_step(c1, o1, T, transforms)); s1.step()
    ref = ({k: v.detach().cpu().clone() for k, v in c1.state_dict().items()}, o1.state_dict(), s1.state_dict(), hist.rng_state(), float(o1.param_groups[0]['lr']))

    _reset_seed(); c2, o2, s2 = _new_objects(device); losses_resume = []
    for _ in range(2): losses_resume.append(hist.train_step(c2, o2, T, transforms)); s2.step()
    progress = output_dir / 'ARACHNE_MAGE_A0_FS1_DISPOSABLE_RESUME.pt'
    hist.save_progress(progress, c2, o2, s2, 2, 2, [{'step': 1}, {'step': 2}], 0.0)
    c3, o3, s3 = _new_objects(device)
    ck = torch.load(progress, map_location='cpu', weights_only=False)
    if ck.get('schema') != hist.SCHEMA + '.Progress' or ck.get('codec_config_hash') != hist.EXPECTED_CODEC_HASH or ck.get('cache_sha256') != EXPECTED_CACHE_SHA or ck.get('binding_sha256') != EXPECTED_TARGET_BINDING_SHA:
        raise RuntimeError('RESUME_FINGERPRINT_MISMATCH')
    c3.load_state_dict(ck['model']); o3.load_state_dict(ck['optimizer']); hist._optimizer_state_to_device(o3, device); s3.load_state_dict(ck['scheduler']); hist.set_rng(ck['rng'])
    if int(ck['step']) != 2 or int(ck['streak']) != 2 or len(ck['trace']) != 2: raise RuntimeError('RESUME_METADATA_DRIFT')
    losses_resume.append(hist.train_step(c3, o3, T, transforms)); s3.step()
    resumed = ({k: v.detach().cpu().clone() for k, v in c3.state_dict().items()}, o3.state_dict(), s3.state_dict(), hist.rng_state(), float(o3.param_groups[0]['lr']))
    exact_resume = losses_ref == losses_resume and all(_tree_equal(a, b) for a, b in zip(ref, resumed))
    if not exact_resume: raise RuntimeError('EXACT_RESUME_REGRESSION_FAILED')

    terminal = output_dir / 'terminal_idempotence'; terminal.mkdir(parents=True, exist_ok=True)
    base = {'schema': hist.SCHEMA, 'surface_hash': hist.EXPECTED_SURFACE_HASH, 'skeleton_hash': hist.EXPECTED_SKELETON_HASH,
            'cache_sha256': EXPECTED_CACHE_SHA, 'binding_sha256': EXPECTED_TARGET_BINDING_SHA, 'codec_config_hash': hist.EXPECTED_CODEC_HASH}
    no_result = terminal / 'NO_RESULT.json'; no_ck = terminal / 'NO_CK.pt'; write_json(no_result, {**base, 'status': 'NO_A0_TERMINAL_CLOSURE', 'final_step': hist.MAX_STEPS, 'terminal_streak': 0}); no_before = sha(no_result)
    no_ok = hist._existing_result_exit(no_result, no_ck) and hist._existing_result_exit(no_result, no_ck) and sha(no_result) == no_before
    pass_result = terminal / 'PASS_RESULT.json'; pass_ck = terminal / 'PASS_CK.pt'; pass_ck.write_bytes(b'FS1_TERMINAL_IDEMPOTENCE_PROBE')
    write_json(pass_result, {**base, 'status': 'A0_TERMINAL_PASS', 'final_step': 96, 'closure_step': 96, 'terminal_streak': 3, 'qualified_codec_checkpoint_sha256': sha(pass_ck)}); pass_before = (sha(pass_result), sha(pass_ck))
    pass_ok = hist._existing_result_exit(pass_result, pass_ck) and hist._existing_result_exit(pass_result, pass_ck) and (sha(pass_result), sha(pass_ck)) == pass_before
    if not (no_ok and pass_ok): raise RuntimeError('TERMINAL_IDEMPOTENCE_REGRESSION_FAILED')

    report = {'schema': SCHEMA + '.Regression', 'status': 'PASS_CPU_RESUME_AND_TERMINAL_IDEMPOTENCE_REGRESSIONS', 'device': 'cpu', 'exact_resume': True,
              'resume_checkpoint_step': 2, 'resume_next_step': 3, 'model_state_exact': True, 'optimizer_state_exact': True, 'scheduler_state_exact': True,
              'rng_state_exact': True, 'loss_trace_exact': True, 'lr_exact': True, 'terminal_no_closure_idempotent': True, 'terminal_pass_idempotent': True,
              'cache_sha256': EXPECTED_CACHE_SHA, 'cache_manifest_sha256': EXPECTED_CACHE_MANIFEST_SHA, 'cache_binding_sha256': EXPECTED_CACHE_BINDING_SHA,
              'binding_sha256': EXPECTED_TARGET_BINDING_SHA, 'supervised_rows': EXPECTED_SUPERVISED_ROWS, 'low_rows': EXPECTED_LOW_ROWS, 'main_scientific_a0_optimizer_steps': 0}
    write_json(output_dir / 'ARACHNE_MAGE_A0_FS1_RUNNER_REGRESSION.json', report)
    return report


def repair_result_metadata(output_dir: Path) -> None:
    result_path = output_dir / 'ARACHNE_MAGE_A0_RESULT.json'
    if not result_path.exists(): return
    d = json.loads(result_path.read_text(encoding='utf-8'))
    if d.get('schema') != SCHEMA: raise RuntimeError('RESULT_SCHEMA_DRIFT_AFTER_RUN')
    d['cache_manifest_sha256'] = EXPECTED_CACHE_MANIFEST_SHA
    d['cache_binding_sha256'] = EXPECTED_CACHE_BINDING_SHA
    d['teacher_weights_content_sha256'] = EXPECTED_TEACHER_WEIGHTS_SHA
    d['confidence_policy'] = {'supervised_rows': EXPECTED_SUPERVISED_ROWS, 'low_rows': EXPECTED_LOW_ROWS, 'low_in_encoder': False, 'low_in_loss': False}
    write_json(result_path, d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True); ap.add_argument('--cache-manifest', type=Path, required=True)
    ap.add_argument('--zero-surface', type=Path, required=True); ap.add_argument('--camera-dir', type=Path, required=True); ap.add_argument('--qualified-skeleton', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--require-cuda', action='store_true'); ap.add_argument('--preflight-only', action='store_true'); ap.add_argument('--regression-only', action='store_true')
    args = ap.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only and args.regression_only: raise RuntimeError('CHOOSE_ONE_DISPOSABLE_MODE')
    patch_historical_runner(args.cache_manifest)
    if args.regression_only:
        report = run_regression(args.cache, args.zero_surface, args.camera_dir, args.qualified_skeleton, args.output_dir); print('A0_FS1_REGRESSION=' + json.dumps(report, sort_keys=True)); return 0
    forwarded = ['--cache', str(args.cache), '--zero-surface', str(args.zero_surface), '--camera-dir', str(args.camera_dir), '--qualified-skeleton', str(args.qualified_skeleton), '--output-dir', str(args.output_dir)]
    if args.require_cuda: forwarded.append('--require-cuda')
    if args.preflight_only: forwarded.append('--preflight-only')
    rc = hist.main(forwarded)
    if not args.preflight_only: repair_result_metadata(args.output_dir)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
