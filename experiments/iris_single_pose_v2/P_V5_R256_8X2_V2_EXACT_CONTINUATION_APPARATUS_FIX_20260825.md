# P-V5 R256 8x2 V2 exact continuation — apparatus package fix (2026-08-25)

Status: `APPARATUS_PACKAGING_BUG__OPTIMIZER_ZERO__FIXED_BEFORE_SCIENTIFIC_STEP_1`

## Incident
The first exact-continuation notebook bundle omitted `train_pv5_r256_8x2_v2.py`. `pv5_r256_8x2_v2_gpu_preflight.py` imports `configure_p_only` and `checkpoint_payload` from that module, so the GPU preflight terminated during Python import with no scientific optimizer step.

This is an apparatus/package dependency failure, not a scientific FAIL and not evidence about the model. The frozen exact-continuation preregistration, parent checkpoint, membership, objective, LR, +1024-step budget, and PASS rule are unchanged.

## Fix
- add `train_pv5_r256_8x2_v2.py` to the exact-continuation release bundle;
- add `pv5_r256_8x2_v2_exact_cont_package_preflight.py`;
- package preflight compiles all package Python sources, validates local dependency closure, imports every package module, and requires the GPU-preflight dependency file explicitly;
- notebook subprocess wrappers print child stdout/stderr before raising, so future apparatus failures are not opaque `CalledProcessError`s.

## Local zero-scientific-step verification
Before V1.1 handoff:
- all 17 Python modules compile: PASS;
- local import dependency closure: PASS;
- clean import of all 17 modules: PASS;
- GPU preflight executes through imports and reaches the expected no-CUDA guard locally with no `ModuleNotFoundError`/`ImportError`: PASS;
- exact parent checkpoint SHA/identity restore: PASS;
- optimizer state coverage: 218/218 trainable tensors;
- scaler and RNG state present;
- resumed AdamW synthetic CPU step advances all restored optimizer step counters from 4096 to 4097: PASS;
- scientific optimizer steps during all preparation/preflight: 0.

Fresh V2 remains immutable FAIL. Exact continuation scientific interpretation is unchanged.
