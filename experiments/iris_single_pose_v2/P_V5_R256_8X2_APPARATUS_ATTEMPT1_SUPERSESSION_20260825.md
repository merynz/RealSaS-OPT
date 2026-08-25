# P-V5 R256 8×2 — Apparatus Attempt 1 Supersession

**Date:** 2026-08-25  
**Scientific result:** NONE  
**Scientific optimizer steps:** 0

The first A100 8×2 execution is superseded as an apparatus-only failure.

## Observed completed gates
- CPU preflight PASS.
- A100 GPU preflight PASS on `NVIDIA A100-SXM4-40GB`.
- 8 assets / 16 cells stage PASS.
- shared truth/cache construction PASS.
- PREOPT authority froze the intended 8×2 protocol at scientific optimizer step 0.

## Root cause
`train_pv5_r256_8x2_v1.py` placed CUDA validation and initialization on one semicolon-separated Python `if` suite:

```python
if not torch.cuda.is_available(): raise RuntimeError('CUDA required'); seed_all(); device=torch.device('cuda'); out=...
```

In Python, every semicolon-separated statement after the colon belongs to the conditional suite. On a CUDA runtime the condition is false, so `seed_all()`, `device=...`, and output-directory initialization were skipped. The trainer therefore failed before model initialization/training when `device` was subsequently referenced.

## Classification
`APPARATUS_IMPLEMENTATION_BUG__OPTIMIZER_ZERO`

This does not falsify the R256 representation, the learner, the optimizer schedule, the 8×2 joint-capacity question, or the frozen PASS threshold.

## V1.1 apparatus changes only
- move CUDA check and initialization onto explicit unconditional lines after the guard;
- strengthen A100 GPU preflight to execute both accumulation microbatches before declaring PASS;
- persist the child training process output as `TRAIN_PROCESS.log` for any future failure.

The scientific protocol, membership, architecture, objective, optimizer schedule, checkpoint set, and PASS/FAIL criterion remain unchanged.
