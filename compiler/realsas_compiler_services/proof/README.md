# Proof Services

This directory contains **subordinate diagnostic/measurement services** used by `compiler/realsas_compiler_core/`.

Authority rules:

1. `realsas_compiler_core` owns proof plans, proof status and canonical product truth.
2. A proof service may derive measurements or failure signatures from already-bound current-compiler inputs.
3. A failure signature is diagnostic evidence, **not owner attribution**.
4. Owner attribution requires a separate controlled causal mutation/fault experiment.
5. No service here may mutate canonical identity, silently repair a product, or convert FAIL/ABSTAIN to PASS.

`failure_signatures.py` is a semantic promotion/rebind of the useful diagnostic separation in historical v0.5 `realsas_deformation/failure_signatures.py`; it does not import or reactivate the historical contract stack.
