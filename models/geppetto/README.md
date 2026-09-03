# Geppetto

Geppetto is the RealSaS learned anonymous skeleton/control proposal subsystem.

## Current mainline

`models/geppetto/v2/`

Current inference source is byte-preserved from the audited R6 source:

- `geppetto_candidate_v2.py`
- `geppetto_conditioning_v2.py`
- `geppetto_checkpoint_v2.py`

Current V2 training/evaluation source is also visible in the same version package:

- `training_targets_v2.py` — byte-preserved V2 teacher projection;
- `geppetto_loss_v2.py` — byte-preserved anonymous/multimodal loss;
- `geppetto_train_v2.py` — byte-preserved train step;
- `geppetto_eval_v2.py` — byte-preserved evaluation metrics;
- `training_targets_v1.py` — narrow compatibility target type only, rebound so V2 training does not import the historical V1 conditioning implementation.

The dated experiment tree remains intact as provenance and contains older V1 architecture, capacity diagnostics, overfit/panel experiments and oracle-substrate gates.

## Authority boundary

Geppetto consumes admitted `RiggingSurfaceIR`-derived conditioning and emits anonymous, multimodal control evidence as `SkeletonProposalIR`.

Geppetto does **not** own canonical joint IDs, final root/tree/forest selection, graph legality, skeleton qualification, product state or proof. Those remain Compiler-owned.

## Training firewall

Teacher information is training/evaluation-only. Teacher control IDs may be used inside a target to reconstruct parent indices, but they never become proposal/product identity. Anonymous geometry matching and current V2 conditioning remain the learned training boundary.

## Older source

`geppetto_candidate_v1.py`, `geppetto_loss_v1.py`, V1 candidate config, oracle-substrate harnesses and behavioral diagnostics remain research/historical evidence unless explicitly promoted by a later decision.
