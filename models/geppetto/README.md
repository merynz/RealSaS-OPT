# Geppetto

Geppetto is the RealSaS learned anonymous skeleton/control proposal subsystem.

## Current mainline

Current V2 inference source is promoted at:

`models/geppetto/v2/`

The three implementation modules are byte-identical to their audited source counterparts in `experiments/geppetto_arachne_r6_20260901/`:

- `geppetto_candidate_v2.py`
- `geppetto_conditioning_v2.py`
- `geppetto_checkpoint_v2.py`

The dated experiment tree remains intact as provenance and contains older V1 architecture, training experiments, capacity diagnostics and behavioral panels.

## Authority boundary

Geppetto consumes admitted `RiggingSurfaceIR`-derived conditioning and emits anonymous, multimodal control evidence as `SkeletonProposalIR`.

Geppetto does **not** own canonical joint IDs, final root/tree/forest selection, graph legality, skeleton qualification, product state or proof. Those remain Compiler-owned.

## Why training is not promoted yet

Current V2 loss/train code still reaches a shared historical teacher-target contract (`training_targets_v1.py`) that itself depends on the older `conditioning_v1.py` lane. That may still be scientifically valid training apparatus, but it is not automatically part of current V2 inference ownership.

It remains under audit rather than being smuggled into mainline merely to make the folder look complete.

## Older source

`geppetto_candidate_v1.py`, `geppetto_loss_v1.py`, V1 candidate config and oracle/behavioral harnesses remain experiment/historical evidence unless separately promoted by an explicit decision.
