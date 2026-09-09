# RealSaS — Geppetto Reference-Strength Loss Freeze V1

Status: `BINDING_COMPANION_TO_FIT1_PREREG__NOT_ACTIVE`

This file closes the loss-definition gap in
`GEPPETTO_REFERENCE_STRENGTH_FIT1_PREREG_V1.md` before any optimizer step.
No training result exists before this freeze.

Implementation:
`geppetto_reference_strength_loss_v1.py`

## Frozen target preparation

- target positions: world/source mechanical-core loci normalized only by the
  current direct `RiggingSurfaceTensorV1` center/scale;
- parent/root: anonymous content-serialized mechanical-core topology;
- support target: deterministic nearest `k=8` surface rows in normalized
  Euclidean space, used as training/evaluation target only;
- source indices, bone names and skin values never enter learner features;
- teacher target values never enter recurrent state.

## Frozen weighted objective

| component | weight |
|---|---:|
| coarse position Smooth-L1 (`beta=0.02`) | 4.00 |
| heteroscedastic position NLL | 0.25 |
| conditional residual diffusion epsilon loss | 0.50 |
| native STOP BCE | 1.00 |
| existence BCE | 0.25 |
| root BCE | 1.00 |
| internal soft-parent CE | 1.00 |
| final all-pairs parent CE | 1.50 |
| support-presence BCE | 0.25 |
| nearest-surface support-set CE | 0.25 |
| projected mechanical-salience BCE | 0.10 |

Parent supervision is legal training target supervision. The recurrent parent
state remains the model's own soft distribution; no teacher parent embedding,
position or state is injected.

The final all-pairs parent objective trains evidence only. Compiler exact tree
qualification remains canonical parent/root authority.

## Invalidating changes

After activation, changing any weight, Smooth-L1 beta, support-target cardinality,
target normalization rule, teacher-feedback rule, or component definition
invalidates the FIT1 preregistration and requires a new preregistration.

This loss freeze does not activate the experiment and does not authorize a
Geppetto refreeze or generalization claim.
