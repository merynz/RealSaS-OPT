# RealSaS — Geppetto Reference-Strength FIT1 Preregistration V1

Status: `DESIGN_FROZEN__NOT_ACTIVE__NO_TRAINING_AUTHORIZED_BY_THIS_FILE`

Branch: `exp/geppetto-reference-strength-fullstack-v1-20260907`

## Single scientific question

Can a shipping-faithful Geppetto formulation that consumes the full current
`RiggingSurfaceIR` evidence available at product time, uses prediction-only
causal mechanical recurrence, and emits learned all-pairs parent evidence fit
the generic Mage mechanical core with terminal structural stability on the
same FIT family?

This is a capability-closure experiment, not a mechanism ablation and not a
generalization experiment.

## Frozen ownership boundary

Learner input is restricted to deterministic tensorization of current
`RiggingSurfaceIR`:

- normalized surface position `P`;
- robust GSA normal `N` + validity;
- per-view support mask;
- production raster bindings for all eight canonical views;
- observed/completed state;
- exact GSA local-relation graph + deterministic relation attributes;
- deterministic canonical-view direction code derived from the product's
  8-view 45-degree yaw contract.

Forbidden learner input:

- teacher mesh vertices/faces/normals;
- teacher joint/control positions;
- teacher parent/root identity;
- teacher skin weights;
- source bone/control names;
- source control indices as features;
- Mage-specific semantic labels;
- learned free absolute view-slot embeddings;
- fixed Mage/product output cardinality.

Teacher skeleton and skin may be read only to build and evaluate the FIT target.
Training recurrence remains prediction-only. Teacher positions may enter the
non-recurrent diffusion objective exactly as implemented by the frozen
candidate, but teacher residuals may not enter recurrent state.

Final legal tree authority remains the Compiler. Geppetto emits root evidence
and all-pairs directed parent evidence; generation-step parent feedback is not
canonical parent authority.

## Frozen upstream witness

IRIS/GSA source:

- promoted Mage scene-first signed run: `20260904T220929Z`;
- IRIS checkpoint SHA-256:
  `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2`;
- clipped signed zero-surface Drive ID: `1ov5QL3H4nVNDAQmVI4j0qcr5iOmC7-cT`;
- clipped signed zero-surface SHA-256:
  `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`;
- exact product camera contract: 8 orthographic yaw views at 45-degree increments,
  resolution 1024;
- GSA `target_nodes=1024`, robust local PCA `k=64`, scene-first self-zbuffer
  support with `visibility_depth_tolerance_norm=0.02`.

Preflight must reproduce the known current-Mage substrate witness before the
first optimizer step:

- compact surface nodes: `950`;
- exact undirected GSA topology edges: `2813`;
- teacher truth entering surface features: `false`;
- production raster channels non-degenerate.

Any mismatch aborts before model construction.

## Frozen teacher target

Target builder:
`RealSaS.GeppettoMechanicalCoreTarget.v1`.

Rule:
`SKIN_SUPPORTED_PLUS_SUPPORTED_BRIDGES__ASSEMBLY_ONLY_ROOT_EXCLUDED`.

The projection must be derived from source parent/deform/skin arrays without
bone names or a family-specific index list. Current Mage preflight independently
produces 22 controls and one projected root; `22` is a FIT target result, never
an architectural output cap.

Position authority is explicitly world/source frame. For the current normalized
corpus the accepted locus source is:
`rest_world_source[:, :3, 3]`.
The corpus `bone_heads` array is canonical-normalized and is forbidden as direct
world-locus supervision unless explicitly transformed back to world/source frame.

Pinned current Mage target content SHA-256:
`0b5a25c877116de60b710b7bb2a7848f30988e1622e2eda8084cad21c8ca23c9`.

A prior draft target hash based on canonical-normalized `bone_heads` is invalid
and was never authorized for training. Any current target hash drift aborts
before the first optimizer step.

## Frozen model arm

Architecture:
`RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion.DeterministicViewDirection.v1`.

Core formulation:

1. factorized direct surface evidence encoder;
2. exact GSA-relation message passing;
3. global full-surface transformer memory;
4. multi-layer prediction-only causal skeleton recurrence;
5. full-surface cross-attention at every generation step;
6. coarse continuous 3D locus prediction;
7. conditional 3D diffusion residual refinement;
8. soft learned previous-parent distribution for internal mechanical feedback;
9. native learned STOP/existence/root/mechanical-salience/support heads;
10. separate all-pairs learned directed-parent evidence after generation;
11. Compiler exact tree qualification after proposal generation.

No fixed product K is permitted. `resource_step_limit` is execution budget only.

## Training horizon and checks

Hardware target: A100-class CUDA.

Frozen optimizer horizon:

- maximum optimizer steps: `16384`;
- check interval: `64` optimizer steps;
- terminal stability requirement: final contiguous `48` checks = `3072`
  optimizer steps satisfying the full structural gate;
- seed: `20260907`;
- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- no scheduler in V1;
- gradient clipping: global norm `1.0`;
- deterministic algorithm mode where supported.

Changing architecture, optimizer, learning rate, target projection, surface
input, serialization or gate thresholds after the first optimizer step closes
this preregistration as invalid; a new preregistration is required.

## Full structural check

A check is `PASS` only if all conditions hold in free-running shipping mode:

1. native STOP generated count equals target count exactly;
2. matched joint-locus normalized P95 <= `0.05`;
3. matched joint-locus normalized MAE <= `0.03`;
4. Compiler-qualified joint count equals target count;
5. exactly one qualified deform root for this FIT target;
6. teacher-aligned root accuracy = `1.0` after deterministic evaluation matching;
7. teacher-aligned parent accuracy = `1.0` after Compiler qualification;
8. illegal parent references = `0`;
9. unsupported qualified joints = `0`;
10. non-finite proposal/qualification values = `0`;
11. teacher feedback used = `false`;
12. learned absolute view-slot parameter count = `0`.

Diffusion robustness subcheck: the structural gate must hold for the frozen set
of inference seeds `{11, 23, 47, 89}`. A check fails if any required seed fails.
The same model weights are used for all four seeds.

The final result is `FIT1_TERMINAL_PASS` only when the run ends with at least 48
consecutive full structural PASS checks. Reaching the region earlier and later
leaving it is not closure.

## Stop / interpretation rules

- Non-finite training loss: immediate fail-close.
- Upstream/target/hash preflight mismatch: `INVALID_PREFLIGHT`, no training.
- No 48-check terminal streak by step 16384: `NO_TERMINAL_CLOSURE`.
- PASS does not authorize held-out/generalization claims.
- PASS does not automatically refreeze Geppetto mainline.
- PASS does not authorize Arachne until a separate promotion/refreeze decision.
- FAIL does not prove Geppetto impossibility; it falsifies this frozen formulation
  under this frozen horizon/gate only.

## Artifact requirements

The run must emit:

- exact repo commit SHA;
- config hash;
- source zero-surface SHA;
- target content SHA;
- tensorization certificate/hash;
- optimizer seed/config;
- complete check trajectory;
- four-seed final free-running reports;
- final `SkeletonProposalIR`;
- Compiler qualification report;
- checkpoint SHA-256;
- final decision JSON;
- human-readable skeleton visualization derived from the qualified artifact.

No artifact may be described as promoted or canonical solely because this FIT1
preregistration passes.
