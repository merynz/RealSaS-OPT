# RealSaS — Arachne / SkinFieldCodec behavioral panel preregistration

**Date:** 2026-09-03  
**Status:** `PREREGISTERED_BEFORE_BEHAVIORAL_EXECUTION`  
**Scope:** generic synthetic capacity + train/decode/Compiler/deformation integrity  
**Real-family data:** `FORBIDDEN`  
**Threshold changes after first execution:** `FORBIDDEN`

## 1. Scientific question

Does the current generic Arachne/Codec stack preserve the intended behavior across the full authority chain:

`teacher W -> Codec A0 encode/decode ceiling -> freeze Codec -> Arachne(S, qualified G) -> decoded W* -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR W -> verified LBS deformation`

A source-level finite loss, simplex softmax, or successful `qualify_skin` call is not behavioral PASS authority.

## 2. Firewall

The panel contains no real-family identifiers, geometry, counts, thresholds or learned checkpoints.

In particular it contains no `rigxl_03490` information and may not be modified in response to any real-family result.

The three witness definitions, seeds, optimizer protocols and PASS thresholds below are frozen before first behavioral execution. A later repair must fix a generic cause and preserve this panel unchanged.

## 3. Shared architecture/protocol

Codec:
- `SkinFieldCodecV1`
- hidden dim `32`
- latent dim `8`
- encoder layers `2`
- decoder layers `2`

Arachne:
- `ArachneCandidateV2`
- model dim `32`
- surface encoder layers `1`
- attention heads `4`
- feedforward dim `64`
- Codec object is shared and frozen before A1 optimization.

Execution:
- CPU deterministic algorithms;
- one witness is optimized independently from its frozen seed;
- same architecture/hyperparameters for every witness;
- no witness-dependent loss weights or thresholds.

### Codec A0 optimizer

- AdamW
- learning rate `1e-3`
- weight decay `1e-4`
- max steps `1536`
- evaluate every `32` steps
- require `3` consecutive PASS evaluations.

### Arachne A1 optimizer

- AdamW over trainable Arachne parameters only
- learning rate `3e-4`
- weight decay `1e-4`
- max steps `2048`
- evaluate every `32` steps
- require `3` consecutive PASS evaluations.

Base loss weights remain the shipping source defaults:
- A0 reconstruction `1.0`, deformation `1.0`;
- A1 latent NLL `1.0`, reconstruction `1.0`, deformation `1.0`, hard-tail `1.0`;
- A1 tail fraction `0.10`.

## 4. Synthetic witness construction

All surfaces contain deterministic observed `SurfaceNode` positions with exact local normals and simple LOCAL adjacency. All skeletons are already qualified canonical skeletons. Dense teacher skin is generated only from synthetic point/joint geometry by normalized radial affinity:

`a_ij = exp(-||P_i - J_j||^2 / (2 sigma^2))`

`W_ij = a_ij / sum_k a_ik`

Thus teacher rows are nonnegative simplex rows and contain no semantic labels beyond synthetic geometry.

Every probe bank contains identity plus deterministic distinct per-joint translations in x/y/z. Probe-motion scale is measured from teacher deformation itself; deformation PASS uses a dimensionless ratio rather than an absolute world-unit threshold.

### Witness 1 — `chain_blend_3`

- seed `20260921`
- `N=9`, `J=3`
- topology `[-1, 0, 1]`
- surface follows a shallow sinusoidal chain in x/y/z
- three joints lie along the chain
- radial sigma `0.32`
- purpose: smooth serial blend and parent-conditioned joint field.

### Witness 2 — `branch_blend_4`

- seed `20260922`
- `N=12`, `J=4`
- topology `[-1, 0, 0, 0]`
- surface occupies a deterministic branched / fan-shaped point set
- root plus three branch joints
- radial sigma `0.28`
- purpose: competing sibling influences and non-chain topology.

### Witness 3 — `sharp_fork_5`

- seed `20260923`
- `N=15`, `J=5`
- topology `[-1, 0, 0, 1, 2]`
- surface contains a fork with narrow transition regions
- radial sigma `0.18`
- purpose: sharper near-sparse rows, branch transitions and hard-tail sensitivity.

## 5. Dimensionless behavioral metrics

For valid surface rows:

`row_L1_i = sum_j |W_pred(i,j) - W_teacher(i,j)|`

`row_L1_p95 = quantile_0.95(row_L1)`

Teacher probe-motion RMS:

`motion_rms = RMS(LBS(W_teacher,T) - rest)`

Prediction deformation error:

`deformation_rms = RMS(LBS(W_pred,T) - LBS(W_teacher,T))`

Normalized deformation error:

`deformation_ratio = deformation_rms / max(motion_rms, 1e-6)`

The ratio makes thresholds invariant to a global rescaling of synthetic probe displacement.

## 6. Frozen PASS thresholds

### Codec A0 authoritative PASS

At one evaluation:
- `row_L1_p95 <= 0.05`;
- `deformation_ratio <= 0.05`;
- max simplex residual `<= 1e-6`;
- negative weight count `== 0`.

Require 3 consecutive evaluations. Only then may the witness emit an A0 PASS token and freeze that Codec for A1.

### Arachne A1 authoritative PASS

Evaluation must use **shipping output**, not teacher latent decode:

`Arachne.propose -> Compiler.qualify_skin -> QualifiedSkinIR`.

Reconstruct the qualified W matrix from canonical surface/joint IDs and run the same verified LBS probes.

At one evaluation:
- every expected surface row exists in `QualifiedSkinIR`;
- every influence references a legal qualified canonical joint;
- `qualified row_L1_p95 <= 0.10`;
- `qualified deformation_ratio <= 0.10`;
- qualified row simplex residual `<= 1e-6`;
- no negative qualified weights;
- Compiler `total_correction_l1 <= 1e-5` so PASS cannot be manufactured by a large downstream repair.

Require 3 consecutive evaluations.

A single transient PASS is not closure.

## 7. Panel closure

`ARACHNE_CODEC_BEHAVIORAL_PANEL = PASS` only if all three preregistered witnesses independently achieve:
- sustained A0 PASS;
- frozen-Codec A1 optimization;
- sustained shipping proposal -> Compiler-qualified W -> LBS PASS.

If any witness fails, the panel is `FAIL` and the witness definition/threshold/seed/protocol stays frozen. Diagnostics may be added, but repair must address the first generic layer where truth behavior is lost.

## 8. What this panel does not claim

PASS does not establish real-family fit or generalization. It closes only generic capacity and behavioral alignment for the Codec/Arachne/Compiler-skin seam.

No architecture refreeze or Family-1 authorization follows automatically from this panel.
