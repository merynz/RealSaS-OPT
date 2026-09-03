# RealSaS — R6 Arachne Oracle-Substrate Preregistration — 2026-09-03

**Status:** `PREREGISTERED_BEFORE_OUTPUT__ONE_FAMILY_U0_THEN_U1`

## Purpose

Test the remaining R6-A1 information-preservation question at the actual shipping semantic boundary:

`RiggingSurfaceIR S + Compiler-qualified G -> shipping/default ArachneCandidateV2 -> frozen shipping SkinFieldCodecV1 decoder -> raw W -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

This gate follows the already closed shipping A0/A1 synthetic behavioral boundary and the R6 causal-arm contract. It asks whether the same Arachne apparatus can infer a useful skin latent from reference-rich U0 versus shipping-observable oracle U1 substrate, before predicted IRIS error.

## Frozen one-family witness

Witness: `branch_blend_4`

Seed: `20260922`

The mechanical witness definition, joint loci, parent semantics and skin sigma are inherited unchanged from the preregistered Arachne behavioral panel.

## Frozen substrate arms

The geometric carrier and camera/visibility machinery are exactly the already committed R6 synthetic oracle apparatus:

`experiments/geppetto_arachne_r6_20260901/oracle_substrate_synthetic_v1.py`

No camera, shell-sampling, visibility, persistence, DTB-ND1 or raster parameter may be changed after observing Arachne output.

### U0_REFERENCE_FULL_SURFACE

Use the authoritative synthetic full shell converted only into `RiggingSurfaceIR` consumer format.

This is a reference-rich diagnostic upper bound and cannot establish product input equivalence by itself.

### U1_OBSERVATION_ORACLE_SUBSTRATE

Use:

`exact synthetic observations -> ObservationEvidenceIR -> persistence -> analytic P=O+dF -> GeometricSubstrateAssembler -> observed local geometry / DTB-ND1 where available -> RiggingSurfaceIR`.

No hidden full-surface completion, source mesh or teacher normal is admitted to the U1 consumer input.

## Frozen qualified-G input

Arachne is not allowed to consume source-rig hidden identity. For each arm, the synthetic mechanical control loci are converted into an explicitly labeled oracle mechanical `SkeletonProposalIR`, bound to that arm's surface lineage, and then passed through the real:

`Compiler.qualify_skeleton_v2`.

The resulting `QualifiedSkeletonIRV2` is the only G input to Arachne.

This isolates the Arachne seam: Geppetto prediction error is intentionally not part of R6-A1. Canonical joint IDs remain Compiler-owned.

## Frozen dense skin truth

For each admitted surface row and each Compiler-qualified joint, define evaluation/training truth only as:

`a_ij = exp(-||P_i - J_j||^2 / (2 sigma^2))`

followed by row normalization:

`W_ij = a_ij / sum_j a_ij`.

`sigma` is the existing `branch_blend_4` witness value `0.28` and is not changed between U0 and U1.

Truth rows are rebound by canonical `conditioning.surface_ids`; joint columns are rebound by canonical `conditioning.joint_ids`. Incidental Python creation order is forbidden as authority.

## A0 prerequisite inside each arm

Before A1, the actual default shipping Codec must earn a capacity token on that arm's admitted `(S,G,W)` shape.

Frozen Codec:

- architecture: `SkinFieldCodecV1()` default shipping config;
- expected config hash: `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`;
- hidden dim `192`;
- latent dim `64`;
- encoder layers `3`;
- decoder layers `3`.

Frozen optimizer protocol:

- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- `CosineAnnealingLR(T_max=1536, eta_min=0)`;
- check every `32` steps;
- maximum `1536` steps;
- `3` consecutive PASS checks required.

Existing shipping A0 acceptance remains authoritative:

- raw row-L1 p95 `<= 0.05`;
- qualified row-L1 p95 `<= 0.05`;
- raw deformation-error / teacher-motion RMS `<= 0.05`;
- qualified deformation-error / teacher-motion RMS `<= 0.05`;
- raw and qualified simplex residual `<= 1e-6`;
- no negative weights;
- Compiler total correction L1 `<= 1e-5`;
- qualified row count equals admitted surface row count.

A1 is forbidden on an arm whose A0 token does not PASS.

## Frozen A1 apparatus

Frozen Arachne:

- `ArachneCandidateV2(codec, ArachneCandidateConfigV2(), freeze_codec=True)`;
- expected config hash: `ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`;
- model dim `128`;
- surface encoder layers `2`;
- attention heads `4`;
- feedforward dim `384`;
- qualified A0 Codec parameters frozen.

Frozen optimizer:

- AdamW over trainable Arachne parameters only;
- `lr=3e-4`, `weight_decay=1e-4`;
- check every `32` steps;
- maximum `2048` steps;
- `3` consecutive PASS checks required.

Existing shipping A1 acceptance remains authoritative:

- raw row-L1 p95 `<= 0.10`;
- qualified row-L1 p95 `<= 0.10`;
- raw deformation-error / teacher-motion RMS `<= 0.10`;
- qualified deformation-error / teacher-motion RMS `<= 0.10`;
- raw and qualified simplex residual `<= 1e-6`;
- no negative weights;
- Compiler total correction L1 `<= 1e-5`;
- qualified row count equals admitted surface row count.

Latent p95 error and uncertainty are telemetry, not independent PASS authority.

## Required output telemetry

For each arm record at least:

- admitted surface node count;
- U1 visible/full sample fraction where applicable;
- U1 support-view distribution;
- U1 DTB-ND1 normal coverage;
- U1 local-relation count;
- qualified skeleton joint/root count and lineage;
- A0 pass step + final raw/qualified static/deformation metrics;
- A1 pass step + final raw/qualified static/deformation metrics;
- Compiler correction magnitude;
- Arachne and Codec config hashes.

## Causal interpretation

- `U0 A0 FAIL` -> shipping Codec capacity/protocol failure on the richer oracle shape; A1 interpretation forbidden.
- `U0 A0 PASS + U0 A1 FAIL` -> Arachne apparatus/latent inference failure under reference-rich geometry.
- `U0 PASS + U1 A0 FAIL` -> representation/protocol cannot support the admitted observation-limited shape; A1 interpretation forbidden.
- `U0 A1 PASS + U1 A0 PASS + U1 A1 FAIL` -> observation-limited substrate is insufficient for the current Arachne apparatus at this gate.
- `U0 A1 PASS + U1 A1 PASS` -> complete hidden/full surface is not required for Arachne on this gate's admitted observation substrate; stronger self-occlusion claims remain limited by measured coverage contrast.

Compiler qualification may not be credited as semantic prediction quality; raw W must independently satisfy the same row/deformation acceptance band.

## Change control

After the first output is observed, no witness, seed, shell/camera parameter, sigma, optimizer, threshold, architecture config, model hash, or Compiler correction allowance may be changed merely to obtain PASS.

A failure may only motivate a new separately preregistered diagnostic or a generic source repair with causal evidence.

## Sequence

1. `branch_blend_4` U0 A0 -> U0 A1;
2. only if U0 passes: `branch_blend_4` U1 A0 -> U1 A1;
3. only if one-family U1 passes: heterogeneous U1 Arachne panel may be opened in a separate preregistered rung;
4. U2 predicted-IRIS remains later.
