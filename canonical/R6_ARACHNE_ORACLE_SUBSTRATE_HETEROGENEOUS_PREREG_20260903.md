# RealSaS — R6 Arachne Heterogeneous U1 Preregistration — 2026-09-03

**Status:** `PREREGISTERED_AFTER_ONE_FAMILY_U0_U1_PASS__BEFORE_HETEROGENEOUS_OUTPUT`

## Entry condition

The one-family `branch_blend_4` R6-A1 gate passed both causal arms under the frozen protocol:

- U0 default shipping Codec A0 PASS;
- U0 default shipping Arachne A1 PASS;
- U1 default shipping Codec A0 PASS;
- U1 default shipping Arachne A1 PASS.

This authorizes the heterogeneous U1 rung specified by `canonical/R6_ARACHNE_ORACLE_SUBSTRATE_PREREG_20260903.md`.

## Frozen panel

Use exactly the existing preregistered witness set and seeds:

1. `chain_blend_3` — seed `20260921`;
2. `branch_blend_4` — seed `20260922`;
3. `sharp_fork_5` — seed `20260923`.

No witness geometry, joint loci, parent semantics or skin sigma may change.

## Arm

Run **U1 only**:

`exact synthetic observations -> ObservationEvidenceIR -> persistence -> analytic P=O+dF -> GeometricSubstrateAssembler -> DTB-ND1 where available -> RiggingSurfaceIR -> Compiler-qualified oracle G -> ArachneConditioningAdapterV2`.

The exact already committed shell/camera/visibility apparatus in `oracle_substrate_synthetic_v1.py` is frozen. No hidden full-surface completion or source mesh is allowed into the consumer.

## Per-witness A0 prerequisite

Each witness must independently qualify the actual default shipping `SkinFieldCodecV1` on its own admitted U1 `(S,G,W)` shape before A1.

Frozen Codec config hash:

`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

Frozen protocol:

- AdamW `lr=1e-3`, `weight_decay=1e-4`;
- cosine annealing to zero over max `1536` steps;
- check every `32`;
- `3` consecutive PASS checks.

Frozen A0 acceptance:

- raw and qualified row-L1 p95 `<= 0.05`;
- raw and qualified deformation ratio `<= 0.05`;
- simplex residual `<= 1e-6`;
- no negative weights;
- Compiler correction L1 `<= 1e-5`;
- exact admitted row coverage.

A1 must not run for a witness whose A0 token fails.

## Per-witness A1

Use actual default shipping `ArachneCandidateV2` with frozen qualified Codec decoder.

Frozen Arachne config hash:

`ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`

Frozen protocol:

- AdamW `lr=3e-4`, `weight_decay=1e-4`;
- max `2048` steps;
- check every `32`;
- `3` consecutive PASS checks.

Frozen A1 acceptance:

- raw and qualified row-L1 p95 `<= 0.10`;
- raw and qualified deformation ratio `<= 0.10`;
- simplex residual `<= 1e-6`;
- no negative weights;
- Compiler correction L1 `<= 1e-5`;
- exact admitted row coverage.

## Product boundary

Every A1 prediction must traverse:

`raw decoded W -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

Raw W must independently pass the same static/deformation bands; Compiler correction cannot rescue a semantic failure.

Qualified G must come from the same explicitly labeled oracle mechanical proposal -> real `Compiler.qualify_skeleton_v2` path used by the one-family rung. Canonical IDs remain Compiler-owned.

## Panel PASS

Panel PASS requires all three witnesses to achieve:

- A0 PASS with `>=3` consecutive checks;
- A1 PASS with `>=3` consecutive checks;
- correct Codec and Arachne config hashes;
- actual Compiler skin qualification;
- raw and qualified acceptance simultaneously.

No averaging may hide a failing witness.

## Interpretation

- any A0 failure -> Arachne-U1 interpretation blocked for that witness; localize Codec/shape capacity first;
- A0 PASS + A1 FAIL -> current Arachne apparatus cannot preserve sufficient U1 information on that witness;
- `3/3 A0 + A1 PASS` -> current heterogeneous small-panel observation-oracle Arachne gate closes.

Because measured U1 visibility for the current capsule shells is near-complete, a `3/3 PASS` still does not establish a strong material-self-occlusion claim. That remains a separately scoped strengthening obligation.

## Change control

After first heterogeneous output, no seed, witness, sigma, camera, visibility, substrate, optimizer, threshold, architecture hash or Compiler correction allowance may be changed merely to obtain PASS.
