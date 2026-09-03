# RealSaS — R6 Arachne Observation-Oracle Closure — 2026-09-03

**Status:** `PASS / CLOSED_AT_CURRENT_ORACLE_GATE__COVERAGE_STRESS_REMAINS_SEPARATE_STRENGTHENING_OBLIGATION`

## Scope and authorities

Preregistrations:
- `canonical/R6_ARACHNE_ORACLE_SUBSTRATE_PREREG_20260903.md`;
- `canonical/R6_ARACHNE_ORACLE_SUBSTRATE_HETEROGENEOUS_PREREG_20260903.md`.

One-family result:
- `canonical/R6_ARACHNE_ORACLE_SUBSTRATE_ONE_FAMILY_RESULT_20260903.md`.

The tested semantic boundary was:

`RiggingSurfaceIR S + Compiler-qualified oracle mechanical G -> ArachneConditioningAdapterV2 -> default shipping ArachneCandidateV2 -> frozen A0-qualified default shipping SkinFieldCodecV1 decoder -> raw W -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

Qualified G came from an explicitly labeled oracle mechanical `SkeletonProposalIR` passed through the real `Compiler.qualify_skeleton_v2`; source-rig hidden identity was not a consumer input and canonical joint IDs remained Compiler-owned.

No predicted IRIS evidence was used. U1 admitted only exact observation-oracle evidence and deterministic product geometry. No hidden full-surface completion or source mesh entered the U1 consumer.

## Frozen shipping identities

Codec config hash:
`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

Arachne config hash:
`ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`

Every arm/witness independently earned a shipping Codec A0 token before Arachne A1. Codec parameters were then frozen. Raw W and Compiler-qualified W had to independently satisfy the same acceptance bands.

## One-family U0/U1

Workflow run `33778934035`, job `100727539409`, region `eastus2`.

Witness: `branch_blend_4`.

### U0 reference-full surface

Codec A0: `PASS`
- rows `160`;
- pass step `704`;
- stable PASS `3`;
- final raw/qualified p95 `0.0390706882 / 0.0390706770`;
- final raw/qualified deformation ratio `0.0158581547 / 0.0158581492`;
- Compiler correction L1 `5.6889257e-06`.

Arachne A1: `PASS`
- pass step `320`;
- stable PASS `3`;
- final raw/qualified p95 `0.0799687132 / 0.0799687132`;
- final raw/qualified deformation ratio `0.0205726642 / 0.0205726624`;
- Compiler correction L1 `6.0896855e-06`.

### U1 observation-oracle substrate

Codec A0: `PASS`
- rows `159`;
- pass step `608`;
- stable PASS `3`;
- final raw/qualified p95 `0.0189781543 / 0.0189781580`;
- final raw/qualified deformation ratio `0.0073177908 / 0.0073177796`;
- Compiler correction L1 `6.1340397e-06`.

Arachne A1: `PASS`
- pass step `512`;
- stable PASS `3`;
- final raw/qualified p95 `0.0577549525 / 0.0577549562`;
- final raw/qualified deformation ratio `0.0314872973 / 0.0314872898`;
- Compiler correction L1 `6.3367770e-06`.

U1 substrate telemetry:
- visible fraction `159/160 = 0.99375`;
- support views min/mean/max `2 / 3.61006 / 4`;
- DTB-ND1 normals `69/159 = 0.43396`;
- local relations `0`.

## Heterogeneous U1 result

Workflow run `33779330043`, job `100728837097`, region `westcentralus`.

All three preregistered witnesses PASSed independently.

### chain_blend_3

Substrate:
- U1 rows `159`;
- visible fraction `159/160 = 0.99375`;
- normal coverage `144/159 = 0.90566`;
- local relations `0`.

Codec A0:
- pass step `544`;
- stable PASS `3`;
- final raw/qualified p95 `0.0446575247 / 0.0446575247`;
- final raw/qualified deformation `0.0117133679 / 0.0117133763`;
- correction `5.8912701e-06`.

Arachne A1:
- pass step `96`;
- stable PASS `3`;
- final raw/qualified p95 `0.0388693474 / 0.0388693213`;
- final raw/qualified deformation `0.0104802987 / 0.0104802977`;
- correction `5.1749812e-06`.

### branch_blend_4

Substrate:
- U1 rows `159`;
- visible fraction `159/160 = 0.99375`;
- normal coverage `69/159 = 0.43396`;
- local relations `0`.

Codec A0:
- pass step `608`;
- stable PASS `3`;
- final raw/qualified p95 `0.0189781543 / 0.0189781580`;
- final raw/qualified deformation `0.0073177908 / 0.0073177796`;
- correction `6.1340397e-06`.

Arachne A1:
- pass step `512`;
- stable PASS `3`;
- final raw/qualified p95 `0.0577549525 / 0.0577549562`;
- final raw/qualified deformation `0.0314872973 / 0.0314872898`;
- correction `6.3367770e-06`.

### sharp_fork_5

Substrate:
- U1 rows `160`;
- visible fraction `160/160 = 1.0`;
- normal coverage `74/160 = 0.46250`;
- local relations `0`.

Codec A0:
- pass step `1056`;
- stable PASS `3`;
- final raw/qualified p95 `0.0239983387 / 0.0239983350`;
- final raw/qualified deformation `0.0048117340 / 0.0048116902`;
- correction `6.4040387e-06`.

Arachne A1:
- pass step `736`;
- stable PASS `3`;
- final raw/qualified p95 `0.0882214159 / 0.0882214084`;
- final raw/qualified deformation `0.0148908431 / 0.0148908459`;
- correction `6.5687720e-06`.

## Causal verdict

`U0_REFERENCE_FULL_SURFACE_ARACHNE = PASS`.

`U1_OBSERVATION_ORACLE_ARACHNE_ONE_FAMILY = PASS`.

`U1_OBSERVATION_ORACLE_ARACHNE_HETEROGENEOUS_SMALL_PANEL = 3/3 PASS`.

`SHIPPING_CODEC_CAPACITY_ON_U1_SHAPES = 3/3 PASS`.

`COMPILER_RESCUE_EXPLAINS_U1_ARACHNE_PASS = FALSIFIED`; raw W independently passed the same static and deformation bands and Compiler correction remained below `1e-5`.

Therefore the current frozen R6 oracle gate supports that the observation-oracle substrate is sufficient for the shipping Arachne+Codec consumer on the admitted small synthetic panel, before IRIS prediction error.

`U2_PREDICTED_IRIS_SUBSTRATE_SUFFICIENCY = UNKNOWN` because U2 was not run.

## Scope limitation: hidden-surface stress

The stronger claim

`COMPLETE_HIDDEN/FULL_SURFACE_IS_UNNECESSARY_UNDER_MATERIAL_SELF_OCCLUSION`

is **not established** by this panel. The current capsule shell is almost fully recoverable across eight horizontal orthographic views: visible fractions are `0.99375`, `0.99375`, and `1.0`.

This is an evidence-strength limitation, not a failure of the closed R6 oracle consumer gate. A separate preregistered self-occlusion/coverage stress may strengthen the claim later without changing or invalidating this PASS.

## New downstream seam exposed

All three U1 surfaces in the heterogeneous gate carried `0` admitted `local_relations`. Geppetto and Arachne did not require those relations to pass their current consumer gates.

However current MWB2 explicitly requires an observed safe local-relation complex to construct direction-local faces. Therefore substrate-to-MWB local relation sufficiency is now an explicit downstream behavioral obligation; it must not be hidden by inventing source-mesh topology or family-specific connectivity.

## Authorization

R6 Arachne observation-oracle consumer gate is `PASS / CLOSED` at the current frozen oracle scope.

Next authorized hardening seam: MWB2 direction-local mesh/skin behavioral fidelity, beginning with the observation-derived local-relation complex requirement.
