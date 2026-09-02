# RealSaS Geppetto / SkinFieldCodec / Arachne — Candidate Architecture V1

**Status:** `SOURCE_CANDIDATE_IMPLEMENTED__OPTIMIZER_NOT_AUTHORIZED`

**Date:** 2026-09-02

## Non-negotiable product ontology

`THREE_D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`

The learned stack consumes observation-grounded mechanical substrate and emits proposal/evidence objects for a directional 2D/2.5D deformable puppet. It does not recover or claim a watertight/full hidden 3D character. Compiler owns canonical IDs, legal topology, legal skinning, qualification, proof, product state and runtime release.

Upstream image authority is unchanged by this candidate stack: clean FIT families use the previously frozen Master/source-textured **8 × 1024 RGBA + exact cameras** observation lane. Geometry-isolation/cel artifacts remain auxiliary/debug evidence and are not promoted here as a new encoder-input authority.

## Causal execution order

1. `R6-G/U0`: reference full-surface diagnostic upper bound.
2. `R6-G/U1`: observation-oracle `RiggingSurfaceIR` -> Geppetto candidate -> actual Compiler skeleton qualifier.
3. `R6-A0`: authoritative dense W -> **SkinFieldCodec encode/decode** -> actual Compiler skin qualifier -> deformation proof.
4. Only after codec representation/deformation ceiling PASS: freeze Arachne predictor execution.
5. `R6-A1/U0,U1`: qualified Compiler skeleton + surface conditioning -> Arachne latent fields -> **same SkinFieldCodec decoder** -> `SkinProposalIR` -> actual Compiler qualifier -> deformation/motion proof.
6. `U2_PREDICTED_IRIS_SUBSTRATE` is blocked until U1 and IRIS qualification.

No optimizer/FIT run is authorized merely by source or unit-test PASS.

## Geppetto V1

Input: admitted `RiggingSurfaceIR` only.

Conditioning is deterministic and sorted by `surface_id`. It carries positions, support-view pattern, support count, qualified local normal, relation summaries and raster-support summaries. Teacher controls/source bone IDs/hidden meshes/canonical product IDs are not accepted by the adapter.

Candidate: global surface Transformer encoder + internal learned set queries + Transformer decoder. Heads emit:

- existence evidence,
- endogenous count,
- abstain evidence,
- continuous joint position evidence,
- root evidence,
- all-pairs directed parent evidence,
- admitted-surface support evidence.

Training uses exact Hungarian matching. Query index is an internal permutation/matching coordinate only. The model emits `SkeletonProposalIR`; it never mints canonical `J:*` identity and never emits learned hard-required/hard-forbidden edges.

## SkinFieldCodec V1 — R6-A0 first

Representation is a compact continuous **per-qualified-joint influence field**, not a fixed source-bone tokenization.

Teacher lane only:

`dense W + surface conditioning + qualified joint conditioning -> per-joint latent field`

Shared decoder:

`latent field + surface conditioning + qualified joint conditioning -> dense N × J simplex W*`

The decoder is the single reconstruction head used later by Arachne. Codec PASS must be established by reconstruction **and deformation-sensitive consequence**, not weight similarity alone.

## Arachne V1 — blocked until codec PASS

Input: admitted `RiggingSurfaceIR` + Compiler-qualified `QualifiedSkeletonIR.v2`.

The adapter serializes canonical joint IDs deterministically only for tensor alignment and carries explicit parent indices; model conditioning contains root/depth/support/topology context. The predictor uses a surface Transformer and joint-to-surface cross-attention to emit one latent field per qualified joint. It calls the exact codec decoder and emits dense `SkinProposalIR` rows over every admitted surface node and every qualified joint.

The model does not sparsify or legalize W. Compiler owns simplex correction policy, sparsification, qualification and final `QualifiedSkinIR`.

## Leakage / authority firewall

Forbidden model inputs or authorities:

- teacher/source bone identity as product identity,
- hidden source mesh/volumetric reconstruction authority,
- source rig canonical IDs,
- direct learned hard skeleton edges,
- model-owned final tree,
- model-owned legal skinning,
- full-3D reconstruction claims.

Teacher projection modules remain training/evaluation-only. Fresh inference must be capable of running without teacher payloads.

## Source gate

Source gate requires:

- deterministic conditioning tests,
- variable N/J tests,
- real PyTorch forward/backward tests,
- Hungarian matching test,
- proposal lineage tests,
- dense skin row/simplex tests,
- topology-sensitive Arachne conditioning test,
- candidate architecture CI PASS.

Passing this gate authorizes **prospective scientific rungs only in the preregistered order above**, not generalization and not a product PASS claim.
