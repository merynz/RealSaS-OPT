# RealSaS — Geppetto/Arachne External-Reference Clean-Room Audit Prereg — 2026-08-31

**Status:** `PREREG_SEALED__REFERENCE_AUDIT_REQUIRED_BEFORE_LEARNED_APPARATUS_SEAL`

## Purpose

RealSaS has deliberately separated Geppetto (skeleton proposal) and Arachne (skin proposal) because external 3D auto-rigging systems provide evidence that these are solvable problems when sufficiently informative geometric input is available. That premise must now be re-audited at code level before the final learned Geppetto/Arachne architecture, loss, optimizer, data encoding or training protocol is sealed.

The audit is not a request to copy an external implementation. Its purpose is to determine exactly:

1. what information the successful reference system actually receives;
2. what representation and causal factorization it uses;
3. what is learned versus analytic/postprocessed;
4. what losses/training signals are essential versus incidental;
5. whether the current RealSaS `RiggingSurfaceIR` / `QualifiedSkeletonIR` contracts provide an information-equivalent or stronger substrate;
6. which mechanisms are therefore justified as RealSaS design hypotheses.

## Frozen upstream reference snapshots

### RigAnything

Repository: `Isabella98Liu/RigAnything`  
Frozen upstream tree/commit for this audit: `d03cdb21dd134fa81df6b0947522469db3f78bd2`.

Observed public-code facts at prereg time include:

- point-cloud + normal input path (`6` channels per point);
- point tokenizer -> autoregressive transformer joint sequence;
- diffusion position decoder for each generated joint;
- explicit parent-candidate scoring;
- point/joint feature pairing for skinning prediction;
- configured public scaffold: 1024 point tokens, up to 64 joints, transformer width 1024, 12 layers, diffusion sampling 300 steps.

**License firewall:** the public repository is under the Adobe Research License and grants use only for noncommercial research. The public RigAnything implementation must not be imported into, copied into, translated line-for-line into, or used as a code dependency of commercial RealSaS product code. The clean-room audit may record facts, algorithms, tensor contracts, equations and independently stated design requirements. This prereg is not legal advice and does not resolve possible patent or other IP questions.

### SkinTokens / TokenRig

Repository: `VAST-AI-Research/SkinTokens`  
Frozen upstream tree/commit for this audit: `273b691d35989d71cd17ff2895fdc735097b92d1`.

Observed public-code facts at prereg time include:

- learned discrete skin representation via an FSQ-conditioned VAE/CVAE family;
- skin encoding conditioned on sampled geometry (positions/normals) and skin fields;
- compact latent/token bottleneck followed by geometry-conditioned decoding;
- TokenRig unified autoregressive skeleton + skin-token generation in the released system;
- the public SkinTokens repository is MIT licensed.

MIT permission does not remove the scientific requirement to distinguish reference facts from RealSaS-specific hypotheses. RealSaS may still choose an independent implementation and representation.

## Current RealSaS consumer prototypes — terminology correction

Two consumer prototypes already exist and were necessary for the DINO/DTB consumer chain:

- `experiments/consumer_interlock_20260829/sacrificial_consumers_v0.py::build_g0_proposal` — deterministic sacrificial G0 based on interior geometry/PCA;
- `experiments/consumer_interlock_20260829/sacrificial_consumers_v0.py::build_a0_proposal` — deterministic sacrificial A0 based on distance-softmax skinning.

These prototypes established route/coupling validity through the exact Compiler. They are not claims about product-quality learned Geppetto/Arachne architecture.

A separate Geppetto G0.1 learned scaffold exists on the historical development branch and passed scaffold regressions, but was explicitly marked `FREEZE_CANDIDATE_NOT_SEALED`. Its old `max_controls=64` configuration is superseded as a candidate by the later structural corpus audit, which froze C0 capacity `160` with `ABSTAIN_NO_TRUNCATION` overflow semantics.

No current sacrificial or scaffold implementation may become the final product model merely because it already executes.

## Core scientific hypotheses

### H-G — Geppetto solvability under information-equivalent substrate

If `RiggingSurfaceIR` supplies an information set equivalent to or stronger than the geometric information actually consumed by successful template-free skeleton generators, then the remaining skeleton proposal problem should be learnable without recovering hidden source-rig identity.

This hypothesis is **not** considered established merely because RigAnything succeeds on 3D meshes/point clouds. Input equivalence must be demonstrated.

### H-A — Arachne solvability under information-equivalent substrate

If exact/qualified surface geometry plus `QualifiedSkeletonIR` supplies an information set equivalent to or stronger than the geometric/skeletal conditioning consumed by successful skin-field systems, then the remaining skin proposal problem should be learnable without hidden authored helper identity.

This hypothesis is **not** considered established merely because SkinTokens/TokenRig succeeds on 3D meshes. Input equivalence and codec/representation ceiling must be demonstrated.

## Clean-room audit protocol

### R0 — Source and license inventory

For each reference, record exact upstream commit/tree, files inspected, public paper/version, released checkpoints/configs, license and missing implementation pieces. Missing training code or missing data semantics must be marked `UNKNOWN`, never reconstructed from guesswork.

### R1 — Exact information contract

Code-level extraction of all product-inference inputs and preprocessing:

- geometric primitives;
- point/vertex count and sampling;
- normals/tangents/topology if used;
- normalization/scale/centering;
- skeleton information visible at each stage;
- ordering/canonicalization assumptions;
- hidden teacher-only fields;
- postprocessing dependencies.

Produce a field-by-field `REFERENCE_INPUT -> REALSAS_AVAILABLE_EQUIVALENT` matrix with statuses:

- `EXACT_EQUIVALENT`
- `STRICTLY_STRONGER_REALSA S`
- `APPROXIMATE_EQUIVALENT`
- `MISSING`
- `NOT_REQUIRED_AT_INFERENCE`
- `UNKNOWN_FROM_PUBLIC_RELEASE`

A typo-free machine-readable version must use `STRICTLY_STRONGER_REALSAS`.

### R2 — Representation / architecture decomposition

For each reference, derive an implementation-independent graph:

`input -> geometry encoder -> latent/sequence representation -> proposal/field decoder -> postprocess/output`.

Record tensor shapes, attention/graph locality, autoregressive ordering, diffusion/quantization/codec choices, stopping/count mechanism, parent representation, skin representation and uncertainty if any.

Direct source-code transcription into RealSaS is forbidden. The clean-room artifact contains only independently worded functional specifications, equations, tensor contracts and observed behavior.

### R3 — Loss and training audit

Extract, when public:

- exact supervised targets;
- matching/assignment rules;
- joint-position objective;
- topology/parent objective;
- skin-field objective;
- codec reconstruction objective;
- regularization, sparsity, deformation or geometry losses;
- optimizer/LR/schedule;
- augmentation/sampling;
- curriculum/teacher forcing;
- inference/training mismatch.

Anything absent from public code is recorded as `NOT_PUBLICLY_VERIFIABLE` rather than inferred.

### R4 — Inference and postprocessing audit

Determine which apparent model capability actually comes from deterministic postprocessing, constraints, rendering, mesh sampling, nearest-neighbor transfer, skeleton serialization, normalization, or DCC export logic.

This prevents assigning to the neural model a capability supplied elsewhere.

### R5 — RealSaS information-equivalence proof

Before final learned seals, use exact teacher/native authority to create oracle RealSaS inputs and ask:

1. Can the reference-required geometry fields be deterministically derived from the current RealSaS exact substrate without source-rig identity leakage?
2. Which fields are missing because IRIS is partial/observed rather than watertight/full-mesh?
3. Are those missing fields genuinely required by the reference implementation or merely available to it?
4. Can the Geppetto/Arachne task be solved on exact RealSaS oracle substrate before predicted IRIS geometry is introduced?

A clean information-equivalence claim requires field-level evidence, not semantic similarity such as “both are point clouds.”

### R6 — Oracle-substrate ceiling gates

Before blaming IRIS or generalization:

- Geppetto must pass one-family and small heterogeneous ceiling tests using exact/oracle `RiggingSurfaceIR` generated from authoritative geometry;
- Arachne must first pass a lossless-or-bounded SkinFieldCodec reconstruction ceiling (if a codec is used), then one-family and small heterogeneous tests using exact/oracle surface + exact/qualified skeleton inputs;
- exact Compiler qualification and deformation proof remain the final evaluator route.

Failure on oracle substrate is an apparatus/representation failure and blocks full training.

### R7 — Independent RealSaS candidate design

Only after R0–R6 close may final candidate architectures be written and sealed.

The design document must separate every mechanism into one of:

- `REFERENCE_SUPPORTED`
- `REALSAS_CONTRACT_REQUIRED`
- `CORPUS_MEASURED`
- `NEW_HYPOTHESIS`
- `LEGACY_SCAFFOLD_ONLY`

No mechanism becomes binding merely because an external paper used it.

## Seal semantics

For Geppetto/Arachne, **seal means**:

> the pre-output contract cannot be changed after scientific output is opened unless a separately recorded causal failure or contract contradiction justifies reopening it.

A seal is not a claim of eternal architectural optimality. It is a change-control boundary against outcome-driven tuning.

## Binding gates before learned-model seal

Final `GEPPETTO_C0_LEARNED_APPARATUS_PREREG` and `ARACHNE_C0_SKINFIELD_APPARATUS_PREREG` may not be promoted to training authority until all are true:

1. external-reference R0–R4 code audit complete;
2. R5 input-equivalence matrix complete with all material `MISSING/UNKNOWN` fields resolved or explicitly accommodated;
3. clean-C0 semantic membership closed;
4. Geppetto structural policy remains bound to max-controls 160 / no truncation unless a causal reopening is preregistered;
5. Arachne helper/non-deform target semantics remain no-transport/no-renormalization unless a causal reopening is preregistered;
6. oracle-substrate ceiling experiment designs are sealed before output;
7. source/config/runtime/loss/evaluator hashes are frozen before optimizer step 1.

## Claim firewall

Permitted after this prereg only:

- RigAnything and SkinTokens provide strong external evidence that skeleton/skin tasks have successful solution families under rich 3D geometric input.
- RealSaS has route-valid sacrificial G0/A0 consumers.
- RealSaS has not yet proven that its partial observable substrate is information-equivalent to every field used by those systems.

Forbidden until audit closure:

- “Geppetto is solved because RigAnything solved it.”
- “Arachne is solved because SkinTokens solved it.”
- “Our input is equivalent because both contain 3D points.”
- copying/reusing RigAnything product code in RealSaS.
- freezing final Geppetto/Arachne architecture from paper-level analogy alone.

## Immediate next work

Run the code-level clean-room audit against the frozen upstream snapshots, produce:

1. `RIGANYTHING_CODE_LEVEL_REFERENCE_AUDIT_V1.md/json`;
2. `SKINTOKENS_CODE_LEVEL_REFERENCE_AUDIT_V1.md/json`;
3. `GEPPETTO_ARACHNE_REFERENCE_TO_REALSAS_INFORMATION_EQUIVALENCE_V1.md/json`;
4. only then draft the final learned apparatus preregistrations.

Scientific training authority remains unchanged by this document.
