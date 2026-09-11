# RealSaS-OPT — Current State

**Date:** 2026-09-11  
**Canonical continuation branch:** `main`  
**Status:** `MAGE_FIT1_RIGGING_CORE_CLOSED__IRIS_GSA_GEPPETTO_ARACHNE_V5_PROMOTED__GENERALIZATION_NOT_CLAIMED__PRODUCT_PASS_OPEN`

This file is the continuation authority on `main`.

**Investor/auditor evidence index:** `canonical/FIT1_EVIDENCE_INDEX_20260909.md`  
**Most recent promoted learned closure:** `ARACHNE_A1_V5_MINIMAL_K4_DIRECT_SIMPLEX_FIT1`  
**Most recent promotion commit:** `03d9f87dbb7100a72293915cf682cbf338335a37`  
**Current scientific module:** `Arachne V5 / generalization preparation`  
**Active experiment gate:** `NONE`  
**Next scientific gate when work resumes:** `V5_FAMILY_DISJOINT_UNSEEN_GENERALIZATION_GATE`  
**Mage FIT1 rigging core:** `CLOSED`  
**Unseen-family generalization:** `NOT CLAIMED`  
**Product PASS:** `NOT CLAIMED`

## One-line state

`The real Mage FIT1 rigging core now closes from raster observations through IRIS/GSA, Geppetto + Compiler QualifiedSkeletonIR, and Arachne V5 + Compiler QualifiedSkinIR. Arachne V5 is FIT1-frozen/promoted with GSA p95 0.04237785 and no A0 runtime dependency. Historical A0/A1/diagnostic evidence remains preserved. The next scientific step is family-disjoint unseen/FIT8-LOFO generalization, but no unseen-family or end-to-end PRODUCT_PASS claim is made.`

## Mandatory rehydration order

1. `canonical/REHYDRATION_PACKET.md`
2. `canonical/FIT1_EVIDENCE_INDEX_20260909.md`
3. `canonical/MAGE_FIT1_RIGGING_CORE_CLOSURE_20260911.md`
4. `canonical/ARACHNE_A1_V5_MINIMAL_K4_DIRECT_SIMPLEX_FIT1_PROMOTION_20260911.md`
5. `canonical/ARACHNE_A1_V5_FIT1_EVIDENCE_MANIFEST_V1.json`
6. `models/arachne/v5/PROMOTED_MAGE_FIT_WITNESS_V1.json`
7. `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`
8. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
9. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
10. `canonical/EXPERIMENT_REGISTRY_V2.json`

## Product ownership

RealSaS targets an **eight-direction editable 2D/2.5D puppet** compiled from raster artwork. Mechanical 3D/world evidence is an internal representation, not a full-3D product claim.

Current learned/deterministic flow:

`8 raster observations + exact cameras -> IRIS learned evidence -> deterministic GSA/RiggingSurfaceIR -> Geppetto SkeletonProposalIR -> Compiler QualifiedSkeletonIR -> Arachne V5 SkinProposalIR -> Compiler QualifiedSkinIR -> appearance/motion/proof/export/runtime`

The Compiler remains the only owner of canonical IDs, legal skeleton/tree state, skin/simplex legality, qualification and product truth.

## Promoted Mage FIT1 stack

### IRIS / GSA

- IRIS checkpoint SHA-256 `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2`;
- signed zero-surface SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`;
- teacher mesh is not an inference input;
- deterministic GSA owns final `RiggingSurfaceIR` assembly/provenance.

### Geppetto / QualifiedSkeletonIR

Current source home: `models/geppetto/reference_strength_v1/`.

- verdict `FIT1_TERMINAL_PASS`;
- closure step `14080`;
- terminal full-structural streak `48/48` checks / `3072` optimizer steps;
- qualified controls `22`;
- exactly one deform root;
- checkpoint SHA-256 `b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30`;
- QualifiedSkeletonIR SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`.

### Arachne V5 / QualifiedSkinIR

Current source home: `models/arachne/v5/`.

Architecture: `RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5`

Shipping FIT1 route:

`RiggingSurfaceIR + QualifiedSkeletonIR -> exact A1 V4 backbone -> K4×512 Z -> 325,313-param direct row-simplex decoder -> SkinProposalIR -> Compiler QualifiedSkinIR`

Frozen composite checkpoint:

- V4 backbone checkpoint SHA-256 `95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763`, Drive ID `1HJC4GPMcWa0jEXCYgctUtW09tXn4I9ND`;
- V5 decoder delta SHA-256 `13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137`, Drive ID `1_o2XWO5BOSDKGvewJDSGS16jlcMuT9Dx`;
- total parameters `138,378,466`;
- A0 continuous-field model loaded at runtime: `false`.

FIT1 closure:

- GSA row-L1 p95 `0.04237784981177733` (`<= 0.05` PASS);
- deformation ratio `0.019856400787830353` (`<= 0.05` PASS);
- articulated deformation ratio `0.002780771814286709`;
- dominant accuracy `0.9957173447537473`;
- Compiler rows `950/950`;
- Compiler total correction L1 `2.9468642839168442e-05`;
- holdout p95 diagnostic `0.15087631421532924`, preregistered safe;
- teacher predictor firewall `true`.

The final direct-simplex diagnostic showed both H and Z routes pass; Z is retained because it is the minimal decoder-seam intervention. The passing V5 closure reproduces the parent Z diagnostic p95 to ~`1e-7` absolute difference.

## Historical Arachne memory — preserved, not deleted

All prior A0/A1/diagnostic evidence remains valid scoped scientific memory for FIT8/LOFO and future causal debugging, including V7 C3/C4 sampling failures; A0 K4 representation/oracle closure; A1 V4 terminal failure under the frozen A0 decoder; owner-localization/boundary/feature/pair-geometry/support-shaping diagnostics; the 8K direct-simplex historical partial run; and the matched 16K H+Z direct-simplex PASS.

Promotion supersedes these paths for **current Mage FIT1 execution** only. It does not erase or relabel their scientific results.

## Claim boundary

Strongest current claim:

`RealSaS has a hash-bound, inspectable Mage FIT1 rigging-core chain from raster observations to Compiler-qualified skeleton and skin, with separately promoted IRIS, Geppetto and Arachne V5 learned components.`

Not claimed: unseen-character/family generalization; FIT8/LOFO success; appearance/motion/editor/runtime end-to-end PRODUCT_PASS; or commercial production readiness from FIT1 alone.

## Immediate execution order

1. keep the FIT1 promoted source/evidence immutable except through explicit contradiction-driven reopening;
2. preserve all historical experiment branches/artifacts for generalization and failure-memory;
3. demo / investor review may use the current four-panel real-artifact witness;
4. when science resumes, preregister and run `V5_FAMILY_DISJOINT_UNSEEN_GENERALIZATION_GATE` / FIT8-LOFO;
5. reserve `PRODUCT_PASS` for its own independent end-to-end contract.
