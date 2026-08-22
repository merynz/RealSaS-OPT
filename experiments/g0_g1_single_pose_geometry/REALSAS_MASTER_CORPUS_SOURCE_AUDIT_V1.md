# RealSaS Master Corpus — Source & Legal Audit V1

Status: `CANONICAL_SOURCE_REGISTRY_V1`
Date: 2026-08-22
Branch: `g0-g1/single-pose-geometry`

## Purpose

This audit defines which public/third-party rigging datasets or asset collections may feed the RealSaS Master Corpus for IRIS, Geppetto and Arachne.

A downloadable/public dataset is **not** automatically training-authorized. Admission requires two independent gates:

```text
TECHNICAL_ADMISSION == PASS
AND
LEGAL_ADMISSION == PASS
```

Legal admission is fail-closed. A dataset-level software/repository license does not override the license of underlying 3D assets.

This document is an engineering compliance policy, not legal advice. Ambiguous cases are quarantined rather than interpreted aggressively.

## Automatic commercial-training policy

Auto-allow source-asset licenses:

- CC0 / public domain dedication;
- CC-BY variants that explicitly permit commercial reuse, with attribution provenance retained.

Quarantine for explicit legal review:

- CC-BY-SA and other share-alike/custom licenses;
- dataset annotations whose own redistribution/training license is not explicit;
- source records whose object-level license cannot be resolved;
- source-specific terms that may add restrictions.

Auto-reject from commercial/product training:

- any NonCommercial license;
- any NoDerivatives restriction where the intended data transformation/training use is not clearly authorized;
- explicit ML/AI training prohibition;
- research-only / academic-only terms;
- all-rights-reserved or unknown-rights records;
- sources requiring rights-holder permission that has not been obtained.

## Core-scale sources

| Source | Scale / truth | IRIS | Geppetto | Arachne | Legal status | Master action |
|---|---|---:|---:|---:|---|---|
| **Objaverse / Objaverse-XL animated originals** | Tens/hundreds of thousands of animated/armature assets; original mesh, armature, vertex groups/skin, animation when present | HIGH | HIGH | HIGH | `PASS_PER_ASSET` | Primary raw artist-rig source. Resolve each object license from Objaverse metadata. Only explicit commercial-permissive records are admitted. Extract truth directly from original assets rather than relying on a derived rigging dataset. |
| **Articulation-XL2.0** | ~48K+ articulated assets; vertices, faces, normals, joints, bones, root, sparse skinning, point cloud | HIGH | HIGH | HIGH | `PASS_CONDITIONAL_PER_ASSET` | Retain dataset provenance and additionally resolve original Objaverse-XL object license by source/fileIdentifier. Historical RealSaS 3456 is a known-good technical anchor, not the new upper bound. |
| **Rig-XL / UniRig** | ~14K rigged assets; vertices/faces/normals, joints, parents, names, matrix_local, dense skin | HIGH | HIGH | HIGH | `PASS_CONDITIONAL_PER_ASSET` | RIGXL269 remains regression anchor. Expand over full RigXL pool. Objaverse-linked records are resolved object-by-object. VRoid records are quarantined by default unless explicit per-model rights are proven. |
| **Anymate** | 230,716 rigged Objaverse-XL assets; joint/connectivity/bone/skin point supervision; test mesh truth | MED | HIGH | HIGH | `LEGAL_REVIEW` | Do not import processed Anymate annotations into commercial master until an explicit dataset-data license is established. Its existence motivates direct extraction from the original Objaverse assets instead. If a future licensed ID/provenance mapping is released, it may become an annotation variant. |

## Artist / legal-gold sources

| Source | Truth value | Legal status | Master action |
|---|---|---|---|
| **MakeHuman / MPFB** | Parametric humanoid geometry + standard rigs/weights; systematic morphology variation | `PASS_CC0` for core/exported assets | Add as deterministic synthetic legal-gold stratum. Preserve generator version, parameters and rig preset. Do not mix optional non-CC0 third-party assets without a separate license gate. |
| **Blender Studio open character rigs** | Production artist rigs, deform bones, vertex groups, corrective shape keys/control setups, real topology | `PASS_CC_BY_PER_ASSET` | Add curated free CC-BY characters. Retain exact credit/license. Treat deform skeleton/weights as teacher truth; preserve control rig and correctives as additional master metadata rather than silently mapping them to deformation bones. |
| **Quaternius rigged/animated packs** | Dozens of stylized humanoid/non-humanoid game-ready assets with rigs and animations | `PASS_CC0` | Add CC0 packs as stylized artist-gold. Prefer source/Blend or glTF/FBX files retaining armature and weights. |
| **KayKit character packs** | Stylized low-poly rigged/animated characters and animation sets | `PASS_CC0` | Add as stylized artist-gold. GitHub-hosted CC0 packs are preferred for reproducible acquisition. |
| **Khronos glTF Sample Assets** | Small high-quality skin/IBMs/animation conformance cases | `PASS_PER_ASSET` | QA/parser/render regression only. Each selected model must have an explicit compatible license in its asset README. Do not treat as scale training data. |
| **OpenGameArt individual CC0 rigged assets** | Potentially useful long-tail artist rigs | `MANUAL_REVIEW_ONLY` | Not auto-crawled. User uploads/provenance vary. Individual assets may be admitted only with exact page snapshot/license and file hash. |

## Rejected or quarantined sources

| Source | Reason | Decision |
|---|---|---|
| **HumanRig** | Dataset card is CC-BY-NC-4.0 | `REJECT_COMMERCIAL_TRAINING` |
| **RigNet / ModelsResource-RigNetv1** | Underlying Models Resource asset terms do not provide blanket commercial reuse rights | `REJECT_COMMERCIAL_TRAINING` |
| **TARig** | Derived from ModelResource/RigNet asset pool; no clean commercial source-rights chain | `REJECT_COMMERCIAL_TRAINING` |
| **Mixamo** | Adobe terms explicitly prohibit using Mixamo service/content/output to train/test/improve ML/AI systems | `REJECT_AI_TRAINING_BAN` |
| **RigAnything dataset/checkpoints as training source** | Adobe research license is noncommercial/research-only | `REJECT_PRODUCT_TRAINING` |
| **OmniRig / ARMO dataset** | Very large and interesting, but dataset release/license/source-rights chain is not sufficiently available/clear for product use at this audit date | `QUARANTINE` |
| **SkinTokens / TokenRig training data** | Composite training data is not released as a clean independently licensed corpus; includes VRoid/ModelsResource lineage | `NOT_A_SOURCE` |
| **Objaverse-SK** | Skeleton is autonomously derived rather than artist rig truth; no skinning advantage over extracting originals | `DERIVED_REFERENCE_ONLY` |
| **Objaverse-TMS** | Useful skeleton/text derivative, but does not improve the master over the legally resolved original artist-rig source for our three-model target | `DERIVED_REFERENCE_ONLY` |
| **TextuRig** | Curated from RigXL/Objaverse-XL primarily for texture-aware skeleton-to-mesh generation; no need to import its derived annotations when originals are already admitted | `DERIVED_REFERENCE_ONLY` |
| **MB-Lab 3D outputs** | Default generated 3D model licensing inherits AGPL-family constraints; unsuitable for our clean commercial corpus policy | `REJECT_COMMERCIAL_MASTER` |
| **SMPL/SMPL-X and many scan/body datasets** | Common research/custom license restrictions; not a clean general-purpose commercial rigging corpus | `QUARANTINE_UNLESS_SEPARATE_COMMERCIAL_LICENSE` |
| **Planet Zoo / proprietary game-derived data** | No blanket right to reuse game assets as product-training corpus | `REJECT` |

## Preferred master-source hierarchy

The source hierarchy is therefore:

```text
TIER A — SCALE + SOURCE-LEVEL LEGAL RESOLUTION
  Objaverse/Objaverse-XL animated originals
  + Articulation-XL2.0 annotation variants
  + RigXL annotation variants

TIER B — CLEAN ARTIST / SYNTHETIC GOLD
  MakeHuman/MPFB CC0
  Blender Studio CC-BY rigs
  Quaternius CC0
  KayKit CC0

TIER C — QA ONLY
  Khronos glTF skinned sample assets

QUARANTINE / REJECT
  Anymate processed data until explicit dataset-data license
  HumanRig, RigNet, TARig, Mixamo, RigAnything, OmniRig, etc.
```

## Why direct Objaverse extraction is important

Large rigging datasets such as Anymate, Articulation-XL2.0 and RigXL overlap with Objaverse/Objaverse-XL. The master therefore treats the **original 3D asset** as L0 source authority whenever possible.

For an admitted original asset, Blender/headless extraction records:

- complete mesh topology;
- world/canonical vertex positions and normals;
- UV/material-slot metadata when available;
- armature(s), bone names and hierarchy;
- deform-bone membership;
- rest/bind/local transforms;
- exact vertex-group skinning weights;
- shape-key inventory;
- action/animation inventory;
- source file hash and object-level license provenance.

External rigging datasets are linked as additional annotation variants instead of silently replacing the original artist rig.

## Cross-source identity and leakage policy

One physical source asset can appear in multiple datasets. Splitting is performed on `canonical_asset_id`, never independently per annotation source. Therefore a RigXL record and an Articulation/Objaverse record that resolve to the same original asset cannot land in different train/test partitions.

## Render admission

A record is renderable only when all applicable gates pass:

```text
LEGAL_PASS
∧ SOURCE_IDENTITY_RESOLVED
∧ GEOMETRY_VALID
∧ RIG_VALID (when rig truth is claimed)
∧ SKIN_VALID (when skin truth is claimed)
∧ CANONICALIZATION_VALID
∧ NO_SPLIT_LEAK
```

Records that contain useful geometry but fail rig/skin gates may be retained in L0 provenance but are not exported as Geppetto/Arachne teacher records.

## Legal provenance retained per asset

Every admitted asset must retain:

- dataset/source name;
- original provider and source URL/fileIdentifier;
- object-level license string;
- license evidence URL and retrieval timestamp;
- attribution text/author when required;
- source SHA-256;
- legal classifier version;
- legal decision and reason;
- any manual review record.

No model export may contain an asset with `legal_decision != PASS`.

## Final policy

The RealSaS commercial training corpus is the **union of all technically useful, legally resolved, commercially reusable assets**, not the union of all downloadable datasets.

A later legal review may promote quarantined sources without changing the identity or provenance of the V1 admitted corpus.
