# RealSaS Learned Model Naming V2

**Date:** 2026-10-05  
**Status:** `CANONICAL_NAMING_AUTHORITY`  
**Scope:** public/paper model identities only; checkpoint, tensor, schema and artifact lineage is unchanged.

## Canonical learned stack

| Name | Expansion | Scientific interpretation |
|---|---|---|
| **IRIS** | **Image-based Relational Inference for Structure** | observation-grounded relational structure evidence |
| **TESSA** | **Topological Evidence for Surface Structure Approximation** | learned surface/topology hypothesis over the mechanical substrate |
| **AXIS** | **Articulation eXtraction through Inferred Structure** | learned articulation/skeleton hypothesis |
| **MIRA** | **Mesh-Informed Rigging Affinity** | learned continuous rig-affinity field used to derive skin/attachment proposals |

Canonical paper order:

`IRIS -> TESSA -> AXIS -> MIRA`

Compiler authority surrounds every learned proposal-to-product transition.

## Why these names

The expansions intentionally use paper language rather than literal product-role labels. They describe the scientific object represented by each learner without asserting stronger mathematical properties than the current implementation has proven. In particular, the names do **not** claim equivariance, invariance, optimality, exactness, or canonical truth.

## Lineage and migration

This naming authority does not rewrite historical implementation identifiers.

- **IRIS** keeps its existing name and perception lineage.
- **TESSA** keeps the current mesh/topology research and implementation lineage.
- **AXIS** supersedes the temporary/public rig names **ATLAS** and **Geppetto**. Existing checkpoint/tensor/schema strings that contain those names remain valid historical identifiers until a dedicated semantic-equivalence migration is proven safe.
- **MIRA** keeps the current skin/attachment research name and supersedes **Arachne** as the public/paper identity. Existing Arachne implementation/checkpoint identifiers remain lineage identifiers during compatibility migration.

## Authority rule

A learned-model name never mints canonical mechanical truth.

- IRIS emits perception evidence.
- TESSA emits surface/topology proposals.
- AXIS emits articulation/rig proposals.
- MIRA emits rig-affinity / skin / attachment proposals.
- Compiler qualification, repair, proof and sealing remain authoritative.

## Forbidden ambiguity

After this naming authority lands:

- `ATLAS` must not be presented as a distinct canonical rig model;
- `Geppetto` is a historical implementation/checkpoint lineage name, not the paper identity;
- `Arachne` is a historical implementation/checkpoint lineage name, not the paper identity;
- canonical documentation should say **IRIS / TESSA / AXIS / MIRA** unless it is intentionally discussing historical lineage;
- cosmetic renaming must never rewrite byte-sensitive checkpoint keys, artifact hashes or schema contracts without explicit equivalence proof.
