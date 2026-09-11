# RealSaS — Mage Full-Subject Reopening + Attachment Contract — 2026-09-12

**Status:** `FIT1_REOPENED__UPSTREAM_SUBJECT_SCOPE_CONTRADICTION__ATTACHMENT_SEMANTICS_REQUIRED`

## Decision

The prior Mage FIT1 product lineage is reopened from the IRIS/H1 signed-geometry authority forward. Historical experiments and architectural discoveries remain preserved, but the promoted Mage product lineage that depends on signed zero-surface SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b` is not sufficient to authorize end-to-end product work.

The contradiction is concrete: the 8 rendered RGBA observations contain a larger subject set than the geometry subset used to form the original H1/Mage geometry authority. Current downstream artifacts therefore describe an internally consistent but incomplete mechanical substrate.

## Measured blast radius

Using the exact 8 orthographic camera contract:

- full deformation-supported Mage source geometry reaches approximately `99.46% .. 99.68%` source-alpha recall per view;
- the currently promoted signed zero-surface reaches approximately `62.24% .. 67.01%` source-alpha recall;
- the current relation-clique MWB2 path reaches approximately `18.22% .. 32.40%` source-alpha recall.

Therefore the current low render coverage is a two-stage loss:

`FULL SUBJECT (~99.5%) -> INCOMPLETE H1/IRIS ZERO SURFACE (~64%) -> CONSERVATIVE MWB2 (~26%)`.

Camera/raster agreement is not the primary fault: full source geometry reprojects to the same observations at near-complete silhouette coverage.

## Full-subject Mage source authority

The full normalized Mage source contains `5321` vertices / `5763` faces. The deformation-supported subset contains `5279` vertices / `5683` eligible faces. The remaining `42` zero-skin vertices / `80` faces form the non-deformation Icosphere provenance object and are excluded from deformation authority.

Known component-level kinematic truth on the full source:

- `Spellbook` -> `handslot.l = 1.0` on all 399 vertices;
- `Spellbook_open` -> `handslot.l = 1.0` on all 418 vertices;
- `1H_Wand` -> `handslot.r = 1.0` on all 158 vertices;
- `2H_Staff` -> `handslot.r = 1.0` on all 498 vertices;
- `Mage_Hat` -> `head = 1.0`;
- `Mage_Cape` -> `chest = 1.0` in the current source truth.

These are not disposable geometry. They are visible product components with explicit kinematic authority.

## Rigid attachment semantic gap

The present chain can represent a rigidly moving component indirectly as skin weights concentrated on one canonical joint. That is kinematically sufficient for motion, but it is not product-level assembly semantics.

`QualifiedSkeletonIR.v2` contains an optional `assembly_root_binding`, but current Mage qualification leaves it empty and there is no dedicated typed attachment set that answers:

- which surface/render component is an attachment;
- which canonical joint/socket owns it;
- whether its deformation mode is rigid or skinned;
- whether it is detachable/swappable;
- its local bind transform / directional render identity;
- provenance and qualification evidence.

Mage is now the required closure witness for this gap.

### Required product semantics

At minimum the Compiler must be able to qualify a component record equivalent to:

```text
component_id
component_surface_ids / render_component_ids
attachment_kind = RIGID_BONE_ATTACHMENT | RIGID_SKINNED_COMPONENT | DEFORMABLE_COMPONENT
canonical_joint_id / socket_id
bind_transform_or_directional_binding
detachable
source_component_identity
qualification_report
component_lineage_hash
```

Weights alone must not silently create detachable/socket semantics. The Compiler owns the final typed classification.

For Mage, the books and wand/staff are mandatory rigid-attachment witnesses. Hat/head-bound and cape/chest-bound behavior must be classified from exact source/component evidence rather than guessed from semantic names.

## Reclosure order — frozen

No ARAP, BBW, XPBD, appearance completion, motion promotion, or PRODUCT_PASS work is authorized until the following sequence closes:

1. repair the Mage H1/IRIS training/evaluation subject contract so the geometry target and the 8 rendered observations cover the same admitted product subject;
2. run corrected Mage IRIS/H1 FIT and require high 8-view source silhouette coverage before promotion;
3. re-emit deterministic GSA from the corrected zero-surface and audit node/support/topology/raster coverage;
4. run existing Geppetto checkpoint frozen on the corrected S first; refit only if the frozen gate fails;
5. qualify corrected skeleton plus explicit Mage component/attachment semantics;
6. run existing Arachne/V5 checkpoint frozen on corrected S+G first; reopen only the minimum FIT training stage if it fails;
7. only then restore/evaluate historical CDT against the corrected substrate;
8. require 8-view mesh coverage and typed unknown/attachment/topology gates before any later solver is considered.

## Claim boundary

Until this reclosure completes:

- `MAGE_FIT1_RIGGING_CORE = REOPENED`;
- prior IRIS/Geppetto/Arachne checkpoints remain preserved scientific artifacts, not deleted;
- prior exact W remains authoritative only for the old S lineage;
- `V5_FAMILY_DISJOINT_UNSEEN_GENERALIZATION_GATE = BLOCKED`;
- `PRODUCT_PASS = NOT CLAIMED`.

The objective is not to weaken a gate to retain prior promotion. The objective is to rebuild the Mage chain from the earliest contradicted authority and close every product-relevant gap exposed by Mage before moving to unseen families.
