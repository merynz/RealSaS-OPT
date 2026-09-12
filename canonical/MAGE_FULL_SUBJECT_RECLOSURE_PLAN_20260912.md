# RealSaS — Mage full-subject reclosure plan — 2026-09-12

**Status:** `ACTIVE__FAIL_CLOSED__MAGE_FIT1_REOPENED`

## Purpose

Mage is not treated as a special-case product profile. It is used as a deliberately difficult capability witness because one asset contains a deformable body plus rigid / quasi-rigid accessory components (book, wand/staff, hat, cape) under the same eight-view product observation contract. The goal is to close generic RealSaS contracts against this witness, not to hard-code Mage semantics.

## Discovered contradiction

The promoted Mage IRIS/GSA chain was built from a signed zero-surface whose source-side geometry authority did not cover the full rendered subject. The historical render/training path selected armature-modified body geometry to form geometry truth / normalization bounds while the rendered RGBA contained additional components. This produced a real mismatch between observation subject and learned geometry subject.

Measured on the exact eight Mage observations and exact cameras:

- full deformation-supported source geometry -> source alpha recall: ~99.46% to ~99.68%;
- promoted signed zero-surface -> source alpha recall: ~62.24% to ~67.01%;
- current conservative MWB2 relation-clique mesh -> source alpha recall: ~18.22% to ~32.40%.

Therefore downstream product closure is reopened at the first contradicted authority: full-subject IRIS signed geometry.

## Claim boundary

The following are **not** revoked as research history:

- architecture discovery and falsifications;
- Geppetto/Arachne architecture choices;
- quotient / graph results;
- Compiler contracts unrelated to the contradicted subject boundary;
- historical checkpoints and experiment evidence.

The following **are not currently product-authoritative for Mage** until reclosed on corrected full-subject S:

- promoted Mage signed zero-surface and its GSA lineage;
- Mage QualifiedSkeletonIR derived from that S;
- Mage Arachne/V5 QualifiedSkinIR derived from that S;
- downstream MWB2/appearance/binding/product artifacts bound to those lineages.

## Mandatory execution order

No later stage may be promoted before every earlier gate below has passed.

1. **Observation authority audit**
   - exact 8 RGBA observations;
   - exact 8 orthographic camera records;
   - per-view hashes;
   - projection/raster replay;
   - full rendered-subject component inventory.

2. **Generic component / attachment truth audit**
   - classify every rendered source component by geometry membership and mechanical mode;
   - at minimum distinguish `DEFORMABLE_COMPONENT`, `RIGID_SKINNED_COMPONENT`, `RIGID_BONE_ATTACHMENT`, and explicitly unsupported/no-mechanical-authority geometry;
   - preserve source component identity and provenance;
   - detachable/swappable is a separate property and must not be inferred merely from rigidity.

3. **Corrected full-subject IRIS geometry FIT**
   - product inference input remains only 8 RGBA + exact cameras;
   - teacher/source mesh may be used only for FIT training/evaluation authority;
   - no component visible in admitted product observations may silently disappear from the target because of source-object selection;
   - report silhouette recall / precision / IoU per view against the product observations;
   - report component-level coverage, not only whole-character aggregate coverage.

4. **Dense signed zero-surface gate**
   - validate predicted-to-truth precision;
   - validate observed silhouette completeness on all eight views;
   - validate camera/reprojection consistency;
   - validate component coverage for body + attachment witnesses;
   - fail closed on missing major observed component.

5. **GSA re-emission**
   - deterministic compaction from corrected dense zero-surface;
   - no learned retraining;
   - retain full provenance and exact camera bindings;
   - distinguish self-visible predicted surface support from actual observation support in metadata/flags;
   - emit a new surface lineage. Old 950-node lineage may not be reused.

6. **Frozen Geppetto replay first**
   - evaluate the currently promoted Geppetto checkpoint on corrected S before authorizing any retraining;
   - if all current skeleton gates pass, requalify only and retain architecture/checkpoint;
   - retraining is authorized only if frozen replay fails a preregistered gate.

7. **Typed component / attachment qualification**
   - component identity must survive into product IR;
   - rigid joint-bound geometry must not be represented only by anonymous skin weights;
   - Compiler must qualify joint/socket binding, component mode, provenance, and detachability state;
   - assembly semantics must be generic and may not contain Mage-specific names or rules.

8. **Frozen Arachne replay first**
   - old 950x22 W is invalid on new S identities and cannot be copied;
   - rebuild full-source teacher projection / evaluation authority for new S;
   - evaluate promoted Arachne checkpoint on corrected S+G before any retraining;
   - retrain only if frozen replay fails unchanged scientific gates.

9. **CDT / MWB2 only after corrected S/G/W**
   - restore historical domain/contour/local-support/CDT numerical capability behind current typed IR;
   - solver has no semantic authority and may not invent hidden structure;
   - measure coverage of corrected S domain and source observation alpha separately;
   - current relation-clique implementation remains diagnostic baseline, not product authority.

10. **Appearance + directional binding + Living Compile**
    - exact 8-view source appearance;
    - typed component membership preserved;
    - exact directional joint/component bindings;
    - static product visualization before motion/runtime;
    - PRODUCT_PASS remains separately proof-gated.

## Compute policy

Prefer deterministic CPU / local self-hosted execution for audits, contract tests, GSA, projection/raster replay, component qualification, mesh/CDT, and serialization. Use the local 6 GB GPU only where model inference fits safely. Escalate to Colab/A100 only when an actual model-memory or training requirement is demonstrated. Do not request large-GPU work preemptively.

## Hard anti-shortcut rules

- No Mage-specific code path.
- No teacher/source geometry at product inference.
- No reuse of old S/G/W lineages after corrected full-subject S is emitted.
- No downstream solver may mask upstream missing geometry.
- No aggregate metric may hide a missing component; component-level coverage is mandatory.
- No retraining before frozen-checkpoint replay proves it necessary.
- No threshold widening in response to failure without a separately justified contract change.
- No PRODUCT_PASS / unseen/generalization claim from this reclosure.
