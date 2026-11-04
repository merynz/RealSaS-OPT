# Corpus370 — generic drawable identity: what a source relation can and cannot prove

**State:** source-hit graphics/occlusion is measured in the exact 3-subject
pilot; drawable partition/owner identity remains `UNKNOWN`. This document
does not claim model, training, graphics-family, TESSA or product authority.

## The operational problem

A puppet's **drawable** is an editable presentation unit, not a source VRM
semantic category, not a Blender mesh object, not a material, not a bone, not
necessarily a connected 3D component and not necessarily a rigid body.
RealSaS must compile sufficient layers/geometry/deformation/draw-order
control to replay the character faithfully across views and requested edits.

The exact opposite-view pilot (`41902211474`, independent visibility court
`41904446622`) proves source-hit compositing at 6 pixels on 3 subjects:
6 directly visible hit contributions, 6 opaque-occluded source hits which
become recolorable when A is removed, 0 visibility ABSTAIN. It is an
**observational source-hit oracle, not drawable partition supervision**.
`qualified_same_drawable_positives=0`,
`qualified_different_drawable_negatives=0`.

## Why neither topology nor authored motion alone solves drawable identity

`topology_partition_court_v1.py` already distinguishes vertex-connected,
surface-edge-connected and manifold-edge-connected faces. Preserve its
label-invariant implementation. A fully welded *skinned* mesh can require a
visible arm/torso drawable split. Two disconnected mesh patches can still be
one editable graphic, and a point-only contact is not a surface weld.

A source-authored animation is an observation of vertex motion, not an
isolated mechanical cause. Two independent articulations can move together
under a common-root control. Conversely, one skinned drawable can bend
internally. Thus:
- unchanged relative distances + face orientations across sampled poses is
  `NO_RELATIVE_DEFORMATION_OBSERVED`, **not SAME_DRAWABLE positive**;
- changed relative distances or face orientations is
  `OBSERVED_RELATIVE_DEFORMATION`, **not DIFFERENT_DRAWABLE negative**;
- shared/nonshared source triangles or authored owner/material names are
  topology or provenance addresses, **never automatic drawable truth**.

Absolute displacement and absolute orientation are misleading for global
rotation. The diagnostic compares the **distance between corresponding
source-barycentric loci** and **relative face frames** under authored poses,
so a shared global translation or rotation is invariant.

## Implemented diagnostic

`tools/corpus/diagnose_corpus370_generic_pair_kinematics_v1.py` accepts
a previously independent-courted source-hit visibility receipt; an original
rest geometry NPZ; an authored pose NPZ; and the **exact authored exporter
receipt**. All parent bytes must match their SHA-256, source SHA,
subject identity, exact face ordering, and rest geometry within a declared
relative tolerance. The output keeps source face IDs **teacher-only** and
reports separate:
- mesh vertex/edge/manifold connectivity and overlap;
- per-face relative orientation variation (radians);
- distance change between source-barycentric loci (source-scale-normalized);
- nonrigid pair *observability*, degeneracy ABSTAIN, and source/court lineage;
- `same_drawable=UNKNOWN`, `different_drawable=UNKNOWN`,
  `mechanical_owner=UNKNOWN`, `training_authorized=false`.

`tests/corpus/test_corpus370_generic_pair_kinematics_v1.py` covers a
globally rotated common part, a **single skinned triangle-pair bending**,
two disconnected triangles moving under one shared root, point contact,
nonmanifold incidence, degenerate posed triangles, face reorder, wrong
parent hash, forged teacher source and duplicate pixels. These are synthetic
**contract tests**, not measured Corpus370 pose-pair coverage.

**Real-data applicability:** current three-source palette/visibility receipts
have independently measured render effects, but no exact matching
authored-pose teacher NPZ + independent canonical rest-face reimport replay
was established in this experiment. A character without authored motion is
explicitly `MASKED` for this diagnostic, not motion=zero and not a bad
graphics character. Do not say this observational kinematic gate passed
3/3 real sources before verifying their actual teacher receipts.

## Next real experiments, predetermined gates

1. Join the existing `CORPUS370_V5_AUTHORED_TEMPORAL_PILOT` and V5 source-rest
   courts to the exact source-hit face pairs, first checking all three
   subjects' native motion inventory. Static cases are counted as MASKED;
   no substitute generated motions.
2. Where actions exist, run paired native temporal frames through the
   new diagnostic **privately**. Validate same source SHA, rest-face ordering,
   camera, frame provenance, and the independent source equivalence court.
   Publish counts and signed SHA receipts only, not private source/model arrays.
3. Preregister actual *isolated* control pairs: zero-control, A\u2192B, B\u2192A,
   root sham, inverse/reversal, varying amplitudes; verify the attempted
   control is distinct and executed, then independently court each
   geometry/alpha/mechanical consequence. Authored playback alone does not
   satisfy this gate.
4. For each candidate drawable split/merge hypothesis, compare exact
   **render/edit consumer consequence** at held-out view/pose/target motions:
   CAA appearance retention, compositing/ray order, seam continuity,
   attachment/visibility persistence and edit independence. Require both
   positive and negative hypotheses plus N/A/ABSTAIN; same-color and
   common-root controls must fail to create labels.
5. Only after a source-relative, canonical-space, lineage-safe mapping passes
   the same consumer, consider qualified drawable pair masks as trainable
   targets. Unknown and absent-family rows must remain masked, never zeroed.

**Next admission:** execute the actual native teacher source-and-canonical
binding. Synthetic CI cannot substitute for that measurement. IRIS geometry
pretraining may proceed on independently qualified geometry if its own sealed
rosters permit it; do not make speculative drawable truth an artificial
global blocker or mint a fake graphics PASS. PR #327 stays draft.
