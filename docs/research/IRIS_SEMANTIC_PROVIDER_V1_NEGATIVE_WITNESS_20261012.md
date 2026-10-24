# IRIS Semantic Provider V1 — Negative Knight Witness (2026-10-12)

## Authority boundary

This is a provider-level negative witness, not a negative result for canonical M,
IRIS geometry, or the semantic-evidence architecture.

Drive run: `RUN_20261012T002055Z`.
Repository revision: `34b2687517ba5984d28cc10013af65ad2283076b`.
Canonical base: `82e31b566cdb811e5b1c41fd4f84a6eaeeee7ce3`.

## Observed result

The V1 `Grounding DINO -> prompted SAM2` provider emitted 141 masks, but many
unrelated text labels collapsed onto near-identical visible-character masks. The
notebook visualization therefore placed multiple label names at almost the same
mask centroid. Direct inspection of `semantic_masks.npz` confirmed that this was
not merely a plotting artifact.

Compiler fusion correctly abstained:

- canonical face count: 6,928;
- qualified semantic faces: 0;
- unknown semantic faces: 6,928;
- owners: 0;
- grip candidates: 0;
- source-overlap relations: 0;
- pose-order conflicts: 0;
- rest M byte equality after diagnostic: true;
- recorded M pose byte equality after diagnostic: true.

Therefore the first run does **not** test whether qualified semantic evidence can
solve grip or presentation-order problems. It only falsifies this first provider
adapter for the Knight source views.

## V2 decision

Do not teach or modify IRIS geometry. Preserve the V1 negative witness.

Replace text-first box grounding with a segmentation-first provider:

1. SAM2 segment-everything produces label-free local candidate masks.
2. CLIPSeg produces frozen text-conditioned dense semantic fields.
3. Each candidate may receive at most one semantic class, and only with a
   sufficient class probability, margin, and localization score.
4. Whole-subject candidates are rejected.
5. Cross-label near-duplicate masks are measured before serialization.
6. `PROVIDER_MASK_COLLAPSE`, `WHOLE_SUBJECT_MASK_DOMINANCE`, insufficient
   proposal count, or insufficient label diversity fail closed before M fusion.
7. Accepted output reuses `RealSaS.IRISFrozenSemanticMaskEvidence.v1`; no second
   geometry carrier or owner authority is introduced.

No runtime grip/order consumer is authorized by this document.
