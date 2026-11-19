# RSS.vNext — Native rig/skin teaching without animation clips

**2026-11-19.** Stacked on PR #599. Code, supervision contracts and
regression tests only; no Drive dataset edits, teacher materialization, new
trained release or source asset license claims.

## Correctness problem

The existing `PRODUCT_SEEDED_IRIS_TESSA_V1` student computed MIRA source
skin/chain losses only after a separately qualified **posed motion witness**.
That excluded licensed source files with genuine deform joints and skin
weights but no authored animation. Source-native skin is a distinct teacher:
observed joint transforms are not necessary to establish vertex weights.

## Independent skin-only source authority

`models/rss_vnext/native_skin_transport_v1.py` transports the source-native
vertex-indexed weights to the exact post-seeded-TESSA M* by independently
registered ORIGINAL face IDs and barycentric coordinates. It rejects:
- absent/mismatched rights and provenance receipt;
- incorrect asset, source witness, final mesh face index or tensor hashes;
- incorrect native source skin simplex and missing/malformed triangle support;
- source-native skin joint-column order not independently matched to the
  exact causal AXIS teacher sequence, including a separate native joint-order
  digest and signed joint-order match receipt;
- source-rest reprojection drift beyond the independently bounded tolerance;
- attempts to construct correspondence from student nearest neighbours.

The transport is a teacher-only tensor and does not enter the two-image
student scene carrier. Its exact support and label hashes are provided by
the independent source launcher and preserved in trainer registry/checkpoints.
It is NOT sufficient to grant source pose deformation, contact, physics,
draw-order, real wing semantics, or a ProductRevision.

## Loss paths

With qualified source native support:
- AXIS causal joint/parent/CHAIN event tokens may train without controls.
- MIRA can train skin and joint-to-chain mass/conditional losses without
  source animation when the independent native skin receipt exists.
- Hard FSQ skin tests may use the same source skin identity.
- Any `posed_surface` or `dynamic_surface` loss MUST still require the
  original independent native pose and G3/G3B consequences. Source skin
  cannot masquerade as a motion teacher.
- The prior fully source-pose-qualified route remains accepted unchanged.
  Parallel source-skin-only and posed transports for the same objective
  are not silently combined.

## Remaining blockers

1. Source foundry must produce **actual** admitted face-barycentric skin
   support and independent right-to-train receipts for real licensed files.
2. Character control grammar and inference-time chain role semanticalization
   are not learned automatically from rigging alone.
3. Long AXIS V6 causal decoding must be GPU qualified under 2048-token budget;
   changing the cap is not cache optimization.
4. Production GPU training and unseen two-image→mechanically correct 2D
   rendered animation still require independent qualification.
