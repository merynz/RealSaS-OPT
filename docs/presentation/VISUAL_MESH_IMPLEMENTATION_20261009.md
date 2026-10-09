# Visual mesh: implemented connection and remaining acceptance

2026-10-09. The user authorized completion on canonical main while preserving
the exact mechanical carrier, hierarchy, weights and sealed motion inputs.
This document is an implementation handoff, not a Knight visual PASS.

## Implemented bounded change

| Boundary | Previous behavior | Current behavior |
|---|---|---|
| Source material | Zero-alpha pixels inside the foreground mask became opaque and were counted as direct source | Missing material fails before source-direct compile; no fabricated observation |
| Stage23 CAA | Provenance was one scalar per view | Each raster texel carries provenance and donor/source-view identity; outside-domain padding is explicit |
| Stage24 qualification | Exact texture equality | Exact texture **and** per-texel provenance/donor equality, with byte/hash verification |
| Stage37 visual mesh | Region arrays were loaded but mixed-region faces and seed/domain contradictions could pass the loader | Coherent face/vertex ownership and seed/domain membership are checked; fixed source UV is verified for new qualified views |
| Stage42 visual binding | Qualified CAA texture consumed, provenance was absent downstream | Exact CAA asset/qualification, UV-byte hash, provenance-byte hash and texel fields travel with the existing visual topology and compiled motion |
| Stage43 package | Source PNG, UV and positions | The same package includes typed material fields and compiler-evaluated positive camera depth; array tampering fails before packing |
| Native compositor | Straight-sRGB interpolation then premultiplication; face enumeration determined overlapping ownership; visible provenance hard-coded to0 | Linear-light premultiplied filtering/composition, canonical depth ordering, classified footprint provenance and source-view identity; unknown support, depth ties and layer overflow fail |
| Stage44/45 proof | RGBA/provenance/face-owner parity and structural motion checks | Donor identity is also compared; global, per-frame and connected-patch compiled-material exposure budgets are consumed; completion dominance cannot pass just because native parity passes |
| Stage46 | Structural source-visual predicates | New material-mode products require material provenance/exposure **and** explicit hidden-layer/contact/order acceptance; this partial producer cannot mint a full product PASS |

The frozen old source-direct package remains reproducible through its explicit
legacy sampling contract. Newly compiled canonical source-visual packages use
the sealed material contract; the canonical consumer rejects the old scalar
provenance representation. This is requalification, not silent upgrading of
historical assets.

No normal map, lighting, relighting or geometry shading is added. RGB remains
painted art. The current affine deformation operator is unchanged; this change
does not claim to replace it with ARAP or a learned rig.

## Rebuild and causal scope

The implementation-closure comparison against
`main@4d3a370c0eaea0a13df6c037f0588935aa2820cf` has direct source changes at
20–25,37–38,42–46. IRIS/TESSA/AXIS/MIRA and mechanical stages19,28,32,35,36
retain their implementation identities. Shared adapter modules legitimately
make several stage identities change together; do not declare only42.

The released DAG invalidates39–41 through25/38. They may replay/reseal the same
motion inputs; they do not fit M/G/W. A controlled Go Attempt must pin exact
qualified baseline artifacts for independent nodes and inspect the actual
resolver result before execution. No file in this change imports raw evidence
as a qualified cache hit or edits an old execution ledger.

## What still prevents full visual closure

1. **Amodal material and geometry.** The current source-visual CAA producer still
   covers the visible source raster. It does not produce a complete per-part
   layer below armor/limb/prop occlusions. The canonical mechanical C(p) solver
   exists, but copying its entire carrier appearance into this visual domain
   would repeat the earlier unsupported geometry/appearance expansion. A
   qualified support/addressing relation is required to transfer a donor or
   completion to the right visual part. Runtime support for admitted provenance
   1/2/4 does not itself manufacture or qualify those fields.
2. **Contact and intended overlap.** Canonical camera depth removes accidental
   face-array order, but cannot certify the artist's intended layering. The
   current safe-adjacency regions are not anatomical part labels. Inter-chart
   joints, grip/prop contacts and semantic overlap need explicit qualified
   relations and independent dynamic checks. The proof records these as
   unqualified; healthy triangles are insufficient.
3. **Matched Knight execution.** The developer-host job in Actions run37986333890
   was observed queued without a runner assignment. The
   external inventory also expressly leaves exact carrier receipts, CAA/camera
   binding and Go baseline qualification pending. A local synthetic native
   regression is not a matched latest-carrier Knight Attempt.

The acceptance target remains IDLE/RUN/SLASH × V0–V7 on the same frozen M/G/W,
motion probes and declared support. Count missing material, missing screen
geometry, contact residuals, semantic overlap failures and conditioning
separately. Only after this scope is qualified should a residual render defect
be used to compare or reject the deformation/completion method.

The machine-readable state is
[`VISUAL_MESH_CLOSURE_20261009.json`](../../canonical/VISUAL_MESH_CLOSURE_20261009.json).
The broader connection map remains
[`VISUAL_DEPENDENCY_MAP_20261009.md`](VISUAL_DEPENDENCY_MAP_20261009.md).
