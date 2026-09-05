# RealSaS — Mage GSA / RiggingSurfaceIR Boundary Audit V1

Status: `AUDIT_OPEN__NO_BEHAVIOR_CHANGE_AUTHORIZED`

Pinned branch before this audit: `e2e/mage-scene-first-v1-20260905 @ 39d2d05b5b76dab415541bb7ffc7922de71edfae`.

## Question

Could part of the current Geppetto difficulty come from the deterministic IRIS→GSA→RiggingSurfaceIR→Geppetto boundary rather than from Geppetto itself?

This audit intentionally does not change Geppetto, GSA, serialization, optimizer settings or product authority. It first establishes what the current Mage witness actually contains and what evidence is lost or ignored at the boundary.

## 1. Mage Geppetto witness provenance is current scene-first GSA, not the old D2 adapter

The frozen compact fixture contains 950 surface points, 950 robust normals, 2813 topology edges and an 8-view support mask. The source signed zero surface is the Mage run `20260904T220929Z`.

Independent reconstruction from `ZERO_SURFACE_PRODUCT_CLIPPED.npz` with the current scene-first adaptive voxel compactor (`target_nodes=1024`) produced:

- voxel divisions: 17;
- compact nodes: 950;
- compact topology edges: 2813;
- edge array: exact equality with fixture;
- compact point nearest-neighbor correspondence: 950/950 bijective;
- compact point max absolute difference from fixture decoding: `3.0841e-05` world units, consistent with fixture quantization;
- robust normal angular difference after current `RobustLocalPCA.v1` + compaction: p95 `0.0198°`, max `0.0280°`;
- scene-first self-zbuffer support reconstructed from the exact 8 orthographic Mage cameras: view mask exact equality for all 950 nodes.

Therefore B1/B1a/B1s are genuinely conditioned on a quantized snapshot of the current scene-first signed GSA output. Findings about the legacy `rigging_surface_from_d2_arrays`, IRIS-V2 persistence lattice, or DTB-ND1 raster-neighbor path must not be projected onto this Mage witness unless separately shown relevant.

## 2. The compact experimental witness drops production raster bindings

Production `rigging_surface_from_scene_first_zero_mesh_v1` writes per-visible-view `raster_bindings` and declares `PIXEL_CENTER_XY` + resolution metadata.

The frozen Mage Geppetto compact fixture retains only points, normals, edges and view mask. Its reconstructed `RiggingSurfaceIR` uses empty `raster_bindings` for every node. Consequently the existing `GeppettoConditioningAdapterV2` emits zeros for feature columns:

- `raster_mean_x`
- `raster_mean_y`
- `raster_std_x`
- `raster_std_y`

Measured on the frozen derived Geppetto input: all four columns are exactly zero over all 950 rows.

Reconstructing the production raster bindings from the exact Mage cameras shows that three of these four channels are materially nonzero on Mage:

- normalized raster mean-x std across nodes: ~0.1503;
- normalized raster mean-y std across nodes: ~0.4624;
- normalized raster std-x mean: ~0.1656;
- normalized raster std-y is effectively zero because all eight yaw cameras share the same screen-up axis.

Interpretation: the current B1-family experiments omit real deterministic evidence that the production GSA already provides. This cannot explain B1 vs B1a vs B1s differences because the omission is constant across all three regimes, but it can raise the absolute difficulty floor and can contribute to regional weakness.

No training intervention is authorized from this finding alone. A source-preserving conditioning ablation is required.

## 3. GSA local topology is preserved in the fixture but currently ignored by Geppetto conditioning

The scene-first GSA emits 2813 `SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR` relations derived from compacted predicted-mesh face adjacency. The compact fixture preserves those 2813 edges exactly.

`GeppettoConditioningAdapterV2` does not consume `RiggingSurfaceIR.local_relations`. The current Geppetto encoder instead constructs a 16-nearest Euclidean neighborhood from XYZ internally; because self-distance is not masked, this is self + 15 external Euclidean neighbors per surface node.

Measured on Mage:

- GSA topology edges: 2813 unique undirected edges;
- Geppetto Euclidean-16NN graph: 7742 unique external undirected edges;
- GSA-edge recall inside the Euclidean graph: ~99.61%;
- Euclidean-graph precision against direct GSA face adjacency: ~36.19%;
- therefore ~4940 Euclidean neighbor edges are not direct GSA topology edges.

The extras are not obviously catastrophic cross-component bridges on Mage: the GSA graph is one connected component, ~78.8% of extra KNN edges connect nodes at graph distance 2 and ~98.9% at distance <=3. Still, the encoder is replacing a sparse surface adjacency with a much denser local relation graph and this semantic change has not been separately justified or ablated.

Interpretation: `local_relations` is currently dead evidence for Geppetto, even though it is produced and preserved by GSA.

## 4. Regional substrate audit for the B1s best-step deficits

At B1s best step 10752, the three failed serialized slots are 34, 35 and 36. Their nearest-surface neighborhoods were compared with the mirror branch and all 41 teacher loci.

Selected 16-nearest-surface support statistics:

- slot 34: mean support views 2.1875, 3/16 zero-support neighbors, 6/16 with support <=1;
- slot 35: mean support views 2.25, 3/16 zero-support neighbors, 5/16 with support <=1;
- slot 36: mean support views 2.5625, 2/16 zero-support neighbors, 4/16 with support <=1.

Across all 41 teacher loci, slots 34 and 35 are in relatively weak support neighborhoods, while slot 36 is not exceptional. Direct-topology degree is also mixed: slot 35 has a low local mean degree, but slots 34 and 36 do not.

Across all 41 loci, B1s per-slot error has no strong monotonic correlation with local support count, zero-support-neighbor count, local topology degree, or local surface density in this simple audit.

Interpretation:

- regional substrate weakness may contribute to slots 34/35;
- it does not explain slot 36 by itself;
- it cannot explain the 10752→10816 training collapse because the substrate is frozen throughout training;
- optimizer/recurrent instability and substrate sufficiency remain separate live hypotheses.

## 5. RiggingSurfaceIR itself is a weakly enforced type boundary

`RiggingSurfaceIR` is currently a frozen dataclass but has no type-level invariant validator. Producer and consumer functions enforce selected invariants locally, but malformed instances can be constructed without a central check.

Examples not globally enforced by the type itself include:

- non-empty / unique `surface_id`;
- finite `P` and finite normals;
- support view range and uniqueness;
- raster-binding/support-view consistency;
- local-relation endpoint existence;
- duplicate relation IDs;
- relation self-loop policy;
- required lineage / operator metadata by builder class.

This is a contract-hardening gap. It is not yet evidence that the Mage fixture is malformed; the measured Mage fixture is internally coherent on the fields audited above.

## 6. Correction to legacy-path findings

The following concerns are real in legacy/alternate substrate paths but do not currently explain the Mage scene-first witness unless those paths are reintroduced:

- `rigging_surface_from_d2_arrays` has weak observation-level provenance and synthetic persistence IDs;
- DTB-ND1 local-plane code has fixed raster offsets and only a weak eigenvalue nondegeneracy check;
- `build_surface_from_persistence` uses an unweighted mean but IRIS-V2 persistence spread is checked upstream;
- IRIS-V2 raster-local-relation logic has its own inferred-lattice assumptions.

Current Mage scene-first GSA instead uses dense signed zero-surface geometry, cKDTree robust normals, adaptive voxel compaction, predicted-mesh face adjacency and exact-camera self-zbuffer support.

## 7. Current decision

Do not reinterpret B1a PASS as proof that every deterministic boundary is optimal. It proves that the frozen scene-first substrate contains enough information for the current decoder to fit Mage when supervision identity is stabilized.

Do not reinterpret B1s FAIL as proof that GSA is the blocker. The frozen substrate cannot explain regime-to-regime differences or the late training collapse.

Before any new Geppetto architecture change after the optimizer fork, perform two source-preserving boundary ablations:

1. **Raster restoration diagnostic** — same scene-first nodes/normals/support, but feed the production raster bindings instead of four zero raster channels.
2. **Topology-consumption diagnostic** — compare current Euclidean KNN locality against a controlled encoder that consumes GSA `local_relations`, without changing target serialization or downstream authority.

Separately harden `RiggingSurfaceIR` with an explicit validator and producer/consumer invariant tests before the full Mage E2E chain is declared deterministic-boundary clean.

## Scope

No generalization claim. No Geppetto promotion. No GSA promotion change. No threshold change. No B2 authorization. This is a boundary audit only.
