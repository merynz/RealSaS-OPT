# RealSaS — Mage GSA→Geppetto Conditioning Ablation Design Freeze V1

Status: `DESIGN_FROZEN__EXECUTION_BLOCKED_UNTIL_B1S_OPT_FORK_READOUT`

Scientific source at design freeze:
`e2e/mage-scene-first-v1-20260905 @ e5b0425a9f22e831d68db06aeb40a502e9c1d0c0`.

This document freezes the *boundary interventions* now, while the B1s optimizer/recurrent-sensitivity fork is still running. It intentionally does **not** select the optimizer regime or launch a scientific training run before that fork is interpreted.

## Question

Does the current B1-family experimental boundary make Geppetto harder than the production scene-first GSA boundary by:

1. dropping production raster bindings; and/or
2. ignoring GSA surface-topology relations in favor of Euclidean KNN locality?

This is not a GSA redesign and not a Geppetto promotion experiment.

## Fixed facts motivating the arms

The Mage B1-family compact witness is a quantized snapshot of the current scene-first signed GSA: 950 nodes, 2813 predicted-mesh topology edges, robust normals and the exact 8-view support mask.

However:

- the compact experimental reconstruction sets `raster_bindings=()` for every node, causing the last four Geppetto conditioning features to be zero;
- `RiggingSurfaceIR.local_relations` are preserved in the fixture but current Geppetto local attention reconstructs Euclidean neighbors from XYZ and never consumes them.

## Frozen four-arm boundary matrix

All future arms must use the same Mage signed-surface source, the same compact nodes/normals/support, the same teacher target, structural serialization, seed, active geometry losses and forced 41-step decode. Only the two boundary factors below may differ.

| Arm | Raster features | Local-neighbor metric |
|---|---|---|
| A0 CURRENT_COMPACT | stripped / zero, matching B1s | current Euclidean KNN |
| A1 RASTER_ONLY | production exact-camera raster bindings | current Euclidean KNN |
| A2 GSA_TOPOLOGY_ONLY | stripped / zero, matching B1s | GSA_GRAPH_16 |
| A3 RASTER_PLUS_GSA_TOPOLOGY | production exact-camera raster bindings | GSA_GRAPH_16 |

`GSA_GRAPH_16` is preregistered as a cardinality-matched locality intervention:

- self + 15 external neighbors, matching current `knn_k=16`;
- rank by shortest-hop distance on `RiggingSurfaceIR.local_relations`;
- deterministic within-hop tie break by Euclidean distance then sorted `surface_id`;
- disconnected nodes sort after all reachable nodes;
- no teacher skeleton/skin/part identity may enter neighbor construction.

This avoids the confound of comparing a ~6-degree direct adjacency encoder against a 16-token local-attention encoder.

## What is intentionally held until the optimizer fork finishes

The optimizer/horizon used by A0-A3 is not selected in this design freeze. The future execution prereg must copy one regime from the completed B1s optimizer fork **before** any A0-A3 training result is observed.

If the baseline fork does not reproduce the historical 10816 collapse at the exact preregistered check, this ablation is not authorized because the optimizer experiment itself lacks parity.

No boundary arm may be selected or dropped because of an early metric preview.

## Required preflight before any training

1. Reconstruct full production scene-first `RiggingSurfaceIR` from the same frozen signed zero surface and exact Mage cameras.
2. Prove node correspondence to the compact fixture:
   - 950/950 one-to-one;
   - topology edge array exact or explicitly permutation-equivalent;
   - view mask exact;
   - compact quantization error reported.
3. Run `validate_rigging_surface_ir_v1(..., require_scene_first_signed_contract=True)` on the full surface.
4. Prove A0 and A1 have identical non-raster feature columns and identical normalized positions.
5. Prove A0/A2 and A1/A3 differ only in the local-neighbor index.
6. Persist neighbor-index hashes and raster-feature summary statistics.
7. Abort on any teacher truth entering GSA reconstruction or boundary features.

## Metrics

Training/product metrics stay exactly those of the future frozen B1s-compatible geometry experiment. Boundary-specific telemetry must additionally report:

- per-arm nearest-target p95/max/MAE;
- outside-capture count and exact multiplicity occupancy error;
- per-slot errors, with slots 34/35/36 and mirror slots 4-9 explicitly surfaced;
- raster feature channel statistics;
- GSA relation-edge count;
- Euclidean-KNN vs GSA_GRAPH_16 neighbor overlap;
- selected graph-hop histogram;
- training stability telemetry inherited from the chosen optimizer regime.

## Interpretation rules

- A1 > A0 with A2≈A0: raster evidence matters.
- A2 > A0 with A1≈A0: topology-locality metric matters.
- A3 materially exceeds both A1 and A2: complementary boundary evidence.
- No arm improvement: current compact boundary is not a meaningful contributor in this Mage regime.
- Any absolute improvement cannot explain historical B1/B1a/B1s *differences* retroactively because those regimes shared the same frozen input.
- No result from Mage establishes unseen/generalization sufficiency.
- No result authorizes changing product authority: GSA remains geometric evidence; Geppetto remains mechanical proposer; Compiler remains canonical graph authority.

## Nonclaims

No generalization claim.
No B2 authorization.
No GSA promotion change.
No Geppetto architecture promotion from this design alone.
No teacher topology as product evidence.
No post-hoc threshold changes.
