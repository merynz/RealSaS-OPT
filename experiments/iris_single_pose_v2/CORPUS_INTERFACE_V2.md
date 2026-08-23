# IRIS Single-Pose V2 — Corpus Interface

Status: `AUDIT_CONTRACT__NO_TRAINING_AUTHORITY`
Date: 2026-08-24

## Master vs IRIS source

The Master Corpus may retain full L1 rig/skinning truth. The IRIS runtime source may not. Before any model/data-prep process runs, selected assets must be staged through a physical allow-list.

Allowed: native RGBA, raster primitive/triangle id, barycentric coordinates, pixel index/coverage, camera/view metadata, canonical vertices/faces or equivalent geometry-only authority, and audit provenance hashes.

Forbidden from staged IRIS source: joints/bones, roots/parents, skin/weights, weight-derived deform masks, constraints/actions, GFDR/mechanical targets, owner/compiler IDs, Pose B.

The staged geometry NPZ must physically contain only the geometry allow-list fields required by the frontend.

## Observation authority

Primary observation is native `1024x1024 x 8`. 512 and 256 controls must be deterministic derivatives of the same 1024 authority with identical asset membership, camera and teacher geometry. No separate rerender or resplit is allowed for a matched resolution ablation.

## Dense truth

Per covered native pixel, primitive/triangle + barycentric authority reconstructs exact continuous canonical P and N.

V2 primary prep may not silently collapse this to the old sparse regime. Preflight target is at least 4096 observation-supported geometry loci per view when coverage permits, or an equivalent systematic/dense scheme, and a target of at least 4096 persistent anchors per asset after dedup/support filtering when coverage permits. Historical 512-sample/384-track settings are controls only.

Exact mini settings are preregistered after memory/I/O preflight and may not be tuned after TUNE results.

## Truth policy

Teacher physical-locus identity/provenance may construct positives/negatives, select evaluation queries, define target coordinates and score predictions. It may not enumerate inference candidates, enter neural input, or become a product identity/owner label.

Every evaluator must declare `GT-query-conditioned` versus `autonomous-source-track`.

## Split policy

Existing canonical-asset-disjoint split is preserved. CAL/DEV/EXTERNAL remain sealed. The next mini uses only open FIT/TUNE and freezes membership before optimizer steps.

## Required no-optimizer census

For every selected asset: 8/8 views present; native resolution/derivative lineage valid; yaw/order valid; raster nonempty; primitive/bary indices valid; P finite and inside audited envelope; N finite/unit-valid; physical IRIS firewall PASS; no FIT/TUNE duplicate; style coverage aligned; no blank/quarantined asset.

The optimizer may start only after the complete census passes.
