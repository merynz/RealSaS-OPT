# RealSaS — FIT_PROXY32 Coordinate and Residual Audit V1

**Date:** 2026-08-29  
**Status:** `CLOSED__CONTROLLED_CAPACITY_LADDER_WARRANTED__PRODUCT_SELECTION_NOT_AUTHORIZED`

## Question

After DTB-ND1, does the historical scratch learner evidence live in a coordinate system comparable to the current depth tolerance scale, and what does its fit/unseen failure shape actually imply?

## Coordinate audit

The SHA-bound historical fit-scale bundle trains only camera-forward depth:

`d_pred = P_pred · forward`

against:

`d_truth = P_truth · forward`

using FP32 SmoothL1.

The model reconstructs:

`P = h*g_x*right - h*g_y*up + d*forward`

and the evaluator reports:

`||P_pred - P_truth||`.

Thus the only non-depth term in historical P error is mismatch between the analytic observable screen-plane coordinates and the raster-authoritative truth surface.

A real FIT_PROXY32 witness (`asset_ea593d044e14f20abe6d2818`) was reconstructed with the exact frozen 4096-sample/view seed, authoritative barycentric geometry, native-alpha `h_sheet`, and all eight view yaws.

Across 8 × 4096 loci:

- screen-plane mismatch p95: **6.39e-5**
- screen-plane mismatch max: **1.22e-4**
- DTB-ND1 lower bracket depth abs-p95 scale: ~**4.90e-3**

The screen p95 term is ~**1.3%** of the current tolerance p95 scale on this witness.

Therefore historical `P_p95` is **depth-dominated and directly useful as a residual-scale diagnostic** at the present precision scale.

We do not claim exact mathematical equality for every FIT_PROXY32 asset without re-running the full checkpoint in explicit `d` metrics.

## Matched-unit context

DTB-ND1:

`0.00250 <= epsilon_RMS_critical < 0.00275`

For the tested ell=0 perturbation this is approximately:

`0.00490 <= abs(depth)_p95 boundary scale < 0.00539`.

Historical P-V5 R256 small 8x2 scratch fit:

- aggregate P_p95: **0.003706**
- worst cell: **0.005741**

So the small-cohort scratch learner can fit at approximately the current tolerance scale.

## The 32 -> 128 -> 512 family ladder is not a capacity ladder

| train families | nominal asset exposures | TRAIN_DIAG8 median cell p95 | TRAIN_DIAG32 median | FIT_PROXY32 median |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1792 | .00731 | .00859 | .17066 |
| 128 | 448 | .01382 | .01819 | .09814 |
| 512 | 112 | .02246 | .03138 | .06593 |

As family diversity increases:

- unseen FIT_PROXY32 improves strongly;
- training-family precision degrades strongly.

This is a diversity/generalization versus fitting/interference trade-off under a fixed model and fixed 7168-step budget.

It cannot falsify or prove the representation-capacity hypothesis because per-family exposure changes 16x across the ladder.

## Rung-512 disaggregation

At the current DTB-ND1 upper p95 scale (~.00539):

- TRAIN_DIAG32 families at/below scale: **0 / 32**
- FIT_PROXY32 families at/below scale: **0 / 32**

Rung-512 source medians:

| source | TRAIN_DIAG32 median | FIT_PROXY32 median |
| --- | ---: | ---: |
| Objaverse animated | .03079 | .07166 |
| Quaternius CC0 | .03311 | .02395 |
| KayKit CC0 | .13438* | .12346 |

`*` TRAIN_DIAG32 contains only one KayKit family in this diagnostic.

Thus `.1998` aggregate FIT_PROXY32 p95 is **not** a universal 54x unseen-family collapse.

The supported decomposition is:

1. a large global fitting/precision deficit already present on training families;
2. a further unseen-family gap concentrated especially in Objaverse;
3. strong family/source tails;
4. almost no cel-vs-ink style separation at rung512.

## Decision

A **true controlled pretrained representation/capacity ladder is warranted**.

This historical data-scale ladder already shows why: more family diversity buys transfer while the fixed scratch system loses the precision needed by the consumer tolerance budget.

The forthcoming DINOv2 S/B/L/g experiment must keep corpus, optimizer/step protocol, fusion, trainable downstream parameter count, foundation raster/framing, and consumer acceptance operator fixed while only frozen pretrained representation capacity changes.

## Execution order / stop rule

This decision does **not** authorize immediate product model selection.

Next:

1. close the minimal real-consumer validity interlock: `ClosedRiggingVolume V0 -> InteriorRiggingSubstrate V0 -> Geppetto G0 -> Compiler -> Arachne A0 -> deformation/proof`;
2. then execute the controlled DINOv2 S/B/L/g representation ladder;
3. do not open another cheap deterministic geometry-operator tolerance intervention before those gates.

Proxy27 remains historical PASS evidence but was not used for this tuning. DEV32 remains closed.
