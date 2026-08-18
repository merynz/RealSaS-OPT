# N1D Global-Foreground Observable GFDR-V2 Closure — Canonical Report V1

**Date:** 2026-08-18  
**Status:** OPEN-DEV FULL-OBSERVABLE MECHANICS PASS  
**sealed21:** CLOSED  
**external10:** CLOSED

## Question

After repairing cross-pose correspondence with frozen N1D persistent descriptors and global B-foreground search, does the downstream N1D GFDR-V2 mechanics chain still pass when exact evaluator geometry/normals/visibility are removed from prediction?

This report concerns the **N1D `GFDRV2.ObservableMechanics.v4` diagnostic contract**. It must not be read as redefining or directly qualifying the older structural GFDR ontology.

## Two-stage causal proof

### Stage A — exact observation geometry, response-only replacement

Prediction geometry/normals/visibility were held to exact observation reference and only the displacement response field was replaced by the frozen global-foreground descriptor route.

Candidate result (20 open-development e00 families):

- D tangent action error: **0.148326**
- F activity Spearman: **0.843860**
- F kernel Spearman: **0.793054**
- R differential error: **0.104815**
- G axis direction abs-cos: **0.880993**
- G axis-line error / diag: **0.087942**
- canonical gates: **9 / 9 PASS**

The previous frozen-current response under the same exact geometry had G axis-line error **0.154159**, failing the `<=0.10` gate. The repaired response reduced it below threshold while preserving direction.

**Causal conclusion:** the old N1D G-line failure was caused by the correspondence/world-response path; the downstream GFDR-V2 operators themselves were not the blocker.

### Stage B — full observable front door

Prediction used only:

```text
Pose A/B rasters + calibrated cameras
  -> Problem-A raster-only A visual-hull carriers X and V_A
  -> frozen N1D persistent descriptor
  -> global B-foreground top-k correspondence sets
  -> deterministic calibrated multiview 3D response
  -> predicted P_B = X + response
  -> frozen N1D normal heads sampled at predicted A/B projections
  -> deterministic B visual-hull visibility V_B
  -> GFDR-V2 observable mechanics
```

No exact observation P/N/V, surface correspondence, joint, parent, weight or intervention metadata entered prediction. Exact sidecars were opened only after **20/20 prediction artifacts were frozen**.

## Full-observable result — 20-family e00

- D tangent action error: **0.148326**
- F activity Spearman: **0.843860**
- F kernel Spearman: **0.813066**
- F all-persistent kernel MAE: **0.004559**
- R differential error: **0.105003**
- G axis direction abs-cos: **0.890067**
- G axis-line error / diag: **0.084111**
- valid F episodes: **19**
- true-supported G carriers / families: **768 / 20**
- canonical gates: **9 / 9 PASS**

The full observable result is effectively equal to or slightly better than the exact-geometry causal response reference on the aggregate GFDR-V2 gates:

| Metric | Exact-geometry causal | Full observable |
|---|---:|---:|
| D tangent | 0.148326 | **0.148326** |
| F activity | 0.843860 | **0.843860** |
| F kernel | 0.793054 | **0.813066** |
| R differential | 0.104815 | **0.105003** |
| G direction | 0.880993 | **0.890067** |
| G line / diag | 0.087942 | **0.084111** |

## Front-door diagnostic quality

These are diagnostics, not gates:

- mean P_A error / diag median across families: **0.027398**
- mean P_B error / diag: **0.039350**
- N_A median angular error: **26.82°**
- N_B median angular error: **29.64°**
- V_A accuracy: **0.6914**
- V_B accuracy: **0.7041**

The important observation is that these primitive estimates are **not exact**, yet the downstream mechanical contract still passes. This is direct support for the RealSaS-Opt criterion:

```text
mechanical sufficiency > exact latent reconstruction
```

It also means the old N1D point-map hard gates are not automatically product blockers for this repaired hybrid front door: the active route obtains geometric carriers deterministically from Problem-A rather than relying on the failed N1D point-map head.

## Scientific conclusion

The post-N1D open-development failure chain is now causally localized and closed for the e00 panel:

1. N1D had strong persistent descriptor evidence.
2. N1D mechanics did not consume it; DIS-centered local transport destroyed a minority of large-error correspondences.
3. Global observable foreground descriptor search removed that bottleneck.
4. Calibrated multiview geometry converts the observation-level candidate sets into useful world response.
5. Repaired response closes the old G-line failure under exact geometry.
6. Replacing exact geometry/normals/visibility with the actual raster-only / predicted front door still closes **all 9 canonical N1D GFDR-V2 gates**.

## Authority

Authorized on open development:

```text
Problem-A X/V_A
+ frozen N1D descriptor / normal evidence
+ global B-foreground candidate search
+ deterministic multiview 3D response
+ B visual-hull visibility
-> N1D GFDR-V2 mechanics
```

**Not yet authorized:** sealed21, external10, product/model-2 handoff. The current evidence is e00-focused. The next required gate is **multi-intervention open-development qualification** with the treatment frozen before opening new episode outcomes.
