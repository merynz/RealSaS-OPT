# Extraction Topology Observability Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `EXTRACTION_TOPOLOGY_OBSERVABILITY_PREREG_V1.md`  
**Workflow:** `iris-extraction-topology-observability-v1-stage`  
**Run:** `40224711550`  
**Status:** `EXTRACTION_OBSERVABILITY_TABLE_ESTABLISHED_V1`

## Result

The court measured base-grid sign observability under sub-cell phase for simple analytic topology families. Values below are in the TESSA normalized domain.

| Resolution | Grid spacing | Sphere p95 / guaranteed diameter | Cylinder p95 / guaranteed diameter | Parallel-sheet p95 / guaranteed gap |
|---:|---:|---:|---:|---:|
| 24 | 0.041667 | 0.057418 / 0.072169 | 0.050125 / 0.058926 | 0.040404 / 0.041667 |
| 32 | 0.031250 | 0.043064 / 0.054127 | 0.037594 / 0.044194 | 0.030303 / 0.031250 |
| 48 | 0.020833 | 0.028709 / 0.036084 | 0.025062 / 0.029463 | 0.020202 / 0.020833 |
| 64 | 0.015625 | 0.021532 / 0.027063 | 0.018797 / 0.022097 | 0.015152 / 0.015625 |

`guaranteed` uses the analytic phase-independent covering bound. The finite 33-phase court was consistent with those bounds.

## Interpretation

Extraction resolution is not merely final mesh density. The signed field is sampled on the base grid before FlexiCubes deformation parameters are applied. If a closed component / thin support / separation gap fails to create the necessary sampled sign structure, later beta/alpha/gamma extraction degrees of freedom cannot recover the absent topology.

At the current product resolution 24, the phase-independent TESSA-domain observability scales are therefore approximately:

- closed sphere-like component diameter: `0.0722`;
- long thin support diameter: `0.0589`;
- two-interface separation gap: `0.0417`.

Resolution 48 halves these scales; resolution 64 reduces them to roughly 37.5% of the res24 values.

## No resolution selected yet

This court deliberately does not authorize a production-resolution change. The next necessary evidence is a real normalized morphology-scale census over target subjects/views using silhouette/local-thickness/gap statistics. Authored triangle edge length is explicitly not a valid proxy.

A fixed higher resolution should also be compared against an adaptive/coarse-to-fine policy if the real morphology distribution shows that fine resolution is needed only in a minority of regions.

## Authority

No Product, Truth139, FIT8 or ModelRelease authority changed.
