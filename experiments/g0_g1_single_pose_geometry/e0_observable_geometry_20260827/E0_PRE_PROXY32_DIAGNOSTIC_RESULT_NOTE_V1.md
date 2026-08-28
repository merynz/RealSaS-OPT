# RealSaS E0 — Pre-Proxy32 Diagnostic Result Note V1

**Date:** 2026-08-28  
**Status:** `DIAGNOSTIC_COMPLETE__INTERPRETATION_FROZEN__PROXY27_DOWNSTREAM_STILL_CLOSED`

Result authority SHA-256: `906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`.

## Firewalls verified

- diagnostic train/evaluation: 64 / 59 frozen FIT assets;
- official E0 compact-pack rebuild count: **0**;
- Proxy32/Proxy27 downstream outcomes remained closed;
- DEV32 remained closed;
- qualification margins were not changed.

## D-A — implicit orientation in X36

Matched normal probes on 30,208 evaluation points:

| probe | mean cosine ↑ | oriented median ↓ | p90 ↓ | p95 ↓ |
|---|---:|---:|---:|---:|
| P3 | 0.528319 | 44.495° | 104.420° | 126.509° |
| X36 | **0.787410** | **24.954°** | **64.274°** | **77.908°** |

The current 36D substrate therefore contains substantial decodable oriented-normal information through its observation-derived support/raster/depth provenance. `N0 = no explicit N` must not be interpreted as `zero normal/orientation information`.

For historical context only, S0-B2's strongest selected deterministic normal reported family-aggregated row-percentile summaries of 24.09° median / 111.78° p90 / 150.38° p95 and a 16.77% >90° orientation-error fraction. The new X36 statistics use a different population and pooled-point aggregation, so these numbers are **not an apples-to-apples estimator comparison**. They nevertheless motivate a stronger visibility/depth-constrained deterministic-N challenger before N-B3 execution.

## D-B — D0 budget/allocation diagnostic

Increasing D0 from 512 to 2048 area-uniform full-mesh points greatly improved geometric coverage on selection59:

- NN median: `0.021264 -> 0.011082`;
- NN p95: `0.044077 -> 0.022910`;
- coverage@0.01: `0.1614 -> 0.4638`;
- coverage@0.02: `0.4897 -> 0.8776`;
- coverage@0.05: `0.9573 -> 0.9993`.

The matched fixed-capacity downstream probes changed only minimally. Arachne CE moved `3.229692 -> 3.224036`; Geppetto joint mean moved `0.220097 -> 0.219943`. D1@512 remained modestly better on those aggregate metrics (`3.203258` CE; `0.217867` joint mean).

### Frozen interpretation boundary

Under the **frozen fixed-capacity downstream probes**, increasing the D0 area-uniform surface budget from 512 to 2048 greatly improves geometric coverage but produces negligible downstream improvement. Thus D0 spatial-coverage scarcity alone does not explain the observed D0@512–D1@512 difference **within this fixed-probe regime**.

This diagnostic does **not** distinguish D0-specific representation/allocation effects from downstream-probe capacity saturation. A future D1@2048 control could separate those explanations, but it is explicitly not required before Proxy27 qualification.

No measurable downstream benefit was detected from a 4× increase in area-uniform full-mesh sampling under the frozen probes; however, the **specific contribution of hidden-surface information was not isolated**. The experiment increased both visible and hidden full-mesh sampling density.

## Consequence

This diagnostic cannot rescue, veto, or modify qualification. The next action remains the one-shot evaluation of the already-frozen 27-member truth-capable Proxy32 intersection under the already-frozen non-inferiority rule.
