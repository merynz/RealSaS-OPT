# Extraction Topology Observability Court — Preregistration V1

**Status:** sealed before execution.  
**Execution class:** deterministic CPU; no learned model.  
**No Product / Truth139 / FIT8 authority.**

## Question

Current product Shape evaluates the implicit field on the pinned FlexiCubes base grid at extraction resolution 24. FlexiCubes deformation parameters are applied after those corner field samples exist; therefore a topology event that never creates a mixed-sign base-grid cube is not recoverable by later extraction weights.

This court asks:

> Across resolution 24/32/48/64, what normalized geometric scale is observable independent of sub-cell phase for simple closed components, thin supports and separated surfaces?

The court measures **field-sign topology observability**, not final mesh quality.

## Exact domain

Use IRIS normalized cube `[-1,1]^3`, matching the pinned extraction wrapper. For resolution `R`, base-grid spacing is `h = 2/R`. Report both IRIS units and TESSA units (`x_tessa = 0.5*x_iris`).

## Analytic families

### 1. Sphere component

SDF:

`f(x)=||x-c||-r`.

Sweep radius relative to grid spacing and translate the center through a full periodic cell phase in x/y/z. A sample is observable iff at least one base-grid edge/cube neighborhood contains both negative and positive field samples associated with the component.

### 2. Infinite cylinder / thin support proxy

Axis-aligned cylinder through the domain centerline with radial SDF. Sweep the two perpendicular grid phases and radius. This approximates long thin appendage/support observability independent of endpoint effects.

### 3. Two parallel sheets / gap proxy

Use two zero sets separated by gap `g`, represented by a signed slab/union construction whose topology contains two distinct interfaces. Sweep phase normal to the sheets. A sample is counted as separated only if the sampled sign sequence contains both interfaces with at least one correctly signed interval between them.

## Resolutions

`24, 32, 48, 64`.

## Phase sweep

Use a deterministic uniform phase grid with at least 33 offsets per relevant axis over one base-cell period. Sphere uses the Cartesian product of phase offsets. Cylinder uses the two perpendicular axes. Sheet gap uses the normal axis.

## Scale sweep

For each family, sweep normalized scale in units of `h` densely enough to locate:

- `p50`: smallest scale with >=50% phase observability;
- `p95`: smallest scale with >=95%;
- `guaranteed`: smallest scale with 100% phase observability over the sealed finite phase grid.

Report physical normalized scale in both IRIS and TESSA units.

## Expected analytic sanity checks

For a sphere, a conservative phase-independent radius sufficient to contain a grid vertex within the interior is `sqrt(3)/2 * h`; the measured 100% transition should be consistent with this scale up to finite phase-grid resolution and the exact mixed-sign criterion.

For a long axis-aligned cylinder, the analogous covering radius is `sqrt(2)/2 * h`.

These are sanity checks, not hard-coded verdict thresholds.

## Selection policy

This V1 court does **not** select a production extraction resolution by itself. It produces a resolution-to-observable-scale table. Production selection additionally requires a real morphology-scale census (silhouette/local thickness/gap statistics) so the resolution is chosen against actual FIT8/target morphology rather than arbitrary analytic scales.

A later decision may choose adaptive/coarse-to-fine extraction if a single fixed resolution is unnecessarily expensive.

## Interpretation limits

- This court does not evaluate learned field error.
- It does not evaluate FlexiCubes triangle quality after a topology is observable.
- It does not use authored mesh triangle edge length as a morphology-scale proxy.
- It does not authorize any increase in product inference resolution.
