# Candidate-Specific Local Motion Energy V1 — Implementation Addendum

**Date:** 2026-08-19  
**Status:** `PRE-RESULT_IMPLEMENTATION_FREEZE`  
**Parent prereg:** `6af5012a37673b5e1826b3d84304463927ff6e23`

This addendum freezes numerical details before any V1 result is inspected.

## Numerical constants

```text
amplitude epsilon = 1e-3 native pixels
energy IQR epsilon = 1e-6
direction minimum magnitude = 1.0 native pixel
pair oracle-separation margin = log(1.10)
amplitude bin width = log(1.05)
```

## DIS sampling

Bilinear sampling is performed directly in the native 256x256 DIS flow field at the native pixel coordinate `project(P_A,v)`. Flow vectors remain in native pixel x/y units.

## Transport conversion

The frozen `transport_offset_srcA` tensor shape is `[8,2,32,32]`. Convert sampled f4 cell offsets by `255/31` independently for x and y.

## delta3D view conversion

For sampled per-view `delta_point_map_srcA[v,i]`:

```text
project(P_A + delta,v) - project(P_A,v)
```

using the canonical orthographic project function and native 256px coordinates.

## Candidate pair ordering implementation

To bound cost without stochastic sampling:

1. if H has <=91 candidates, use all candidates;
2. otherwise choose exactly 91 candidate indices by rounded `linspace(0, len(H)-1, 91)` and unique them;
3. evaluate every unordered pair among these selected candidates (at most 4095 pairs);
4. keep only pairs whose difference in absolute log-amplitude oracle error is >= `log(1.10)`;
5. score a correct strict energy preference as `1`, incorrect as `0`, exact energy tie as `0.5`.

This sampling is deterministic and truth does not affect which candidate indices are selected.

## Oracle-amplitude-bin rank

Candidate log amplitudes are quantized into bins of width `log(1.05)`, starting at the carrier-specific minimum log amplitude. For each bin, use the minimum candidate energy in that bin. The evaluator-only oracle candidate's bin percentile is:

```text
(number of bins with strictly lower energy + 0.5 * tied-lower-energy bins) / number_of_bins
```

Lower is better. This metric is diagnostic only and is not a promotion gate.

## Fusion

For each candidate and each available amplitude arm, normalize energy across that carrier's H:

```text
(E - median(E_H)) / max(q75(E_H)-q25(E_H), 1e-6)
```

Then take the median across available arms candidate-wise. No fitted parameter or family statistic enters fusion.
