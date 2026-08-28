# RealSaS E0 — Downstream Proxy Margin Decision V1

**Date:** 2026-08-28  
**Status:** `MARGINS_FROZEN_BEFORE_PROXY32__PROXY32_DEV32_CLOSED`

This decision is frozen after the preregistered four-member FIT calibration was inspected and before any `qualification_proxy32` member is opened. It applies only to the research information-isolation proxies; it is not a product Geppetto/Arachne quality claim.

## Authorities

- calibration result SHA-256: `82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`
- V1.3 seal: `RealSaS.E0.DownstreamProxySeal.v1_3`
- frozen FIT population: 374 train / 59 selection / 4 calibration
- Arachne common-target train population: 371
- Geppetto train population: 374
- D0/D1/D2 unchanged; D2 admission remains teacher-free `MUTUAL_P003`; Proxy32 and DEV32 remain closed.

## Calibration observation

Lower is better unless noted.

| Metric | D0 | D1 | D2 | D0→D1 | D1→D2 | D0→D2 |
|---|---:|---:|---:|---:|---:|---:|
| Arachne CE | 1.195226 | 1.131971 | 1.146689 | -5.29% | +1.30% | -4.06% |
| Arachne influence displacement mean | 0.0125986 | 0.0117482 | 0.0119394 | -6.75% | +1.63% | -5.23% |
| Geppetto joint mean | 0.137285 | 0.143399 | 0.136530 | +4.45% | -4.79% | -0.55% |
| Geppetto family-p95 | 0.171975 | 0.179822 | 0.164098 | +4.56% | -8.74% | -4.58% |

Arachne shows a small but consistent D1→D2 persistence tax: all four calibration assets have positive CE and influence-displacement deltas. The tax is small in aggregate (+1.30% CE, +1.63% influence). The 59-FIT selection diagnostics show the same direction at smaller magnitude (+0.37% CE, +0.47% influence).

Geppetto shows a small D0→D1 observable-coverage tax, while D2 recovers it in both calibration and the 59-FIT selection diagnostics. D2 is slightly better than D0 in aggregate joint mean on both panels. This must not be interpreted as deterministic persistence being intrinsically superior to the oracle; each arm is retrained and the result is an information-sufficiency comparison.

## Frozen non-inferiority gates

For `qualification_proxy32`, all comparisons use the aggregate metric on the truth-capable qualification intersection.

### Primary continuous gates — 5% relative degradation maximum

For each of the following lower-is-better metrics:
- Arachne `CE`;
- Arachne `influence_disp_mean`;
- Geppetto `joint_mean`;

all three ratios must satisfy:

```text
D1 / D0 <= 1.05
D2 / D1 <= 1.05
D2 / D0 <= 1.05
```

The 5% value is a round practical-effect bound for this information-isolation gate, not an exact fit to any one observed calibration delta. It is deliberately much smaller than a permissive 10% primary tolerance.

### Geppetto tail guard — 10% relative degradation maximum

For lower-is-better `family_p95`:

```text
D1 / D0 <= 1.10
D2 / D1 <= 1.10
D2 / D0 <= 1.10
```

### Threshold-catastrophe veto

PCK is retained as a secondary threshold diagnostic rather than the primary margin because the four-asset calibration is visibly noisier at fixed thresholds. Nevertheless qualification is vetoed if D2 suffers a catastrophic direct threshold loss:

```text
PCK@0.05(D2) >= PCK@0.05(D0) - 0.10
PCK@0.08(D2) >= PCK@0.08(D0) - 0.10
```

## Qualification decision rule

`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS` requires every primary ratio, the Geppetto tail guard, and both PCK catastrophe vetoes to pass on the unopened truth-capable Proxy32 intersection. No margin, metric role, population rule, or D2 admission rule may be changed after Proxy32 is opened.

Secondary metrics (`weight_mae`, `top1`, `gt_mass_top4`, `joint_p95_mean`, paired distributions) remain reported diagnostics and may explain a result but cannot rescue a failed frozen gate.

## Firewalls

- `qualification_proxy32 = CLOSED` at the time of this freeze.
- `DEV32 = CLOSED`.
- no product-domain filtering was applied retrospectively.
- the three zero-common-Arachne-target train assets remain in the frozen E0 population and Geppetto population; they are excluded only from Arachne gradient eligibility symmetrically across D0/D1/D2.
- no product PASS is claimed from Calibration4.

**Next:** open Proxy32 only under this frozen evaluator and make a single qualification decision.