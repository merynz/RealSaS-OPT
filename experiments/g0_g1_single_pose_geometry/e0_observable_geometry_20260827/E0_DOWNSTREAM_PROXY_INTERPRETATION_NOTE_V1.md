# RealSaS E0 — Downstream Proxy Interpretation Note V1

**Date:** 2026-08-28  
**Status:** `INTERPRETATION_FROZEN_AFTER_CALIBRATION4__MARGINS_UNCHANGED__PROXY32_CLOSED`

This note records interpretation caveats raised after Calibration4 and after the numerical margin decision was frozen, but before any Proxy32 member was opened. It does **not** change any frozen margin, metric role, population, model, or D2 admission rule.

## 1. D0 is a fixed-budget full-mesh reference, not an unconstrained information ceiling

D0 samples exactly 512 surface points by **area-uniform full-mesh sampling**. D1/D2 do not use that sampling measure: each view contributes a deterministic visible-P pool (up to 2048 rows/view), followed by common-frame farthest-point sampling over the pooled visible union.

Therefore D0 and D1/D2 share the same 512-slot downstream budget but differ in sampling measure. D0 has access to the full canonical mesh and oracle observation provenance, but its 512 slots can be spent on surfaces irrelevant to the visible/deformation task. D1/D2 allocate their 512 anchors over the actually observed union and spatially spread them with FPS.

Accordingly, Arachne D0->D1 improvements must **not** be interpreted as "less information is intrinsically better". A plausible mechanism is fixed-budget sample allocation: visible-union FPS can be a better task-targeted allocation than full-mesh area-uniform sampling. This mechanism is an interpretation hypothesis, not a proven causal decomposition.

The frozen `D2/D0 <= 1.05` qualification gate remains unchanged. Its correct interpretation is non-inferiority to the **fixed-budget full-mesh reference**, not to an unconstrained oracle ceiling.

## 2. D2 may act as a noise-filtered persistence subset

Frozen MUTUAL_P003 removes a small number of base deterministic correspondences, disproportionately false/ambiguous ones. For a finite-capacity set decoder, this can plausibly reduce training/input noise even though it removes information in the set-theoretic sense.

Therefore a D2 improvement over D1 is not evidence that deterministic persistence contains more physical information than oracle persistence. The defensible conclusion is only that deterministic persistence is **not currently a downstream bottleneck** under these matched proxy consumers.

## 3. Persistence-tax magnitude

Calibration4 shows Arachne D1->D2 degradation of +1.30% CE and +1.63% influence displacement; the 59-FIT selection panel shows the same direction at smaller magnitude (+0.37% CE, +0.47% influence displacement).

The shared direction argues against a one-asset sign artefact, but Calibration4 is too small to estimate the magnitude precisely. The current evidence supports "small measurable tax"; it does not justify treating ~1.5% as a precise population estimate.

## 4. PCK veto is intentionally a catastrophe veto, not a quality gate

Calibration4 Geppetto aggregate values:

- PCK@0.05: D0 = 0.242103, D2 = 0.197657;
- PCK@0.08: D0 = 0.451571, D2 = 0.397014.

The frozen absolute veto `D2 >= D0 - 0.10` therefore permits, at the Calibration4 D0 baselines, as much as approximately:

- **41.3% relative degradation** at PCK@0.05;
- **22.1% relative degradation** at PCK@0.08.

At an S0-B-like baseline near 0.17, the same -0.10 absolute rule would indeed permit ~58.8% relative degradation. Thus the PCK rule is deliberately only a catastrophic-threshold-loss veto. The primary Geppetto quality protection is the 5% joint-mean gate plus the 10% family-p95 tail guard.

No PCK margin is changed by this note.

## 5. Normal-channel clarification

The downstream consumer feature slot is exactly 36D: canonical P, support mask, matched raster XY, support fraction, and camera-forward depths. **No normal vector is exposed to Arachne or Geppetto in D0, D1, or D2.**

D2 correspondence admission does use a normal-consistency gate, but the normal is `N_derived`, deterministically reconstructed from observable P + raster neighborhood; it is not teacher normal. D1 oracle persistence uses teacher physical carrier identity for persistence only; it does not expose teacher normal to the consumer.

Therefore this D0/D1/D2 result already demonstrates downstream proxy sufficiency without an explicit normal feature **given exact observable P**. It does not settle whether a learned/direct N head is useful when P is predicted/noisy, nor whether direct N adds useful information beyond structured N(P). That remains the purpose of N-B3.

## Firewalls

- margin decision unchanged;
- Proxy32 remains closed at the time of this note;
- DEV32 remains closed;
- no retrospective product-domain filtering;
- no D0/D1/D2 rebuild;
- no product Geppetto/Arachne claim.
