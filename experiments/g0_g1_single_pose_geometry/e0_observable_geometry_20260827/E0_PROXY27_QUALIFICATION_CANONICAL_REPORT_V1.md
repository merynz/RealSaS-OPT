# RealSaS E0 — Proxy27 Qualification Canonical Report V1

**Date:** 2026-08-28  
**Status:** `E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS`  
**Scope:** frozen 27-member truth-capable intersection of the pre-existing FIT_PROXY32 qualification panel; information-sufficiency proxies only, not product Geppetto/Arachne and not an IRIS predicted-P qualification.

## Byte authority

- Qualification result: `E0_PROXY27_QUALIFICATION_RESULT_V1.json`
- Result SHA-256: `86153066f487bc16fdfd4e817494fa3bd052def076ef97cd998c013455ee74e5`
- Qualification seal: `E0_PROXY27_QUALIFICATION_SEAL_V1.json`
- Seal SHA-256: `0bbd00843436e78c4cc38cb50765295ba3ba7844c57b7ca7cb23fe3f7c57fe78`
- Ordered population SHA-256: `b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817`
- Calibration authority SHA-256: `82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`
- Pre-Proxy diagnostic authority SHA-256: `906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`

The six frozen V1.3 Arachne/Geppetto checkpoints were loaded unchanged. No training or retuning occurred. The non-binding expectation record was not used for the decision. DEV32 remained closed and qualification margins were unchanged.

## Aggregate metrics

| Metric | D0 | D1 | D2 |
|---|---:|---:|---:|
| Arachne CE ↓ | 2.630374 | 2.405529 | 2.501573 |
| Arachne influence displacement ↓ | 0.0118685 | 0.0120806 | 0.0122102 |
| Geppetto joint mean ↓ | 0.227764 | 0.233639 | 0.226213 |
| Geppetto family-p95 ↓ | 0.409818 | 0.422785 | 0.426079 |
| Geppetto PCK@.05 ↑ | 0.102829 | 0.081119 | 0.087815 |
| Geppetto PCK@.08 ↑ | 0.220206 | 0.198208 | 0.209494 |

## Frozen qualification checks

All **14/14** preregistered checks passed.

| Check | Observed | Limit | Result |
|---|---:|---:|---|
| Arachne CE D1/D0 | 0.914520 | <=1.05 | PASS |
| Arachne CE D2/D1 | 1.039926 | <=1.05 | PASS |
| Arachne CE D2/D0 | 0.951033 | <=1.05 | PASS |
| Arachne influence D1/D0 | 1.017873 | <=1.05 | PASS |
| Arachne influence D2/D1 | 1.010727 | <=1.05 | PASS |
| Arachne influence D2/D0 | 1.028791 | <=1.05 | PASS |
| Geppetto joint mean D1/D0 | 1.025794 | <=1.05 | PASS |
| Geppetto joint mean D2/D1 | 0.968215 | <=1.05 | PASS |
| Geppetto joint mean D2/D0 | 0.993190 | <=1.05 | PASS |
| Geppetto family-p95 D1/D0 | 1.031641 | <=1.10 | PASS |
| Geppetto family-p95 D2/D1 | 1.007792 | <=1.10 | PASS |
| Geppetto family-p95 D2/D0 | 1.039679 | <=1.10 | PASS |
| Geppetto PCK@.05 D2-D0 | -0.015014 | >=-0.10 | PASS |
| Geppetto PCK@.08 D2-D0 | -0.010712 | >=-0.10 | PASS |

The narrowest primary headroom is Arachne CE D2/D1: deterministic persistence pays a `+3.9926%` CE tax versus D1 under a frozen `+5%` maximum degradation. This leaves approximately `1.0074` percentage points of relative-ratio headroom and should be treated as the most sensitive rung when predicted-P/depth noise is introduced later.

## Interpretation

### What is now supported

The exact observable A×8 common-frame substrate is downstream-information-sufficient under the frozen matched Arachne/Geppetto information-isolation proxies.

- D0→D1 does not cause a disqualifying downstream collapse.
- Replacing oracle persistence with teacher-free `MUTUAL_P003` in D2 remains within all frozen primary, tail and catastrophe margins.
- Geppetto joint mean shows the calibration direction again: D1 pays a small localization tax (`+2.58%`) and D2 recovers it, ending modestly better than D0 (`-0.68%`).
- Arachne CE shows D1 substantially better than D0 (`-8.55%`); D2 pays a `+3.99%` tax versus D1 but remains `-4.90%` better than D0.

### What did not generalize exactly from calibration4

The pre-open expectation was deliberately non-binding, and Proxy27 provided genuinely new evidence rather than merely replaying calibration4.

- Arachne influence displacement was expected to improve from D0→D1, but on Proxy27 it worsened slightly (`+1.79%`), with D2 ending `+2.88%` versus D0. It still passes the frozen 5% gate.
- Geppetto family-p95 was expected to recover strongly in D2, but Proxy27 D2 is instead `+3.97%` versus D0 and `+0.78%` versus D1. It still passes the frozen 10% tail guard comfortably.
- Therefore the calibration4 magnitude/detailed tail pattern must not be promoted as a population law. The qualification conclusion rests only on the preregistered margins.

## Boundary of the PASS

This is **not** an E0 product pass and does not prove a trained IRIS can yet deliver a sufficiently accurate `P`.

The experiment grants exact observable P to D1/D2. Product IRIS will instead produce depth-derived predicted points `P_hat = O + d_hat F`. Therefore a separate predicted-P / ray-depth robustness bridge remains mandatory before freezing the final IRIS contract.

Likewise, N-B3 remains independent and must decide whether explicit/direct N has marginal downstream value beyond X36 plus the stronger visibility/depth-constrained deterministic `N3v` construction.

## Decision

`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS = TRUE`

The E0 exact-observable downstream information-sufficiency gate is closed.

Next scientific sequence:

1. execute N-B3 V1.1 (`N0 / N3a / N3v / N3b / N2`, epsilon `{0,.001,.003,.010}`);
2. derive a ray-aligned forward-depth noise tolerance ladder from the downstream margins;
3. use that tolerance as the target for the depth-only IRIS construction;
4. qualify `P_hat -> MUTUAL_P003 -> downstream` before freezing the product IRIS contract.

`DEV32_EXTERNAL = CLOSED` remains unchanged.
