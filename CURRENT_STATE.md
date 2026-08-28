# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_DOWNSTREAM_CALIBRATION_COMPLETE__MARGINS_FROZEN__PROXY32_NEXT__DEV32_CLOSED`

This file is the single continuation authority.

## Active scientific question

Does the exact observable A×8 common-frame substrate preserve enough rigging-relevant information relative to a full-mesh ceiling?

```text
D0 = full/closed canonical surface + oracle observation provenance
D1 = exact observable surface + oracle physical persistence
D2 = exact observable surface + frozen teacher-free MUTUAL_P003 persistence
```

Interpretation:
- D0→D1 = observable coverage/support tax.
- D1→D2 = deterministic persistence/admission tax.

These are fixed research information-isolation proxies, not product Geppetto/Arachne.

## Frozen substrate/cache authority

`prep_v1_2` is now an immutable reusable dataset artifact for this gate:
- 437 compact packs complete;
- pack rebuild is **not** required for downstream model/evaluator changes;
- geometry source SHA-256: `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`;
- camera recovery uses robust through-origin observable-only regression; `camera.json` is forbidden forward;
- D2 admission remains `MUTUAL_P003`, cycle threshold `0.003`, teacher identity forbidden.

Rebuild the 437 packs only if the substrate/pack contract itself changes.

## Downstream execution authority

Frozen FIT population:
- train: 374;
- selection: 59;
- calibration truth-capable: 4.

Arachne training uses the symmetric target-support eligibility:
`COMMON_D0_D1_D2_AT_LEAST_ONE_VALID_SKIN_TARGET`.

This yields:
- Arachne train: 371;
- Geppetto train: 374;
- selection: 59/59 for both;
- calibration: 4/4 for both.

The three Arachne-zero-common-target assets remain in the frozen E0 and Geppetto populations. This is not retrospective product-domain filtering.

V1.3 result SHA-256:
`82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`

V1.3 run seal:
`E0_DOWNSTREAM_PROXY_SEAL_V1_3.json`

Status in seal:
`FIT_CALIBRATION_COMPLETE__MARGINS_NOT_FROZEN__PROXY32_DEV32_CLOSED`

The subsequent margin decision below supersedes only the margin-status field; scientific artifacts are unchanged.

## Calibration4 findings

Lower is better.

| Metric | D0 | D1 | D2 | D0→D1 | D1→D2 | D0→D2 |
|---|---:|---:|---:|---:|---:|---:|
| Arachne CE | 1.195226 | 1.131971 | 1.146689 | -5.29% | +1.30% | -4.06% |
| Arachne influence displacement mean | 0.0125986 | 0.0117482 | 0.0119394 | -6.75% | +1.63% | -5.23% |
| Geppetto joint mean | 0.137285 | 0.143399 | 0.136530 | +4.45% | -4.79% | -0.55% |
| Geppetto family-p95 | 0.171975 | 0.179822 | 0.164098 | +4.56% | -8.74% | -4.58% |

Interpretation:
- Arachne has no observable-coverage penalty in aggregate. D2 pays a small, consistent persistence tax versus D1 (~1–2%), while remaining better than D0 overall.
- Geppetto shows a small observable-coverage tax at D1; D2 recovers it in both calibration4 and the 59-FIT selection diagnostics. D2 is essentially equal/slightly better than D0 in aggregate joint mean.
- D2 outperforming D1 on Geppetto must not be interpreted as deterministic persistence being intrinsically superior to oracle persistence; each arm is retrained.

## Frozen non-inferiority margins

Canonical decision:
- `E0_DOWNSTREAM_PROXY_MARGIN_DECISION_V1.md`
- `E0_DOWNSTREAM_PROXY_MARGIN_DECISION_V1.json`

Frozen before Proxy32 is opened.

Primary lower-is-better metrics:
- Arachne CE;
- Arachne influence displacement mean;
- Geppetto joint mean.

For each:
```text
D1 / D0 <= 1.05
D2 / D1 <= 1.05
D2 / D0 <= 1.05
```

Geppetto tail guard:
```text
family_p95 D1/D0 <= 1.10
family_p95 D2/D1 <= 1.10
family_p95 D2/D0 <= 1.10
```

PCK catastrophe veto:
```text
PCK@0.05(D2) >= PCK@0.05(D0) - 0.10
PCK@0.08(D2) >= PCK@0.08(D0) - 0.10
```

After Proxy32 is opened, changing these margins, metric roles, population rules, or D2 admission is forbidden.

## Firewalls

- `qualification_proxy32 = CLOSED`.
- `DEV32 = CLOSED`.
- no pretrained consumer.
- no product Geppetto/Arachne claim.
- no product-domain retrospective cleanup.
- no teacher identity in D2 admission.
- no non-inferiority rescue by secondary diagnostics.

## Next executable action

Open the truth-capable intersection of `qualification_proxy32` exactly once and evaluate D0/D1/D2 with the frozen checkpoints and frozen margin decision. Do not retrain, retune, alter margins, alter D2, or rebuild the 437 substrate packs.

If every frozen primary/tail/PCK veto passes:
`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS`.

Otherwise:
`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_FAIL` and localize the failed rung without changing the sealed qualification.

## Parked

- `N_B3`: strongest structured N(P) falsification; resume after D0/D1/D2 qualification.
- `PRODUCT_DOMAIN_V1`: prospective product-domain filtering after E0; giant non-deforming props/planes may motivate criteria but cannot retroactively alter this frozen E0 population.
- C1/C2/C3, full MapAnything wrapper, full PatchMatch remain unauthorized.

## Authorization state

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROXY_PREP_V1_2 = COMPLETE_437_REUSABLE_PACKS`

`E0_DOWNSTREAM_PROXY_CALIBRATION_V1_3 = COMPLETE_FIT_ONLY`

`DOWNSTREAM_NONINFERIORITY_MARGINS = FROZEN_V1`

`E0_PROXY32_QUALIFICATION = CLOSED__NEXT_GATE`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`N_B3 = PARKED_UNTIL_AFTER_D0_D1_D2_QUALIFICATION`
