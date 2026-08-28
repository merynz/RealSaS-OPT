# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_PRE_PROXY32_DIAGNOSTICS_PREREGISTERED__RUN_DIAGNOSTICS_NEXT__PROXY32_DOWNSTREAM_CLOSED__DEV32_CLOSED`

This file is the single continuation authority.

## Frozen E0 downstream gate

The current research question is whether the exact observable A×8 common-frame substrate preserves enough rigging-relevant information for matched downstream information-isolation proxies. Product Geppetto/Arachne do not yet exist.

```text
D0 = fixed-budget full-mesh reference + oracle observation provenance
D1 = exact visible-union substrate + oracle physical persistence
D2 = exact visible-union substrate + frozen teacher-free MUTUAL_P003 persistence
```

Important interpretation correction: D0 is **not** an unconstrained information ceiling. D0 spends 512 area-uniform full-mesh slots; D1/D2 pool visible rows across the eight views and use common-frame FPS for 512 anchors. Therefore D0→D1 mixes information access with fixed-budget sample allocation. This is recorded in `E0_DOWNSTREAM_PROXY_INTERPRETATION_NOTE_V1.md`; frozen qualification margins are unchanged.

## Reusable substrate cache

`prep_v1_2` is immutable reusable authority for the official gate:
- 437 compact packs complete;
- ordinary downstream/evaluator changes must not rebuild them;
- geometry source SHA-256 `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`;
- observable camera scale uses robust through-origin regression; no `camera.json` forward;
- D2 remains `MUTUAL_P003`, cycle threshold `0.003`, teacher identity forbidden.

## Calibration V1.3 complete

Frozen FIT population:
- train 374;
- selection 59;
- calibration truth-capable 4;
- Arachne common-target train 371;
- Geppetto train 374.

V1.3 result SHA-256:
`82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`

Calibration4 aggregates, lower is better:

| Metric | D0 | D1 | D2 | D0→D1 | D1→D2 | D0→D2 |
|---|---:|---:|---:|---:|---:|---:|
| Arachne CE | 1.195226 | 1.131971 | 1.146689 | -5.29% | +1.30% | -4.06% |
| Arachne influence displacement | 0.0125986 | 0.0117482 | 0.0119394 | -6.75% | +1.63% | -5.23% |
| Geppetto joint mean | 0.137285 | 0.143399 | 0.136530 | +4.45% | -4.79% | -0.55% |
| Geppetto family-p95 | 0.171975 | 0.179822 | 0.164098 | +4.56% | -8.74% | -4.58% |

Defensible conclusion only: deterministic persistence is not currently a downstream bottleneck under these matched proxy consumers. Do not interpret D2>D1 as more physical information.

## Frozen qualification margins

Authority: `E0_DOWNSTREAM_PROXY_MARGIN_DECISION_V1.md/json`.

Primary lower-is-better metrics Arachne CE, Arachne influence displacement, Geppetto joint mean must each satisfy:

```text
D1/D0 <= 1.05
D2/D1 <= 1.05
D2/D0 <= 1.05
```

Geppetto `family_p95` uses the same three ratios at `<=1.10`.

PCK is catastrophe-only:

```text
PCK@0.05(D2) >= PCK@0.05(D0)-0.10
PCK@0.08(D2) >= PCK@0.08(D0)-0.10
```

Margins/metric roles/population/D2 admission cannot change after Proxy32 downstream evaluation opens.

## Pre-Proxy32 interpretation diagnostics — preregistered, outcomes unseen

Before qualification, two cheap interpretation controls were preregistered using **only already-open FIT train/selection**. They cannot change or rescue the frozen qualification gate.

Authority files:
- `E0_PRE_PROXY32_DIAGNOSTIC_PREREG_V1.md`
- `E0_PRE_PROXY32_DIAGNOSTIC_SPLIT_V1.json`
- `E0_PRE_PROXY32_DIAGNOSTIC_SPEC_V1.json`
- `E0_PRE_PROXY32_DIAGNOSTIC_PREFLIGHT_V1.json`
- `E0_PRE_PROXY32_DIAGNOSTIC_PACKAGE_V1.json`
- `N_B3_EPSILON_PREREG_V1.md`

Diagnostic population:
- train64 = first 64 frozen train assets after existing common Arachne target-support eligibility;
- evaluation = frozen selection59;
- all 123 are disjoint from the 32 Proxy families.

Diagnostics:
1. **Implicit-N decodability**: matched `P3 -> N` versus current `X36 -> N` probes, exact geometric N evaluator-only. This determines how much orientation is already implicit in support/raster/depth features.
2. **D0 allocation/budget**: diagnostic-only D0@2048 versus official D0@512 and D1@512, with geometric coverage plus matched Arachne/Geppetto consumers. Official 437 packs are reused, not rebuilt.

The N-B3 epsilon ladder is preregistered independently with `epsilon={0,.001,.003,.010}` and arms N0/N3a/N3b/N2, but is **not executed** by the pre-Proxy32 diagnostic notebook.

## Proxy32 metadata-only freeze

Original frozen `FIT_PROXY32` membership is 32 families (26 Objaverse + 3 Quaternius + 3 KayKit), selected before this downstream gate and disjoint from train512.

Metadata-only master-ledger intersection:

```text
capabilities.iris && capabilities.geppetto && capabilities.arachne
```

produces 27/32 downstream-truth-capable Proxy families. Exact authority:
`E0_PROXY32_TRUTH_INTERSECTION_V1.json`.

Truth-capable ordered-ID SHA-256:
`b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817`

This metadata inspection does **not** open Proxy32 downstream metrics. `proxy32_downstream_evaluation_opened=false` remains authoritative.

## Next executable action

Run `RealSaS_E0_PRE_PROXY32_DIAGNOSTICS_V1.ipynb` on Colab GPU.

It must:
- rebuild **0/437** official packs;
- reuse only 123 existing official packs;
- create only diagnostic caches for those 123 assets;
- run implicit-N and D0-budget diagnostics;
- leave Proxy32 downstream CLOSED;
- leave DEV32 CLOSED;
- leave frozen qualification margins unchanged.

After its result is inspected, open the frozen 27-member truth-capable Proxy32 intersection exactly once with the already-frozen qualification evaluator. No diagnostic outcome may alter that qualification rule.

## Firewalls / authorization

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROXY_PREP_V1_2 = COMPLETE_437_REUSABLE_PACKS`

`E0_DOWNSTREAM_PROXY_CALIBRATION_V1_3 = COMPLETE_FIT_ONLY`

`DOWNSTREAM_NONINFERIORITY_MARGINS = FROZEN_V1`

`E0_PRE_PROXY32_DIAGNOSTICS = PREREGISTERED__NOT_YET_EXECUTED`

`N_B3_EPSILON = PREREGISTERED__NOT_YET_EXECUTED`

`E0_PROXY32_TRUTH_INTERSECTION = FROZEN_METADATA_ONLY_27_OF_32`

`E0_PROXY32_DOWNSTREAM_EVALUATION = CLOSED`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`PRODUCT_DOMAIN_V1 = DEFERRED_PROSPECTIVE_AFTER_E0`
