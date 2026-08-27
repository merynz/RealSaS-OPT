# RealSaS-OPT — Current State

**Date:** 2026-08-27  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_CALIBRATION8_V1_3_COMPLETE__E0B_MUTUAL_P003_FROZEN__DOWNSTREAM_PROBE_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active E0 question is still whether the **observable single-pose A×8 common-frame substrate is downstream-rigging sufficient**. No E0 product PASS/FAIL has been declared. Proxy32 / DEV32 / external panels remain closed.

## Frozen E0 geometry arms

```text
E0-0  full/closed mesh canonical ceiling
E0-a  exact observable P + oracle physical persistence
E0-b  exact observable P + deterministic SurfaceBuilder persistence
```

Primary causal interpretation:

```text
E0-0 -> E0-a = observable coverage / supported-surface tax
E0-a -> E0-b = deterministic persistence / admission tax
```

Camera scale is recovered from observable exact P + raster pixel-center authority; `camera.json` is not consumed. The invalid fixed-half-extent V1.2 run remains non-authoritative. Corrected V1.3 is the geometry authority.

## Corrected Calibration-8 V1.3 — COMPLETE

Population is the historical frozen **FIT calibration-8** only. This is mechanism/calibration evidence, **not generalization**.

Raw deterministic E0-b BASE aggregate:

- macro precision `0.904444`;
- macro recall `0.961679`;
- micro precision `0.906806`;
- micro recall `0.963237`;
- micro F1 `0.934170`;
- false pairs `964`;
- definite cross-component false pairs `35`;
- definite cross-component accepted rate `0.3384%`;
- median TP-only match-P95 `0.001766`.

Interpretation boundary: cross-component false is only a **definite unsafe lower bound**. Same connected-component wrong matches can still be harmful near articulation boundaries and must be judged downstream.

## E0-b safety calibration — FROZEN

Five observation-only admission variants were evaluated prospectively on the same frozen FIT calibration-8. Teacher identity was evaluator-only after each admission mask was frozen; optimizer steps `0`.

Selected rule: **`MUTUAL_P003`**.

Exact rule:

1. begin with a BASE accepted source→target E0-b pair;
2. run the same deterministic `derived_match_row` from that target observation back into the source view;
3. abstain if no reverse match exists;
4. compute common-frame cycle error to the original source anchor;
5. admit iff `cycle_P <= 0.003`;
6. do not impose anchor-level `support>=2` or `support>=3`;
7. triangle/bary/teacher identity are forbidden from admission.

Calibration effect, BASE → `MUTUAL_P003`:

- micro precision `0.906806 -> 0.935181` (`+2.837 pp`);
- micro recall `0.963237 -> 0.960053` (`-0.318 pp`);
- micro F1 `0.934170 -> 0.947454`;
- false pairs `964 -> 648` (`-32.8%`);
- true-positive pairs `9380 -> 9349` (`-0.33%`);
- cross-component false `35 -> 26`;
- definite unsafe cross-component accepted rate `0.3384% -> 0.2601%`.

The precision gain is positive on **8/8 calibration families**. `MUTUAL_P006` is too weak; `MUTUAL_P003_SUPPORT2` gives negligible extra precision while materially reducing recall; `SUPPORT3` is over-aggressive.

Canonical safety report:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/E0_B_SAFETY_CALIBRATION_RESULT_V1.md`

## Next executable gate — matched downstream sufficiency probe

Use the prior S0 fixed downstream-isolation philosophy, but do not train on the eight calibration families.

Required matched arms:

```text
D0 = E0-0 full/closed canonical substrate
D1 = E0-a observable + oracle persistence
D2 = E0-b observable + frozen MUTUAL_P003 admission
```

The same probe architecture, optimizer budget, random seed, target authority and evaluator must be used across all three arms.

Primary questions:

```text
D0 -> D1 = does visible-only coverage materially hurt rigging evidence?
D1 -> D2 = does the residual deterministic persistence error materially hurt rigging evidence?
```

Calibration-8 may be used to set/freeze numerical non-inferiority margins **only after the probe/training split and implementation are frozen**. No gradient may touch calibration-8. Only after that may a separate frozen Proxy32 qualification package be authorized.

## Downstream probe authority to reuse

Prior S0 evidence established the relevant fixed-isolation probes:

- Arachne isolation: GT product skeleton/parents fixed; padded 57D point/joint pair features → MLP `96 -> 96 -> 1`, softmax over legal controls; seed `1862`; AdamW `2e-3`, wd `1e-4`; 6 epochs.
- Geppetto isolation: oracle-count joint-locus probe only; the unconstrained count/existence decoder was previously invalid for head decisions. This remains an information-isolation test, not full product skeleton generation.

Exact S0 implementation/split authority should be recovered and reused rather than casually reimplemented with a different capacity.

## Product-domain audit — deferred but required

Do **not** retroactively alter the frozen E0 calibration membership.

After E0 closure, open a prospective `PRODUCT_DOMAIN_V1` eligibility audit for later train/dev/generalization panels. Candidate exclusions include assets outside the intended product morphology, such as giant rectangular props that occlude most views or other rigging-irrelevant/non-product structures. Criteria must be frozen before looking at new-panel outcomes.

## After E0 only — perception intervention

Exactly three matched arms remain authorized after substrate sufficiency is established:

```text
C1  scratch + current objective
C2  scratch + tail-aware correspondence objective
C3  pretrained visual/multiview prior transplant
    (MapAnything-family prior candidate; NOT full MapAnything product/camera wrapper)
```

Only if representation ranks correct correspondence truth adequately may multiview search/fusion/propagation/PatchMatch-style mechanisms be promoted.

## Authorization state

`D2_D5_CORRESPONDENCE_LOCALIZATION = COMPLETE`

`E0_CALIBRATION8_V1_2 = INVALIDATED`

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_SAFETY_CALIBRATION = COMPLETE_FIT_ONLY`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROBE = NEXT`

`DOWNSTREAM_NONINFERIORITY_MARGINS = NOT_FROZEN`

`E0_PROXY32_QUALIFICATION = CLOSED`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`C1_C2_C3 = NOT_YET_EXECUTED`

`MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`

`FULL_PATCHMATCH = NOT_AUTHORIZED`
