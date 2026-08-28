# E0-b Safety Calibration V1 — preregistration

Date: 2026-08-27
Status: `FROZEN_FIT_CALIBRATION_ONLY__NOT_GENERALIZATION`

## Scientific question

With E0 V1.3 exact observable `P` frozen, how much unsafe persistence remains after observation-only admission rules, before any learner or pretrained prior is considered?

This is **not** an IRIS/MapAnything test. E0-b consumes exact teacher-observable `P`; teacher surface identity is forbidden during admission and opens only after every candidate mask is frozen for evaluation.

## Population

Exactly the historical frozen FIT calibration-8:

- `asset_551ea351b43a1787d0f55536`
- `asset_36fb02305846592b1ecdf3d4`
- `asset_0679fdef64f19a4832a6d521`
- `asset_76313e4bd82b82fcd1659c70`
- `asset_6f086a5b1a66378ffe04d7e4`
- `asset_425122d500ecf5767404f9c0`
- `asset_5a19f8c5254be7bf30c504f5`
- `asset_f8a40d6c5d815fe79c8b5e42`

No membership changes after outcome inspection. DEV32, Proxy32, TUNE/CAL/EXTERNAL remain closed.

## Frozen source outcome

The only admissible E0 input is `E0_CALIBRATION8_GEOMETRY_V1_3`.

V1.2 remains invalidated by the fixed-half-extent apparatus defect and is not outcome authority.

## Admission variants

All variants are computed from E0-b observation-only data before teacher truth opens:

1. `BASE` — current E0-b accepted non-self pairs.
2. `MUTUAL_P006` — BASE plus reverse deterministic match and common-frame return error `<= 0.006`.
3. `MUTUAL_P003` — BASE plus reverse deterministic match and common-frame return error `<= 0.003`.
4. `MUTUAL_P003_SUPPORT2` — `MUTUAL_P003` and at least 2 admitted non-self target views for the source anchor.
5. `MUTUAL_P003_SUPPORT3` — `MUTUAL_P003` and at least 3 admitted non-self target views.

No variant is automatically selected by the runner. Calibration evidence is reviewed first; any chosen admission rule must then be frozen before downstream E0-a/E0-b non-inferiority evaluation or any Proxy32 qualification.

## Evaluator-only truth

After admission masks are frozen, teacher geometry may classify each admitted pair using:

- E0-a visibility/support authority;
- source teacher triangle only for connected-component membership;
- target raster triangle only for connected-component membership;
- physical common-frame error `<= 0.003` as the current E0 physical-match witness.

Teacher identity never alters admission masks.

## Metrics

Per asset and aggregate:

- precision / recall / F1;
- false pair count;
- `cross_component_false_pairs`;
- `same_component_wrong_pairs`;
- `oracle_hidden_bridge_pairs`;
- `unsafe_cross_component_commit_rate = cross_component_false / admitted`;
- `cross_component_share_of_false`;
- `all_accepted_match_P_p95` — includes false accepted pairs;
- `true_positive_match_P_p95` — TP-only, explicitly separated from the prior ambiguous P95.

Connected-component crossing is an **unsafe lower bound**, not the complete definition of a harmful correspondence. Same-component wrong matches may still be harmful near articulation boundaries and are decided by downstream probes.

## Firewalls

- learner optimizer steps: `0`;
- MapAnything/pretrained prior: forbidden;
- DEV32: closed;
- Proxy32: closed;
- teacher identity in admission: forbidden;
- automatic winner/rule selection: forbidden;
- product/generalization PASS claim from this panel: forbidden.

## Next after this calibration

Freeze one observation-only safety/admission policy, then run the same matched downstream probes on E0-0 / E0-a / frozen-safe E0-b. The decisive quantity is downstream `E0-a -> E0-b` persistence tax, not raw correspondence precision alone.
