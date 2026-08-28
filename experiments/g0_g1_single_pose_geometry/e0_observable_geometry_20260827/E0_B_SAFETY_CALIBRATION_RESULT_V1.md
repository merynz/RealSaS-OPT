# E0-b Safety Calibration Result V1

- Date: `2026-08-27`
- Population: **frozen FIT calibration-8 only; not generalization**
- Source geometry run: `E0_CALIBRATION8_GEOMETRY_V1_3`
- Result SHA-256: `015d1e79aec7fe37dffd049e5ac7cba8d6c6dd4d7051b3ee6354853f462b9a1d`
- DEV32 / Proxy32 / TUNE / CAL / EXTERNAL: **closed**
- Scientific optimizer steps: `0`
- Teacher identity in admission: **false**; teacher geometry is evaluator-only after masks are frozen.

## Aggregate

| Variant | micro P | micro R | micro F1 | false | cross false | unsafe cross / accepted | TP-only P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 0.906806 | 0.963237 | 0.934170 | 964 | 35 | 0.3384% | 0.001766 |
| MUTUAL_P006 | 0.907683 | 0.963237 | 0.934635 | 954 | 35 | 0.3387% | 0.001766 |
| MUTUAL_P003 | 0.935181 | 0.960053 | 0.947454 | 648 | 26 | 0.2601% | 0.001744 |
| MUTUAL_P003_SUPPORT2 | 0.935400 | 0.923393 | 0.929358 | 621 | 26 | 0.2705% | 0.001729 |
| MUTUAL_P003_SUPPORT3 | 0.926358 | 0.684638 | 0.787363 | 530 | 19 | 0.2640% | 0.001812 |

## Frozen admission decision

`MUTUAL_P003` is selected **using this FIT calibration panel only** and is frozen before any Proxy32/DEV evaluation.

Exact observation-only rule:

1. Start from an E0-b BASE accepted directed source→target match.
2. From the accepted target observation, run the same deterministic `derived_match_row` back into the source view.
3. If no reverse match exists, abstain on that pair.
4. Compute common-frame cycle error between the returned source point and the original anchor.
5. Admit iff `cycle_P <= 0.003`.
6. Do **not** require anchor-level `support>=2` or `support>=3`.
7. No triangle/bary/teacher identity may be consumed by this admission rule.

Calibration rationale:

- micro precision: `0.906806 -> 0.935181` (`+2.837` percentage points);
- micro recall: `0.963237 -> 0.960053` (`-0.318` percentage points);
- false pairs: `964 -> 648` (`-32.8%`);
- true positive pairs: `9380 -> 9349` (`-0.33%`);
- cross-component false: `35 -> 26`;
- unsafe cross-component accepted rate: `0.3384% -> 0.2601%`.

`MUTUAL_P006` is effectively too weak. `MUTUAL_P003_SUPPORT2` gives negligible extra precision while materially reducing recall; `SUPPORT3` is rejected as over-aggressive.

## Per-asset BASE → MUTUAL_P003

| Asset | P base | P m003 | R base | R m003 | false base→m003 | cross base→m003 |
|---|---:|---:|---:|---:|---:|---:|
| 551ea351 | 0.954188 | 0.971574 | 0.979427 | 0.979427 | 64→39 | 24→18 |
| 36fb0230 | 0.818020 | 0.851024 | 0.922535 | 0.919517 | 204→160 | 0→0 |
| 0679fdef | 0.930283 | 0.955838 | 0.976372 | 0.973323 | 96→59 | 0→0 |
| 76313e4b | 0.895072 | 0.938023 | 0.958298 | 0.953191 | 132→74 | 1→1 |
| 6f086a5b | 0.931768 | 0.951165 | 0.970206 | 0.967150 | 93→65 | 1→1 |
| 425122d5 | 0.911862 | 0.935374 | 0.960938 | 0.954861 | 107→76 | 0→0 |
| 5a19f8c5 | 0.893511 | 0.926343 | 0.954667 | 0.950222 | 128→85 | 8→5 |
| f8a40d6c | 0.900850 | 0.933824 | 0.970992 | 0.969466 | 140→90 | 1→1 |

## Interpretation boundaries

- `cross_component_false` is a **definite unsafe lower bound**, not a complete semantic-error measure.
- `same_component_wrong` may still be harmful near articulation boundaries and must be judged by the matched downstream probes.
- `oracle_hidden_bridge` is evaluator-only evidence and is not used to admit/reject pairs.
- `all_accepted_match_P_p95` includes false matches and must not be compared to learner P prediction error.
- `true_positive_match_P_p95` is the correct spatial-tolerance diagnostic for accepted true correspondences.

## Next gate

Run matched downstream E0-0 / E0-a / E0-b probes with **E0-b using frozen `MUTUAL_P003` admission**. Calibration-8 is for margin setting / mechanism inspection only. Proxy32 remains closed until downstream probe/margins are frozen.
