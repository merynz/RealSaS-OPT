# IRIS Controlled V1 — Representation Ceiling PASS / Open Pilot Decision

**Date:** 2026-08-23

## Exact representation ceiling: PASS

64 deterministic open assets (51 FIT / 13 TUNE), 65,830 cross-view pairs.

Key results:

- P_EXACT top1 = 0.9976302598
- P_EXACT top4 = 1.0
- P_EXACT top8 = 1.0
- P_EXACT family top1 p10 = 0.9934295619
- PN_EXACT top1 = 0.9942123652
- PN_EXACT top8 = 0.9998936655
- P_NOISE_0p0025 top8 = 1.0
- PN_P0p005_N20 top1 = 0.9206744645
- PN_P0p005_N20 top8 = 0.9996658059
- ambiguous-within-0.003 fraction = 0.0151754519

Primary gate thresholds were exceeded; representation ceiling = PASS.

Interpretation: legal observable P/(P,N) can address persistent surface loci under controlled exact-camera observations. This is not learner proof.

Important nuance: the fixed naive P+N scoring rule slightly underperformed P alone on exact top1. N remains legal/rich evidence, but should not be forced into direct correspondence scoring with an uncalibrated fixed coefficient.

The strong noisy top8 result supports high-recall set-valued/top-k ambiguity retention: even P noise 0.005 + N noise 20° retained true locus in top8 for 99.9666% of pairs.

## Operational full-cache result

Fast V2 ceiling cache: 64 assets in 311 s.

Full 3248 open cache initially burst above 1 asset/s but decayed to ~0.26 asset/s under Google Drive random-read pressure, implying ~3.3 h cache preparation. This is an operational I/O issue, not a scientific failure.

Full cache completion is therefore paused before wasting hours.

## New next gate: OPEN PILOT LEARNER

Use the existing 64-asset local ceiling manifest:

`/content/IRIS_CONTROLLED_V1_FAST_CACHE/IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json`

Run 8 epochs on 51 FIT and evaluate on 13 TUNE only. No sealed split is opened.

The pilot compares random-init vs trained state. Diagnostic PASS requires:

- TUNE P Euclidean error reduced by at least 20%;
- coarse persistence top8 improves by >5 percentage points;
- fine persistence top8 improves by >5 percentage points.

This gate asks whether the controlled neural architecture can begin learning the exact observable target before spending hours building the full cache.

## Training implementation correction

Original trainer selected `best.pt` with a TUNE total whose term set changed between warmup epochs 0–3 and persistence-active epochs >=4. That made cross-epoch selection non-comparable.

`train_iris_controlled_v1_v1_1.py` fixes this by always evaluating TUNE selection with the full post-warmup objective while preserving the intended warmup schedule for FIT optimization.

This correction occurred before any production optimizer run and is mandatory for future full training.

## Authorized next command

After interrupting the slow full-cache process, from the package directory:

```bash
python run_iris_controlled_v1_open_pilot.py --epochs 8
```

Do not resume full 3248 cache until this pilot has been interpreted.
