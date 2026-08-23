# IRIS Architecture Discipline Audit — 2026-08-24

Status: `IN_PROGRESS__TRAINING_FORBIDDEN`
Branch: `audit/iris-architecture-discipline-20260824`

## Executive finding

The scientific reframing is sound, but the executable Controlled V1/M256 path drifted away from its own research lineage in resolution, fine-descriptor role, evaluator units and checkpoint/evidence-consumer semantics. The completed M256 result therefore localizes useful learner evidence but cannot authorize scale-up.

The repair is not a patch to the 128-grid matcher. A clean `experiments/iris_single_pose_v2/` path is being created from the canonical contracts and preserved research evidence.

## Authorities reviewed

Product/problem: `canonical/PRODUCT_CONTRACT_V1.md`, `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`, G0 contract, native-1024 contract and Unified Master Corpus contract.

Research transfer: the 2026-08-21 correspondence study reviewed RoMa v2, MASt3R, TAPIR/TAPNext, CoTracker, XFeat, DINOv2/DINOv3 and DUNE. Its central finding was alignment of correspondence objective, spatial resolution and inference use. It proposed D1 observation-level matching, D2 coarse/fine role separation, then D3 query-conditioned local refinement only if a hard tail remained.

Post-study experiments: D1 directionally supported; G2 reciprocal/cycle promoted; D2 local precision improved but global ranking/mean/tail behavior failed, so global authority was falsified and local retained-top-k role retained. M4 audit preserved the set-valued observable substrate direction and left E0-E5 downstream sufficiency open.

## Mismatch ledger

| ID | Intended/authority | Controlled V1 executable | Audit verdict | V2 action |
|---|---|---|---|---|
| R1 | native 1024 primary, 512 matched control, 128 historical | 512 source -> 256 model -> fixed 128 matcher | FAIL | resolution-parametric model/matcher; 1024 primary |
| R2 | high-res local path + pooled global context | cross-view f16 with learned `max_w=32` | FAIL | fixed pooled global context + full local f16 |
| R3 | coarse global, fine local retained-top-k | Z_fine included in global RRF fusion | FAIL | global admission Zc/P only; Zf local only |
| R4 | fine local objective | Z_fine global multi-positive NCE | FAIL | local offset/lattice classification only |
| R5 | observation-level D1 matching, no early view pooling | partial multi-positive loss | PARTIAL | multi-positive + bidirectional pair matching + hard negative |
| R6 | explicit native-pixel localization | normalized fixed-grid tolerances | FAIL | exact coarse cells + native-1024 pixel error |
| R7 | geometry U vs match ambiguity distinguished | one U_geo reused by singleton heuristic | FAIL | U_geo geometry-only; ambiguity from candidate evidence until calibration |
| R8 | no premature singleton | heuristic singleton pre-calibration | FAIL | top-k hypotheses; singleton disabled |
| R9 | physical IRIS firewall | master geometry file can include rig fields; prep merely ignores them | PARTIAL | stage physical geometry-only allow-list |
| R10 | dense native truth | 512 geom samples/view, <=384 tracks | FAIL for primary claim | 4096-class/dense truth target; sparse path control-only |
| R11 | G0 absolute/tail metrics | relative improvement/mixed means dominate | FAIL | P/N absolute tails, spread, native-pixel tails |
| R12 | style policy fixed end-to-end | training random style, checkpoint cel-only, final bi-style | FAIL | one frozen bi-style selection/eval policy |
| R13 | observable checkpoint selection | mixed `val['total']` loss scalar | FAIL | prereg observable metric selection key |
| R14 | sparse/oracle ceiling wording | exact-P sparse candidate top4 overread as dense search | FAIL wording | preserve as candidate-conditioned ceiling only |
| R15 | one active executable authority | many V1/v1.1/v1.2/mini launchers | FAIL hygiene | V1 historical; V2 sole active path |

## Legitimate M256 evidence

The run is preserved because P/N learned strongly from random initialization, coarse/fine fields learned non-random correspondence structure, both styles behaved similarly, and the observation-derived candidate universe exposed a real consumer/localization gap. It did not prove native-1024 quality, final fine-local matching, calibrated ambiguity, SurfaceBuilder closure or an information limit.

## V2 decisions

1. resolution-safe shared encoder at R=256/512/1024;
2. full-resolution local f16 reasoning;
3. fixed pooled cross-view context with known yaw and continuous x encoding;
4. P/N/U_geo/Z_fine at R/2, Z_coarse at R/8;
5. no unproven hard P clip;
6. Z_coarse global high-recall objective;
7. Z_fine local-only objective and inference role;
8. global basin admission Z_coarse + predicted-P rescue;
9. no global Z_fine ranking;
10. no singleton until calibration;
11. explicit coordinate authority in `coords.py`;
12. evaluator errors in native authority pixels and object-space tails.

## Synthetic code preflight already passed

- V2 sources compile;
- exact align_corners=False pixel/grid round-trip at 128/256/512/1024;
- model forward shape checks at 256/512/1024;
- loss forward/backward finite;
- adversarial matcher role test proves a globally perfect far-away Z_fine match cannot enter when its coarse basin was not admitted;
- native-pixel metric unit test reports an exact 4-pixel displacement as 4.0;
- exact coarse-cell containment test.

## Remaining blockers before mini training

- [ ] verify committed V2 bytes against preflighted local sources;
- [ ] physically mark V1 executable folder historical/no-run;
- [ ] add real-corpus geometry-only staging + dense cache prep;
- [ ] run selected-asset no-optimizer corpus census: P range, N, raster/camera, firewall, split, density, style lineage;
- [ ] GPU memory/throughput preflight at 256/512/1024 without optimizer updates;
- [ ] freeze mini membership, metric panel and checkpoint selection key;
- [ ] create prereg only after all above PASS;
- [ ] only then create one Colab Run-All notebook.

Until every item is closed, training remains forbidden.
