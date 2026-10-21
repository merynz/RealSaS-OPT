# IRIS H-GEO-P3-EFREQ Decision Ledger — 2026-10-20

## Trigger

H-GEO-P3 completed cleanly on exact source `e3c48678f3f563bb60624278e206f4ba8f71c272`, 4800 carrier steps, exact P1 triangle truth, and byte-exact shared V8 preservation. The independent court passed locked distance, normal, near-surface, local-alignment and shared-preservation gates but failed endpoint p95 and the required endpoint improvement versus P2.

Sealed P3 result authority: `docs/research/IRIS_H_GEO_P3_RESULT_20261020.json`.

## Question

Why does closest-surface endpoint tail precision remain above the locked `0.075` IRIS p95 bar after P3 gains in distance, normal and near-surface evidence?

This is a zero-optimizer diagnostic. It does **not** train P4, SURF-A2 or TESSA.

## Locked factors

The audit separates three primary failure families:

1. **Target ambiguity / discontinuity** — nearest-surface vector target changes abruptly near competing non-adjacent sheets or medial regions.
2. **View-evidence ceiling** — endpoint-tail triangles are under-resolved or weakly constrained by the frozen 8-view evidence. The audit uses geometric orthographic coverage/footprint proxies and does not claim true occlusion visibility.
3. **Carrier bandwidth ceiling** — exact target vectors change at fine spatial scales faster than P3 `H_geo` responds, despite 128² residual refinement.

The audit also records P3 internal utilization: 128² refiner residual RMS/high-band spectrum and continuous-position residual magnitude.

## Query court

- split: exact P1 held-out triangle dataset;
- eligible queries: exact distance `<=0.12` IRIS;
- tail: top 10% P3 training-aux endpoint error within each subject, capped at 384;
- control: deterministic distance-stratified median-error match;
- aux head is **only a tail selector**, never qualification authority;
- perturbation scales: `1/256`, `1/128`, `1/64` IRIS;
- P2 and P3 carrier response is measured with per-channel holdout whitening;
- authored teacher meshes are reloaded and normalization is replay-verified before triangle-level analysis.

## Classification

The preregistered output is one of:

- `TARGET_AMBIGUITY_DOMINANT`
- `VIEW_EVIDENCE_DOMINANT`
- `CARRIER_BANDWIDTH_DOMINANT`
- `MIXED`
- `UNRESOLVED`

No classification automatically authorizes training. Each only selects the next preregistration family.

## Hard prohibitions

- optimizer steps: `0`;
- probe optimizer steps: `0`;
- parameter mutation: forbidden;
- checkpoint write: forbidden;
- threshold relaxation: forbidden;
- teacher truth in inference forward: forbidden;
- TESSA invocation: forbidden;
- SURF-A2 authorization: false;
- TESSA authorization: false.

## Expected artifact

`H_GEO_P3_EFREQ_AUDIT.json` plus `H_GEO_P3_EFREQ_QUERY_TABLE.csv`, both hash-bound to the frozen source and exact P3/P2/V8/dataset/teacher authorities.
