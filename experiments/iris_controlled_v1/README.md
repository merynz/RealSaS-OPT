# IRIS Controlled V1 — HISTORICAL / DIAGNOSTIC LINEAGE

Status: `SUPERSEDED_AS_ACTIVE_EXECUTION__DO_NOT_RUN`
Date retired from active path: 2026-08-24

This directory is preserved for provenance. It contains the Controlled V1 learner, representation ceiling, Open64/M256 experiments, sparse/dense matcher variants, preregistrations and launchers that produced useful evidence.

**It is not an active training authority. Do not execute any `launch_*`, `run_*`, trainer or notebook in this directory during the architecture audit.**

The sole candidate active implementation path is:

`experiments/iris_single_pose_v2/`

The reason for retirement is scientific-role drift identified in `audit/IRIS_ARCHITECTURE_AUDIT_20260824.md`, including:

- native-1024 contract reduced through 256 model input and fixed 128 matcher lattice;
- normalized-grid localization tolerances with ambiguous native-pixel meaning;
- `Z_fine` reused in global rank fusion despite D2 having been falsified as global authority and retained only as local evidence;
- geometry U reused as match/singleton ambiguity heuristic before calibration;
- sparse/dense evaluator lineages and multiple v1/v1.1/v1.2 launchers remaining visually active at once;
- checkpoint/evaluation policy not equal to the final intended observable metric panel.

Historical results remain valid **only for the exact questions they actually measured**. In particular, the M256 run retains real learner evidence but cannot authorize native-1024 scale-up or an information-limit claim.

Nothing in this folder is deleted merely because its interpretation was superseded. Git history preserves the full original README and execution documentation.
