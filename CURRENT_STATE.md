# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__MINI_EXTRACTABILITY_V1_FROZEN__CI_RELEASE_REQUIRED_BEFORE_NOTEBOOK_OPTIMIZER`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

CI104 Representation Authority is complete and canonically interpreted in `experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_CANONICAL_INTERPRETATION_CI104_20260824.md`.

Canonical representation result: `P_GEOMETRY_SUFFICIENT`. R4 and SOI-2 are not opened.

## CI104 result that authorizes the learner target

Exact P on the frozen 256 OPEN panel / 12,288 queries: top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical-error median/p90/p95/max=0/0/0/0; family tails perfect. R2 P-noise remains the descriptive learner-accuracy reference.

## Normal policy after CI104

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`: `track_n_view` is a nearby view-specific raster witness normal, not exact differential N at the same persistent P locus. Mini V1 keeps legal `geom_n` as observation-local dense orientation target and reports N angular error, but forbids N from correspondence admission/ranking and checkpoint selection. Production correspondence stays Z_coarse+P global admission, Z_fine local refinement.

## Mini Extractability / Generalization V1 — FROZEN BEFORE OPTIMIZER

Prereg: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_GENERALIZATION_PREREG_V1.md`

Membership: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_MEMBERSHIP_V1.json`

Frozen authorities:

- CI104 representation seed byte SHA-256 `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af`
- source panel ordered-ID digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- mini membership canonical-JSON SHA-256 `4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055`

Frozen roles: FIT_TRAIN128; FIT_SELECT32; TUNE_FINAL26; unused FIT70; CAL/DEV/EXTERNAL_HOLDOUT unopened.

Input resolution=256. This mini tests extractability/generalization feasibility, not final1024 product precision.

Training: 16 epochs, AdamW3e-4, microbatch1, grad accumulation4, FP16, geometry warmup epochs1-3, correspondence/local refinement from epoch4, candidate checkpoints4/8/12/16. Checkpoint selection uses FIT_SELECT only. TUNE_FINAL is evaluated once only after best-checkpoint freeze. cel_clean and ink_cel are evaluated separately and equal-macro.

Allowed labels: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

## Training apparatus fixes before release

Image-bearing learner cache now uses `track_err[ok,v] = err[ok]`; hidden entries remain +inf; cache fingerprint binds prepare_cache.py+geometry.py+coords.py. Z_fine local loss uses deterministic position-uniform track thinning rather than first-64 prefix. Both are mandatory CI regressions.

## GPU release gate

The closed 256/512/1024 architecture executable preflight is not repeated. Before scientific optimizer step1, exact mini artifact must pass real CUDA R=256 production-width full post-warmup loss forward/backward, finite gradients and AdamW two-moment memory accounting, with zero scientific optimizer steps. OOM/non-finite => `APPARATUS_CAPACITY_REOPEN_REQUIRED` and no optimizer.

## NEXT EXECUTABLE STEP

CI must compile/test the committed apparatus, run learner partial-visibility + mini semantic regressions, build/test exact `iris-v2-mini-extractability-bundle-v1` outside the repo checkout, then upload it. Only a successful artifact may be mirrored to Drive and bound into the Colab GPU notebook. Notebook first runs zero-step GPU capacity release, then the frozen mini iff capacity passes.

Do not change membership, thresholds, checkpoint key or TUNE policy after optimizer begins.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
