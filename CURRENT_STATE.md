# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `R0_R3_REPRESENTATION_ONLY_CI96_FROZEN__REAL_RUN_NEXT__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Read this first

This file is the single continuation authority.

The preserved pre-audit branch is `g0-g1/single-pose-geometry`. Controlled V1/M256 executables are historical/no-run. The sole active candidate implementation is `experiments/iris_single_pose_v2/`.

**NO optimizer run is authorized.** The immediate next scientific executable is the real frozen 256-asset R0-R3 Representation Authority measurement, followed by a separate canonical interpretation.

## Final pre-result execution freeze

Current authority:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_EXECUTION_FREEZE_CI96_20260824.md`

CI69 and CI86 are superseded at the apparatus layer only. No R0-R3 scientific result was opened under either launcher.

CI86 reached representation-only staging `256/256`, then failed before cache generation because its uploaded execution bundle omitted the local dependency `coords.py` required by `geometry.py`. This was a packaging/CI-coverage failure, not a representation result.

### Exact code authority

- exact code-bearing head: `2f1b8ff199f0c147b9641ff2ec2cd22f56c67cb5`
- workflow: `IRIS V2 Preflight`
- run: **#96**
- run ID: `32677490168`
- conclusion: **SUCCESS**
- artifact ID: `9503132948`
- artifact: `iris-v2-r0-r3-execution-bundle-v3`
- artifact ZIP SHA-256: `488e053bbdcd84eb846b3cc856984f70c69f026220256c2872fa44eea35fb7ee`

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI96.zip`

Drive file ID:

`1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`

Prepared launcher:

`RealSaS_IRIS_V2_R0_R3_Representation_Only_CI96.ipynb`

Notebook SHA-256:

`d0cb6260bcd021a1286d2ad16e18f9f184f3122626d8cea215b04fcedccaebcd`

## CI96 bundle closure

CI96 changes apparatus/provenance only; frozen R0-R3 scientific definitions are unchanged.

The uploadable bundle now contains the complete local execution dependency set:

- `build_representation_seed_v1.py`
- `stage_representation_authority_v1.py`
- `prepare_representation_cache_v1.py`
- `audit_representation_stage_cache_v1.py`
- `representation_authority_study_v1.py`
- `compact_representation_handoff_v1.py`
- `geometry.py`
- `coords.py`
- `run_representation_authority_v2.py`

CI no longer tests only the repository checkout. Before artifact upload it now:

1. scans bundled Python imports against repository-local modules and fails on any missing local dependency;
2. copies the representation-only synthetic smoke into the bundle directory and runs it from there;
3. compiles the isolated bundle;
4. verifies `SHA256SUMS.txt`;
5. removes all test `__pycache__` artifacts;
6. asserts the exact final bundle file set before upload.

Run #96 passed the isolated dependency-closure/synthetic-smoke step and the exact bundle-content assertion.

## Representation-only execution boundary

Stage profile:

`representation_authority_geometry_only`

R0-R3 stages/consumes only:

- sanitized `vertices/faces`;
- native-1024 `raster_authority.npz`;
- `camera.json`.

It does **not** stage, decode, hash, or consume `cel_clean` / `ink_cel` images. RGB remains mandatory later for the actual IRIS learner; it is excluded here because this gate measures exact legal P/P+N representation authority, not image-to-evidence learning.

The physical/fingerprint boundary also excludes hidden rig/mechanics fields from `primary_geometry.npz`.

The dedicated representation-data preflight proves:

1. deliberately invalid RGB decoys are never read;
2. RGB mutation does not change the representation-stage fingerprint;
3. hidden rig/mechanics mutation does not change the legal fingerprint;
4. raster-authority mutation does change it;
5. stage contains only legal geometry/raster/camera files;
6. truth cache builds with `rgb_consumed=false`;
7. continuous projected track truth remains exact;
8. auditor rejects staged image/unexpected-file leakage.

All architecture/coordinate/matcher, learner firewall, observable evaluator and R0-R3 semantic preflights also pass on the same exact head.

## Controlled corpus authority

- canonical selected: **3993**;
- excluded: 56 repair-pending current-A-render mismatches, 6 active non-Basis shape-key assets, 1 all-eight-view blank asset;
- controlled corpus: **3930**;
- FIT: **2935 OPEN**;
- TUNE: **313 OPEN**;
- CAL: 246 SEALED;
- DEV: 270 SEALED;
- EXTERNAL_HOLDOUT: 166 SEALED;
- open FIT+TUNE: **3248**.

Frozen source authorities:

- `IRIS_CONTROLLED_V1_SPLIT_FREEZE.json` SHA-256 `9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66`
- `CANONICAL_VARIANT_SELECTION.json` SHA-256 `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

CAL/DEV/EXTERNAL remain sealed. No resplitting and no post-apparatus asset substitution.

## Frozen confirmatory panel

- **256 OPEN assets**;
- at most 48 queries/asset: 16 adjacent, 16 skip-one, 16 opposite;
- split: 230 FIT / 26 TUNE;
- provider: 245 Objaverse / 7 Quaternius / 4 KayKit;
- strongest capability: 217 Arachne / 33 Geppetto / 6 IRIS;
- 12 non-empty `(provider, split, strongest-capability)` strata;
- ordered asset-ID-list SHA-256: `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`.

## Frozen R0-R3 semantics — unchanged by CI96

- SAME-locus truth = exact physical P within `0.003` canonical units;
- R0 = exact P;
- R1 = exact P+N;
- R2 P sigma = `0, .0005, .001, .0025, .005, .01`;
- R3 = full 6x5 P/N grid with N angles `0, 5, 10, 20, 40` degrees;
- P+N diagnostic score = `dP + 0.05*(1-cos N)`, fixed before results and never swept;
- source and target observations independently perturbed;
- deterministic Philox noise;
- exact tangent-plane normal rotation;
- top1/top4/top8 = physical set-containment, not cached-index accuracy;
- reciprocal + three-view cycle use legal candidate universes;
- family physical-error median/p90/p95 tails required;
- `nearest_non_equivalent_physical_gap` = geometry-only confusability diagnostic, not semantic symmetry or promotion threshold;
- mean-only promotion forbidden.

R4 remains intentionally undefined. If R0-R3 evidence motivates it, a separate formulation prereg must be frozen before any R4 result is opened.

## Active execution

Preferred launcher:

`RealSaS_IRIS_V2_R0_R3_Representation_Only_CI96.ipynb` -> **Run all**.

CPU runtime is sufficient; GPU is not used.

If the same Colab runtime that completed CI86 representation staging is still alive, the CI96 notebook may salvage that local stage **only after** verifying the CI86 seed, frozen panel digest, identical stage-builder SHA, all 256 stage markers, exact staged file sets/hashes, physical firewall, and absence of RGB. Otherwise it stages fresh.

The runner/notebook may not decide P sufficiency, R4, SOI-2 or training authorization.

A successful measurement may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

with:

- `optimizer_steps = 0`;
- `training_authorized = false`;
- `sealed_splits_opened = false`;
- `rgb_staged = false`;
- `rgb_consumed = false`.

## NEXT EXECUTABLE STEP

**Run the CI96 representation-only Colab notebook. Do not train.**

After the persistent result exists, inspect the compact handoff + full result + hard-tail and write one canonical interpretation without changing frozen definitions.

Only after representation interpretation may the sequence proceed to optimizer-zero production-width GPU capacity preflight, frozen mini membership/style/checkpoint key, a new mini prereg, and only then a learner optimizer run.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
