# Representation Authority Execution Freeze — CI104 — 2026-08-24

Status: `FROZEN_FOR_REAL_R0_R3_RUN__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Authority

- branch: `audit/iris-architecture-discipline-20260824`
- exact code-bearing head: `7d99eef46d2c071eca1d883e0c916bf4adecda35`
- workflow: `IRIS V2 Preflight`
- run: `#104`
- run ID: `32679394221`
- conclusion: `SUCCESS`
- artifact ID: `9503678775`
- artifact name: `iris-v2-r0-r3-execution-bundle-v3`
- artifact ZIP SHA-256: `bc1e733049fafcd01f933d26c309f7fb5a9cff0d2abd04124219c4ea9ba39473`
- Drive mirror: `RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI104.zip`
- Drive file ID: `1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`
- launcher: `RealSaS_IRIS_V2_R0_R3_Representation_Only_CI104.ipynb`
- launcher SHA-256: `c7cd5cf48b2273df8dba82b4dc6717e55555c6c8942927581670e53cbd033680`

## Why CI104 supersedes CI96

CI96 completed representation-only source staging `256/256` but failed at cache construction before audit/study/result because partial visibility exposed a cache-builder indexing bug:

`track_err[:, v] = err[ok]`

attempted to write the visible subset into the full track column. No R0-R3 result was opened.

CI104 changes apparatus implementation only; the frozen panel, candidate universe, SAME tolerance, R0-R3 arms, scoring, noise, reciprocal/cycle semantics and thresholds are unchanged.

Correct semantics:

`track_err[ok, v] = err[ok]`

Hidden/non-witnessed tracks retain the initialized `+inf` error.

## New regression closure

CI104 adds `representation_cache_visibility_preflight_v1.py`, which constructs genuine partial visibility and requires:

- some tracks visible and some hidden in the same view;
- `track_visible.shape == track_surface_error.shape`;
- visible witness errors finite;
- hidden witness errors `+inf`;
- `coords.py` included in the cache-builder semantic fingerprint.

The same regression is executed again inside the isolated uploadable bundle before artifact upload.

CI104 also retains the CI96 bundle closure checks:

- complete local dependency closure;
- isolated compile/import;
- representation-only synthetic stage/cache/audit smoke;
- SHA256SUMS verification;
- exact final bundle file-set assertion;
- no test `__pycache__` or undeclared artifact leakage.

All CI104 steps passed.

## Frozen scientific semantics

Unchanged from the parent prereg/addendum:

- 256 OPEN assets; 230 FIT / 26 TUNE;
- ordered panel digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`;
- SAME-locus tolerance `0.003` canonical units;
- R0 exact P;
- R1 exact P+N;
- R2 P sigma `{0,.0005,.001,.0025,.005,.01}`;
- R3 full 6x5 P/N grid with N `{0,5,10,20,40}` degrees;
- fixed P+N coefficient `0.05`;
- deterministic Philox source/target perturbations;
- physical set-containment top1/top4/top8;
- reciprocal + three-view cycle;
- family/provider/capability/component/support/pair-category tails;
- geometry-only nearest-non-equivalent physical gap diagnostic;
- optimizer steps `0`;
- sealed splits unopened;
- training authorization `false`.

## Execution boundary

Representation staging consumes only sanitized `vertices/faces`, native-1024 `raster_authority.npz`, and `camera.json`. RGB is not staged or consumed.

If the same Colab runtime still contains the completed CI96 local stage, the CI104 launcher may reuse it only after full seed/panel/marker/file-set/file-hash/firewall/no-RGB verification. Otherwise it stages fresh.

## Next authority

Run the CI104 notebook. A successful run may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

Then write the canonical R0-R3 interpretation. Do not train before that interpretation closes the representation gate.
