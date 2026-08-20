# R6 / Rank-2 Continuity Checkpoint — 2026-08-20

## Authority state

- Branch: `agent/n1d-observable-functional-audit-v2-20260820`
- R5: **diagnostic-only / canonicality invalidated**. Do not rehabilitate or tune toward R5/e00 outcomes.
- R6.1: clean replication target is **untouched `e01`**.
- Stale local file named `INPUT_MANIFEST_R6.json` that still points at `e00`: **FORBIDDEN / non-authoritative**.
- No R6 outcome has been executed after this checkpoint.
- Rank-2 R result cannot become canonical until R6 clean replication closes.

## R6.1 source pre-freeze authority

Repository directory: `research/n1d/observable_functional_audit_v2_20260820/r6_source/`

- `BUILD_R6_1_SOURCE.py` blob SHA: `b93091e53863b45acfdb3f0952df2d06e1b2ab20`
- `R6_1_BUILD_RECIPE_CORRECTION.md` blob SHA: `73de43902bfc8b4b46cf5b3da8a00ac5cf3e6f3f`
- Original `BUILD_R6_SOURCE.py` blob SHA: `763c57aad9ff0e907cd4d2c7b18dd1f0e2450207`
- Correction is build-recipe/import-loader only; frozen base evaluator SHA guard remains authoritative. No scientific objective/threshold change is authorized.

## Exact R6 e01 input population

Family IDs, fixed before truth-open:

`[09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290]`

All eight are `dev`, `COMPLETE`, contain `e01`, and their family-manifest declared `content_sha256` values match `EPISODE_INDEX.json` exactly.

- Corpus folder ID: `1-qseF9kjusXIN9ftYIuaBkqNMXVHLnIx`
- EPISODE_INDEX Drive file ID: `1XjdMi_-C8Kkuk0TFzeiLZt2IwYot-OpJ`
- EPISODE_INDEX declared content SHA: `907bca7323c8e7b3832ad276d44af9a5733e606b12c763a492775387f4afb765`
- EPISODE_INDEX actual file SHA-256: `e1a6a3e00fc3ac38858e3ac649777a4025dba442a4ced5efd916a37200c69362`
- `sealed21=CLOSED`; `external10=CLOSED`.

Canonical input manifest to use next:

- `r6_input/R6_E01_INPUT_MANIFEST_V2.json`
- logical content SHA-256: `eea102387cd052d7580eab13196f2dd352613783f99ef7b6a2c5f647766314d7`
- serialized file SHA-256: `15364eefafcbe8a1bd6d53813ac5579ebaf509b32ccaecace91cc308b774076a`
- GitHub persistence commit: `4d97fd4ee70b736814b11fc845cc41477cc23046`
- GitHub blob SHA: `fc033d175a07fc2649e09ca24ca25c8f912703d2`

Raster hash authority is **not duplicated** in R6 manifest. Each raster must be checked against the `poseA_hashes` and `episodes[e01].poseB_hashes` inside the corresponding family manifest, after that family-manifest file itself passes its recorded byte SHA.

## Input-role firewall

Before truth-open, sidecar access is limited to:

- `camera_center`
- `camera_half_extent`
- `family_id`

Mechanics input is limited to Pose-A/Pose-B rasters plus outputs of frozen observable components. Teacher exact `surface_points_*`, `surface_normals_*`, `surface_visibility_*`, `surface_xy_*`, dense weights, skeleton, and intervention labels are forbidden as mechanics input. Those teacher fields may only be opened evaluator-side after the candidate/world state is frozen, for teacher-nearest labeling/evaluation as preregistered.

This firewall is specifically intended to prevent recurrence of the historical 0/48 audit's exact-N/V decoration leak.

## Required next gates

1. `R6_E01_INPUT_MANIFEST_V2.json` is persisted and read-back verified on GitHub. Mirror to Drive/Library only as a non-authoritative replica if used.
2. Materialize e01 bytes from the exact locators; verify family-manifest byte SHA, then PoseA/e01 PoseB raster SHA and sidecar SHA. Any mismatch => fail closed.
3. Build frozen R6.1 source from the guarded base evaluator; run source/import/static tests; persist source bundle + member hashes + prereg **before execution**.
4. Execute R6 clean replication on e01 without truth-open mechanics leakage.
5. Only after R6 closure, start new R implementation gate: first historical rank-3 behavioral parity, then full-64 set-valued rank-2 global-world solve.

## Scientific question for later rank-2 experiment

The rank-2 experiment is not "does one chosen XYZ equal teacher geometry?" It asks whether the frozen global objective `R`, when allowed to choose within each carrier's observation-consistent candidate set `H_i`, can select a **jointly coherent observable world** whose downstream mechanical evidence/compile behavior is acceptable, without teacher geometry entering the solve. Teacher is evaluator-only.
