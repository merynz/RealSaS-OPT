# Corpus370 V18f — full original-source physically deleted editable-graphics benchmark

Date: 2026-11-05. Status: **EXPERIMENT EXECUTION BEGUN; INDEPENDENT RESULTS PENDING**.  
PR: #343, branch `corpus370-v18b-reactive-stain-source-edit-court-v1`; run: `42114401233` (https://github.com/merynz/RealSaS-OPT/actions/runs/42114401233).

## Immutable research lineage and measured previous results

1. V18a fixed mechanical-reactive staining, exact original-source run `42097722762-1`: 3/3 real sources + independent court SHA PASS, but not graphics ownership.
2. V18b 52-arm/4-view original-source edit run `42100552844-1`: physical source/soup/no-op parity 4/4 on Cat, Triceratops, Cowfish, but Cat/Triceratops SOUP_EMPTY failed because source triangles were assigned transparent shader rather than physically removed; Cowfish 24/24 original single-face references unobservable.
3. V18d independent original pixel retrospective `42105598958-1`: Cat 14/24 and Triceratops 24/24 single-triangle observable image consequences, versus Cowfish 0/24. Original RGB/source/soup parity worked on every view.
4. V18d causal original Blender renderer witness `42106976442-1`: both Cat and Triceratops independent source courts PASS, observed alpha remnants vanish when Cycles transparent bounce depth raised or all render soup geometry truly hidden. No artist/learner labels.
5. V18e true physical mesh face-delete canary `42111807283-1`, **3/3 original jobs PASS**, exact independent source court SHA readback:
    - Cat `72c80db1bd73b39f6bdd0cfc51789a7bf76de0b8c8f4c71f236d1d67193ea030`: 2/4 camera views have observable original single-face reference. Physically empty scene has zero alpha violation in all four views. At 0.1, group isolate effects 10/0/10/0 pixel instances, group remove 0/0/0/0. Original source/renderer parity PASS.
    - Triceratops `fb1fb612c89cf3b79098ab84f800d71dc732087ad2af1558061ab9d03d026d4b`: 4/4 camera views have observable single-face reference (isolation), source/no-op/empty all PASS; 0.1 group-isolate effect 2552/5331/2549/5330 pixels, group-remove effect 2109/5272/2125/5242 pixels across view indices 0/2/4/6. **These are physically observed effect magnitudes, NOT declared artist editing errors.**
6. No current graphics_owner_authorized, iris_training_authorized, tessa_training_authorized or materialization_authorized. No actual artist SAME/DIF drawable labels.

## Experimental motivation, frozen design

The old 52-arm V18b RGBA array was contaminated by the wrong mechanism: visibility via shader transparency, subject to a finite Cycles transparency-path limit. V18e proved a source-faithful counterfactual mechanism in both previously failing sources. V18f **reuses the exact V18a groups, V11 locked blind face targets, V14 original source/actions/cameras and Blender renderer** but applies deletion by physically removing polygons from a copy of the original posed triangle-soup mesh with preserved source UV and material slots.

**Frozen before observation**:
- Real sources: original Cat ordinal 0 and original Triceratops ordinal 1, with exact V18a/V14 original source SHA pinned.
- Six previous V11 blind target ordinals `0,2,4,6,8,10` (not selected from results).
- Three V18a original similarity thresholds `0.045, 0.10, 0.18` (not optimized on heldout).
- Original third authored source pose (pose ordinal 2), cameras `0,2,4,6`, 192x192, 32 Cycles samples.
- **52 render arms per camera, 208 genuine renders per source, total 416**; each arm acts on the same source geometry without accumulating removals.
- Every geometry edit has independently auditable exact face census: N/full/no-op = N, empty 0, target removal N−1, isolation 1, group removal N−G, group isolation G where G is recalculated by the independent court from frozen source-face labels.
- Compare all per-target, per-threshold raw edit consequences, alpha, premultiplied RGB and visible support from actual Blender output. Never count an unobservable single-face reference as a correct/incorrect grouping.

## A specific scientific caution

Single-triangle reference is a **physical diagnostic probe**, not the artist's editable part. Extra group-effect pixels relative to this tiny reference are not automatically "collateral" or a cannot-link relation. V18f intentionally renames the diagnostic as `group_minus_single_face_proxy_effect_pixels` and reverse `single_face_proxy_minus_group_effect_pixels`. Such measurements may reveal giant groups or unobservable edits and help design **new intervention hypotheses**, but they do not yet create attractive/repulsive graphics ground truth. The representation may require overlaps and partial-order layers rather than a one-to-one partition.

For the first full results, compare:
1. Render source↔V14, source↔soup and soup↔no-op parity and truly physically empty alpha across all 4 views.
2. Exact independently source-courted geometry face-count invariants for all 52 arms and four views.
3. Number of observable original single-face reference cases per source and their missing/under/overlap effects.
4. Candidate group K and group size distribution, per-threshold heldout effect magnitude and view stability, **without ranking by K reduction alone**.
5. Identify and retain `UNKNOWN` whenever source edit is invisible or artist grouping cannot be identified.

Next independent step after true instrument validity is a graphics-specific signed-relation investigation: propose ATTRACT/REPEL/UNKNOWN under **independent declared edit intents**, use Mutex Watershed/Lifted Multicut as potential graph optimizers, and compare on genuinely heldout edit consequences, not pixel similarity. IRIS's future anonymous head can be considered only after the teacher/output target and consumer constraints are qualified.

## Reproduction and exact source paths

- `tools/truth_foundry/blender_corpus370_v18f_full_physical_edit_benchmark.py`
- `tools/corpus/court_corpus370_v18f_full_physical_edit_benchmark.py`
- `tools/corpus/hosted_corpus370_v18f_full_physical_edit_benchmark.sh`
- `tests/corpus/test_corpus370_v18f_full_physical_edit_benchmark.py`
- `.github/workflows/corpus370-v18f-original-full-physical-face-edits.yml`

The workflow requires SHA-readback of both real V18e canary receipts and all frozen V18a original-source independent courts before Blender runs. Its new independent result must also SHA-readback from private Drive. At writing, V18f contracts PASS, original Cat and Triceratops workers IN_PROGRESS. All original failed experiments remain immutable and inspectable. Main and PR #327 are unchanged.
