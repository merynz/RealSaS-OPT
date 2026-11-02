# Truth139 source inspection and selective revalidation

Truth139 remains the base artifact inventory. Preserve existing bytes and
revalidate each family against its actual dependencies. Do not re-render or
re-materialize every family merely because the corpus, court, or repository SHA
changes. Do not reuse an output whose source, coordinate frame, addressing,
producer semantics, or relevant policy has changed.

The new requirement is source/teacher inspection for every subject, including
the base. A mesh-object count proves neither modular semantics nor closed solid
geometry. A closed output may have multiple components and nonzero genus; zero
boundary/nonmanifold edges alone does not prove a valid embedded solid.

## Existing evidence and current measurements

On 2026-11-01 the historical readiness index and full source-cache manifest each
enumerated the same 139 subject IDs. All 139 source files were recovered and
matched the manifest's expected and actual SHA-256 values. These glTF files have
no external buffer/image URIs. The original base IDs and artifact hashes were
preserved, including appearance variants; no subject was dropped by this pass.

The new raw-bind structural pass measured all 139 sources:

| Measurement | Subjects |
| --- | ---: |
| Source byte identities verified | 139 |
| Raw-bind structural inspection completed | 139 |
| Multiple mesh owners | 53 |
| Open boundaries under owner-preserving exact-position quotient | 122 |
| Nonmanifold edges under that quotient | 21 |
| Edge orientation conflicts under that quotient | 3 |

These categories overlap. They are diagnostic observations, **not admission or
exclusion counts**. Qualified cross-owner equivalence can change the mechanical
surface interpretation. Active-scene/importer equivalence, rest/pose replay,
vertex links, self-intersections, nested/intersecting shells and explicit solid
assembly semantics still require their own evidence. The remaining 17 rows are
not automatically qualified solids. The raw bind reader is not a posed skin
replay or an importer-equivalence court.

The previous geometric-repair receipt reports 17 completed and 122 failed rows.
At least the inspected failure is a `face_component_id` court mismatch. That
receipt must not be reinterpreted as a 17-subject solid-admission result.

## Two concrete defects corrected

1. The component geometry producer explicitly states that union occupancy is
   not minted. Its training adapter nevertheless exposed those rows as global
   sign targets while dropping `signed_component_id`. The adapter now retains
   component addresses and rejects `global_sign_rows()`. Component evidence can
   still serve a separately conditioned component consumer.
2. `evaluate_candidate` replaced every failed measured court with PASS when
   `kind == SOURCE_FALLBACK`. That override is removed. The V1/V2, V3, V4 and V5
   court paths now require all declared checks and independently check rest
   self-intersection for fallback and simplified candidates alike. Legacy metric
   rows without the measured-evaluation contract are rejected for re-evaluation;
   their forced PASS booleans cannot serve as current proof. The V1 xyz
   spatial index is explicitly three-dimensional.

This does not prove that either defect alone caused the historical training
behavior. The inspected historical Adventurer geometry manifest, for example,
already records zero valid global-sign queries. Its separate RGB manifest binds
eight image hashes; its selected TESSA teacher is a source fallback. The sign
coverage gap, RGB evidence and teacher-court defect are distinct facts.

The regression uses nested boxes: the nearest inner component gives +0.2 while
the union solid gives -1.8 at the same query. Correct global occupancy multiplied
by distance to all raw surfaces would also give the wrong magnitude. Sign and
distance must describe the same explicitly chosen solid boundary.

## Reuse decisions

The machine-readable policy is
`canonical/CORPUS_TEACHER_INSPECTION_AND_REUSE_POLICY_V1.json`.

| Existing evidence | Default next action |
| --- | --- |
| Source/import, cameras and observation images | Preserve; verify exact source/resources, rest/frame/camera and importer equivalence |
| Surface UDF/samples and oriented normals | Preserve; revalidate surface identity, winding/frame and distance semantics |
| Global SDF/occupancy | Establish explicit solid scope and qualified boundary; regenerate only affected targets |
| TESSA teacher | Re-evaluate measured courts without fallback exemption; reuse mesh bytes only if current checks pass |
| RGB/appearance and measured ray evidence | Preserve; verify pose/camera/material/addressing and student-observability admission |
| Source intervention tensors | Preserve; recompute adjacency or carrier-bound descendants only if their inputs change |
| Authored rig, skin and animation | Preserve source evidence; independently verify bind/addressing and canonical replay |

A new court may certify unchanged bytes without rerunning a correct producer.
An invalidated producer/input requires regeneration of the affected output and
its actual descendants. Unknown legacy dependency identity calls for
revalidation, not automatic certification or indiscriminate regeneration.

The preparation tool verifies source bytes, inspects raw structure and keeps the
historical artifact index unchanged. It deliberately does not claim to have
reopened the historical NPZs or their complete parent graphs:

```bash
python tools/corpus/plan_teacher_truth_revalidation_v1.py \
  --source-manifest FULL_SOURCE_CACHE_MANIFEST.json \
  --prior-readiness V32_TRAINING_READINESS_RECEIPT.json \
  --source-root /mounted/source_root \
  --output-root /output/teacher_revalidation
```

For mounted artifact validation, retain the existing
`audit_truth139_family_inventory_v1.py` and the independent source/family courts.
Do not replace these with the new cheap preparation pass.

## Execution handoff and remaining work

This change is source review and read-only preparation. No Platform Attempt,
family admission, new TRAIN/UNSEEN split, optimizer or training run was created.
The Platform architecture map/impact commands could not complete in this
environment because Go is absent; dependency ownership remains unresolved rather
than being asserted empty. Static consumers of the changed adapter and the
shared TESSA implementation were inspected.

Before the materialization run, enter the real Platform Attempt against deployed
main and inspect its resolved reuse/execute scope. The notebook remains an
external compute worker. The final two-notebook goal remains: one Run-All truth
materializer with dependency-based reuse, followed by one IRIS–TESSA trainer
consuming its exact qualified consumer manifests and fresh lineage-safe split.

The expanded roster, solid/closed-target qualification, final per-consumer N and
those two end-to-end notebooks are not completed by this source-review change.
In particular, source-fallback rebind plus transferred source trajectories is
not a proof of a new candidate's independent skin/rig replay or watertightness.

Validation: 41 relevant CPU regression tests passed, covering component geometry,
independent courts, sign scope, source fallback failure, structural inspection,
source-drift rejection, historical-authority isolation and the xyz spatial index.
