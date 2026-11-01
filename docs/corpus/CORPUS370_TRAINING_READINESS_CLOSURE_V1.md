# Corpus370 training-readiness closure

Status: **OPEN — no training authority**

This work closes the 370-candidate corpus by subject × truth-family authority.
It does not treat a successful source import, a producer artifact, a historical
`training_authorized` flag, a publisher inventory, or a pack-level capability as
training readiness.

The candidate universe is exactly 370 source identities: 139 established Truth139
sources plus 231 expansion candidates. `Corpus400` remains a campaign/source-line
name and is not a claim that 400 subjects are admitted.

## Failure being corrected

Truth139 produced substantial source, geometry, observation, response and teacher
evidence, but the old readiness gate was too coarse. In particular, geometry
truth was allowed to coexist with zero valid global-sign rows while the corpus
could still be described as training-ready. Historical TESSA teachers also chose
`SOURCE_FALLBACK` without proving the current mechanical-topology consumer claim.

The closure therefore separates **evidence existence** from **consumer authority**.
Every model consumes a projection of one shared candidate universe. There is no
single universal `N subjects ready` count.

## Typed truth carriers

The following domains are intentionally distinct:

```text
SourceSurface
    exact authored/observed surface evidence

ExplicitSolid
    separately qualified solid/assembly for global occupancy/SDF only

MechanicalSurfaceTopology
    mechanically qualified topology consumed by TESSA

SkinCarrier
    exact carrier on which skin weights are authored/qualified

RigAnimation
    authored/qualified hierarchy, frames and animation

AppearanceAddress
    CAA / visibility / compositing addressing domain
```

They may refer to the same subject without sharing one vertex set. A source may
be useful for several families and masked for others.

## Authority matrix

The machine policy is
`canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json`.
The matrix builder is
`tools/corpus/build_corpus370_subject_family_authority_matrix_v1.py`.

Every subject receives one cell for each family:

1. `SOURCE_IMPORT`
2. `CAMERA_OBSERVATION`
3. `SURFACE_GEOMETRY_UDF`
4. `GLOBAL_SOLID_SDF`
5. `SURFACE_NORMALS`
6. `APPEARANCE_CAA`
7. `VISIBILITY_OCCLUSION`
8. `RAY_COMPOSITING`
9. `MECHANICAL_SURFACE_TOPOLOGY`
10. `CAUSAL_RESPONSE`
11. `RIG_HIERARCHY`
12. `SKIN_WEIGHTS`
13. `ANIMATION_KINEMATICS`
14. `TESSA_TEACHER`

Cell statuses are:

- `PASS`: a current independent court proves the exact family claim.
- `MASKED`: the family is absent/inapplicable; no loss is applied for that target.
- `FAIL`: a current court tested and rejected the claim.
- `QUARANTINED`: provenance, source identity or conflicting evidence is unresolved.
- `UNKNOWN`: no current court is bound. It is never training-consumable.

A missing target is never represented by numeric zero.

## Global SDF rule

Global signed supervision is only valid for an
`EXPLICIT_QUALIFIED_SOLID_ENTITY_OR_ASSEMBLY` with the convention
negative-inside / positive-outside / zero-boundary. The distance magnitude must
be distance to the actual boundary of that same solid.

The following are explicitly forbidden:

- component-local parity promoted to global sign;
- nearest-component sign promoted to global occupancy;
- union occupancy multiplied by raw-surface UDF and called exact union SDF;
- silent hole closing or silent decorative-geometry removal;
- missing sign rows encoded as zeros.

Open/modular source geometry therefore remains useful for exact source surface,
UDF, graphics, rig/skin, animation or mechanical truth when those families pass.
Its `GLOBAL_SOLID_SDF` cell is `MASKED` unless a separate explicit-solid court
passes and the required camera/normalization bindings are sealed.

## Model projections

### IRIS

Core corpus eligibility requires `SOURCE_IMPORT`, `CAMERA_OBSERVATION` and
`SURFACE_GEOMETRY_UDF`. Global SDF, normals and graphics/response families are
sample-level masks. `GLOBAL_SOLID_SDF=MASKED` does not drop an otherwise useful
IRIS sample; it disables only the corresponding loss.

### TESSA

Requires current `MECHANICAL_SURFACE_TOPOLOGY` and `TESSA_TEACHER` authority.
Historical `SOURCE_FALLBACK` is not teacher authority. Geometry, deformation and
intersection courts must prove the current consumer contract.

### MIRA

Requires exact `RIG_HIERARCHY` plus `SKIN_WEIGHTS` bound to the exact skin
carrier and current source/import equivalence. Raw evidence that a skin exists is
routing information only, not MIRA admission.

### AXIS

Requires current rig and animation/kinematic authority. Corpus eligibility is
still insufficient to begin AXIS V6 training until the frozen IRIS/TESSA product
shared-scene identity and frozen MIRA codec identity are available.

## Closure order

1. Bind the exact 370-source roster and hashes.
2. Reuse historical evidence only after current semantic revalidation.
3. Run missing family producers/courts through the Platform research lane.
4. Resolve required matrix cells to PASS/MASKED/FAIL/QUARANTINED; no required
   `UNKNOWN` may survive a sealed model roster.
5. Perform global lineage/dedup analysis.
6. Isolate one global end-to-end unseen pool **before any model training**.
7. Derive model-specific TRAIN/VAL rosters from the remaining eligible rows.
8. Seal family manifests, matrix, rosters, split and exact materialization plan.
9. Mint a training-readiness receipt only after all consumer-contract hashes and
   producer/court versions are bound.
10. Materialize training tensors/shards from those sealed manifests.
11. Deliver a Run-All materialization notebook that verifies seals and performs
    transport/materialization only; it may not recompute or override eligibility.

Training order after closure is IRIS/TESSA, frozen shared scene, MIRA codec,
then AXIS V6, followed by Compiler/product end-to-end evaluation on the globally
sealed unseen pool.

## Execution authority

Git branches/PRs are source-code review surfaces only. Heavy corpus measurement
and truth production belongs to the Platform `Experiment -> Campaign -> Attempt`
lineage against a sealed/deployed EngineRelease. GitHub Actions may coordinate or
make progress visible, but an Actions PASS is not research authority. Exact CAS /
Artifact Registry evidence and Attempt lineage are retained per subject.

## Deliverables

The closure is complete only when the repository contains reviewable generated
summaries backed by sealed Platform evidence:

- `CORPUS370_SUBJECT_FAMILY_AUTHORITY_MATRIX_V1.json`
- a human-readable matrix summary/CSV
- IRIS/TESSA/MIRA/AXIS corpus-eligibility rosters
- lineage/dedup manifest
- global unseen + model-specific TRAIN/VAL split receipt
- materialization manifest/shard plan
- training-readiness receipt
- final Run-All materialization notebook

Until those seals exist, `training_authorized` remains false.
