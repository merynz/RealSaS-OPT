# Corpus370 Training Readiness Closure V1

## Objective

Close the expanded 370-subject corpus from raw source inspection to model-specific training readiness without inheriting historical `training_authorized` or treating artifact completion as scientific admission.

The closure invariant is:

`TRAINING_READY = SOURCE_AUTHORITY ∧ SUBJECT×FAMILY_AUTHORITY_MATRIX ∧ MODEL_SPECIFIC_ROSTER ∧ LINEAGE_SAFE_SPLIT ∧ CONSUMER_CONTRACT_MATCH ∧ SEALED_MATERIALIZATION`

No optimizer may start before the corresponding model roster is sealed.

## Candidate universe

- 139 established Truth139 identities
- 231 expansion candidates
- 370 total candidate rows

Candidate count is not training count.

## Truth-family routing

Every candidate must receive an explicit disposition for every family. Missing truth is represented as `MASKED`, `FAIL`, or `QUARANTINED`; never as numeric zero or implicit PASS.

Families:

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

Allowed dispositions:

- `PASS`
- `MASKED`
- `FAIL`
- `QUARANTINED`
- `UNKNOWN`

`UNKNOWN` is forbidden in a sealed training roster.

## Global sign / SDF rule

Global signed supervision is admitted only for a qualified explicit solid entity or assembly whose actual boundary is the distance domain.

Forbidden substitutions:

- nearest-component sign as global solid sign
- component-local parity promoted to subject-global occupancy
- union occupancy multiplied by surface UDF as an exact union SDF
- silent hole closing or silent decorative-surface deletion
- zero-filling missing sign rows

Open/modular assets may retain surface/UDF, normals, appearance, visibility, rig, skin, animation, and mechanical evidence while `GLOBAL_SOLID_SDF` is `MASKED`.

## Model-specific admission

### IRIS

Required core:

- `SOURCE_IMPORT=PASS`
- `CAMERA_OBSERVATION=PASS`
- `SURFACE_GEOMETRY_UDF=PASS`

Optional/masked:

- `GLOBAL_SOLID_SDF` may be `PASS` or `MASKED`
- learned graphics families must be individually admitted or masked according to the active recipe

Global SDF losses consume only `GLOBAL_SOLID_SDF=PASS` rows.

### TESSA

Required core:

- `SOURCE_IMPORT=PASS`
- `MECHANICAL_SURFACE_TOPOLOGY=PASS`
- `TESSA_TEACHER=PASS`

A historical `SOURCE_FALLBACK` selection is not sufficient. Every selected teacher, including exact-source fallback, must pass current geometry, deformation, attachment, inversion/fold-through, self-intersection, response/strain, and consumer-rebind courts.

### MIRA

Required core:

- `SOURCE_IMPORT=PASS`
- `RIG_HIERARCHY=PASS`
- `SKIN_WEIGHTS=PASS`

Skin must bind to the exact qualified carrier. Approximate semantic or nearest-neighbour rebind is forbidden unless separately qualified by a future contract.

### AXIS

Required core:

- `SOURCE_IMPORT=PASS`
- `RIG_HIERARCHY=PASS`
- `ANIMATION_KINEMATICS=PASS` where animation supervision is used

Final AXIS V6 training additionally requires frozen product shared-scene output and frozen MIRA codec identity.

## Closure sequence

1. Preserve the 370-row immutable candidate roster and raw-source hashes.
2. Finish source/import parity classification for all rows.
3. Finish explicit-solid/assembly authority for all candidates that can support global SDF; mark all others `MASKED`, `FAIL`, or `QUARANTINED` with reason codes.
4. Requalify all TESSA teachers under the current mechanical/deformation/intersection courts.
5. Complete rig/skin and canonical animation replay authority for every eligible source.
6. Complete source/camera/observation and graphics measured-domain courts.
7. Complete mechanical topology and causal-response courts.
8. Emit one canonical 370 × family authority matrix with evidence refs and reason codes.
9. Resolve every `UNKNOWN` required by any active model recipe.
10. Compute global dedup and leakage lineage.
11. Derive model-specific eligible rosters from the same authority matrix.
12. Build lineage-safe TRAIN/UNSEEN splits per model/compatible shared cohort.
13. Seal manifest + exact builder/schema/producer hashes + consumer contracts into a `TRAINING_READINESS_RECEIPT`.
14. Materialize deterministic training shards from the sealed manifest.
15. Re-open and verify every shard against the receipt before optimizer construction.
16. Train IRIS/TESSA first; freeze shared-scene checkpoint.
17. Train/freeze MIRA codec.
18. Train AXIS V6 on frozen shared-scene + frozen MIRA identities.

## Non-negotiable gates

- Historical Truth139 `training_authorized` is inventory only.
- Historical 112/27 split is not inherited automatically.
- `139/139 artifacts completed` is not a training-readiness claim.
- Candidate/source counts never stand in for model-specific eligible counts.
- A builder cannot grant authority to its own output without an independent court.
- A PASS receipt must identify the exact physical/semantic quantity that was proven.
- No fallback path bypasses the same courts required of non-fallback candidates.
- No final manifest contains `UNKNOWN` for any family consumed by that model.

## Final deliverables

The closure is complete only when the repository contains:

- canonical `CORPUS370_SUBJECT_FAMILY_AUTHORITY_MATRIX_V1`
- per-family court receipts and explicit failure/mask reason codes
- model-specific IRIS/TESSA/MIRA/AXIS roster manifests
- leakage-lineage manifest
- lineage-safe split manifests
- deterministic materialization manifest/shard index
- one sealed training-readiness receipt per training stage
- end-to-end Run-All notebook(s) that consume only the sealed authority roots

Until then, training remains closed.
