# IRIS vNext Decision / Falsification Ledger — 2026-10-18

Status: **ACTIVE RESEARCH LEDGER**

This file prevents conversational or branch-history drift. Every major design idea records evidence, what it proves, what it does not prove, and the replacement when falsified.

## Status vocabulary

- `SUPPORTED_BASELINE`: current evidence supports keeping the design.
- `FALSIFIED`: experiment/source audit disproved the claimed use.
- `PARTIALLY_SUPPORTED`: useful subset survives; stronger claim is false/unproven.
- `TRUTH_DEBT`: target authority inadequate.
- `CONDITIONING_DEBT`: target depends on information unavailable to the prediction path.
- `OPTIMIZATION_DEBT`: target may be valid/conditioned but convergence is inadequate.
- `CONSUMPTION_DEBT`: evidence exists but downstream behavioral use is incomplete/unproven.
- `PENDING_COURT`: awaits independent court.
- `HISTORICAL_ONLY`: provenance only.
- `QUARANTINED`: artifact/path exists but current authority/provenance is not trusted until reverified.

## Ledger

| Item | Current status | What evidence says | What it does not prove / replacement |
|---|---|---|---|
| V2/V3 long G2 continuation | FALSIFIED | non-empty surface telemetry was insufficient; long G2 degraded quality | bounded G2 + post-optimizer gates + surgical migration |
| V5 downstream single-objective rewriting of shared H | FALSIFIED | appearance improved while mechanics/surface collapsed | downstream rewriting is unsafe; test retention then structural isolation |
| V6 cumulative retention | PARTIALLY_SUPPORTED | prior losses preserved mechanics better | geometry/extraction still drifted because carrier stayed trainable; loss balancing alone is insufficient |
| V7 | HISTORICAL_ONLY / REPAIR | repaired TESSA coordinate contract | did not change V6 optimizer/curriculum/carrier-update policy |
| V8 post-M1 scene-core seal | SUPPORTED_BASELINE | geometry/extraction stayed invariant while A/R/T learned | not scientific Fit8 PASS; keep seal fail-closed |
| V8 optimizer topology | SUPPORTED HISTORICAL FACT | physical groups were `scene_core`, combined `evidence_heads`, `structure`; stage objectives functionally isolated active specialist gradients | do not rewrite V8 as if it had head-separated optimizer groups |
| Post-V8 specialist optimizer V2 | SUPPORTED SOURCE / PENDING EXPERIMENT | `scene_core=0`; mechanical/appearance/relation/structure have separate explicit LRs and per-group gradient/update telemetry; CI green | does not prove better scientific quality until focused held-out experiment runs |
| V8 completion = scientific PASS | FALSIFIED as claim | completed 12k run is interpretable baseline | no metric-specific preregistered gates/topology court |
| Frozen H lacks graphics information | FALSIFIED under current teachers | sealed H supported A/R/T learning | no immediate justification for generic LoRA or carrier reopening |
| Near-zero relation loss = graphics solved | FALSIFIED as claim | current teacher targets are decodable | does not establish teacher authority, balanced held-out discrimination, conditioning, or runtime behavior |
| Full owner/amodal/presentation truth recoverable from generic FIT8 3D | FALSIFIED as generic program | source geometry yields strong visibility/depth/material/rig evidence | intended drawable owner, hidden support, 2D override, named ontology remain unidentifiable without added authority |
| Foundation measurable 3D tools | PARTIALLY_SUPPORTED | exact-source geometry/material/skin/ray/dynamic/residual/local-continuation evidence can be measured | selective salvage only; whole foundation branch is not authority |
| Named semantic taxonomy required | NOT SUPPORTED | V8 + architecture evidence do not require body/hair/cloth labels | keep role-free evidence; revisit only from courts |
| Boundary Causality Teacher V1 | PENDING CORPUS / COURT | contract separates provable causes and abstention | not yet a sealed populated authoritative corpus |
| Historical V1 cut helper | TRUTH_DEBT / HISTORICAL_ONLY | deterministic anonymous-owner complement | not independent drawable-cut truth |
| V8 shard cut proxy | TRUTH_DEBT | `(owner differs) OR (material differs)` was decodable | not authoritative drawable cut; replace with explicit validity/provenance |
| V1 relation projected-boundary branch | CONDITIONING_DEBT + TRUTH_DEBT | raster-view targets were compared to camera-free static predictions | residual is not pure LR/capacity debt |
| Graphics Truth V2 contract/loss | SUPPORTED SOURCE | explicit value+authority+provenance+validity; missing truth abstains | source contract alone is not a materialized corpus |
| Relation Truth V2 producer | SUPPORTED SOURCE / MATERIALIZATION PENDING | `build_fit8_relation_truth_v2.py` writes versioned sidecars/manifest + family support stats | repository state has not yet sealed an actual produced Fit8 V2 corpus as experiment authority |
| GraphicsRelationHeadV2 | SUPPORTED SOURCE / MIGRATION ARTIFACT PENDING | projected boundaries moved to camera-conditioned view branch; V1-compatible rows preserved; new rows neutral | no real V8 checkpoint→V2 migrated artifact/receipt yet |
| Occlusion Truth V2 support | SINGLE_CLASS / COURT INELIGIBLE | valid two-hit occlusion rows currently provide positive support only | do not claim binary discrimination or fabricate negatives |
| Static same/cut/attachment Truth V2 | ABSTAIN WITHOUT AUTHORITY | producer refuses to invent them | no PASS/negative labels without independent source authority |
| AppearanceEvidenceHeadV1 boundary | OPTIMIZATION_DEBT + TRUTH_DEBT | it is already camera-conditioned | relation-head conditioning mismatch does not explain its residual; audit truth/balance/gradients separately |
| `~0.05` p95 global threshold | FALSIFIED as global rule | useful historical planning context only for comparable normalized metrics | preregister family-specific held-out gates; `0.1–0.2` is not auto-success |
| TESSA low training loss = topology PASS | FALSIFIED as claim | topology objective is optimizable from sealed H | independent actual-surface topology court required |
| StructureHeadV2 omitted offset default | FALSIFIED IMPLEMENTATION / FIXED | CI caught `None`; normalized to zero | keep direct regression test |
| StdGEN++ fixed semantic classes | REJECTED | semantics perform decomposition/extraction functions | translate functions, not body/hair/cloth vocabulary |
| StdGEN++ specialization/LoRA discipline | PARTIALLY_SUPPORTED TRANSFER | constrained specialization can protect shared representation | generic adapters not authorized before clean-target capacity court |
| StdGEN++ coarse-to-fine extraction | PENDING_COURT | plausible high-res thin-structure route | use only if topology/surface court proves resolution bottleneck |
| StdGEN++ hole/sign regularization | PENDING_COURT | explicit topology continuity objective may help | do not mutate signs/add losses blindly |
| StdGEN forced empty-SDF sign repair | REJECTED AS AUTHORITY | illustrates empty-field failure mode | IRIS remains fail-closed; no silent repair-as-success |
| Compiler role-free graphics graph | PARTIALLY_SUPPORTED | visibility/order/fragments affect runtime state | not every serialized signal is behaviorally consumed |
| Runtime attachment | CONSUMPTION_DEBT | survives Compiler→runtime binary | current reference raster does not behaviorally use it |
| Runtime support-band | CONSUMPTION_DEBT | serialized/validated | current reference renderer does not apply it |
| Self-hosted prepare run `39121291171` / #22 | HISTORICAL_ONLY / CANCELLED | stale PR head `90c9c2f...` locally sealed teacher/cohort/preparation, then entered Drive persistence and was cancelled | not current authority; cannot be consumed as current preparation |
| `gdrive:RealSaS_Fit8_IRIS_vNext_20261016` after stale #22 | QUARANTINED | stale run spent ~150 s inside `rclone copy` before cancellation; round-trip SHA verification never completed | do not assume remote unchanged or valid; explicit file-level SHA/provenance re-audit required before reuse/replacement |

## Foundation salvage map

Authority source: `agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`.

Selective source-defensible paths:

```text
geometry/topology
  tools/corpus/qualify_f_geometry_carrier_v1.py
  tools/corpus/fit8_surface_topology_invariants_v1.py

render/geometric + material/skin provenance
  tools/corpus/fit8_render_vertex_equivalence_v1.py
  tools/corpus/fit8_gltf_static_carrier_v1.py
  tools/corpus/ufbx_static_census_probe_v2.c
  tools/corpus/ufbx_static_census_probe_v3.c
  tools/corpus/fit8_fbx_static_carrier_v1.py
  tools/corpus/fit8_fbx_static_carrier_v2.py

static all-hit evidence
  tools/corpus/fit8_static_ray_census_v1.py
  tools/corpus/fit8_static_ray_census_v2.py
  tools/corpus/fit8_static_ray_census_v3.py

authored-motion fixed-frame evidence
  tools/corpus/fit8_gltf_dynamic_pose_v1.py
  tools/corpus/fit8_dynamic_fixed_frame_presentation_v1.py
  tools/corpus/fit8_dynamic_mechanics_metrics_v1.py
  tools/corpus/fit8_gltf_dynamic_census_v2.py

residual boundary
  tools/corpus/fit8_f_residual_boundary_graph_v1.py
  tools/corpus/fit8_f_residual_signature_v1.py
  tools/corpus/fit8_face_strand_boundary_incidence_v1.py

local continuation
  tools/corpus/fit8_local_patch_teacher_v1.py
  tools/corpus/fit8_patch_face_support_v1.py
  tools/corpus/fit8_f_patch_continuation_court_v1.py
  tools/corpus/fit8_f_face_qualified_continuation_court_v1.py
```

Owner/amodal/presentation/semantic-partition code is not promoted as generic FIT8 truth. Unknown remains abstention.

## Relation truth correction

Historical producers remain distinct:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

V8 Fit8 shard builder:
    cut = (owner differs) OR (material differs)
```

Current replacement:

```text
static branch -> same/cut/attachment only when independently authoritative
view branch   -> front/occlusion/visibility/projected alpha/projected appearance
```

`GraphicsRelationHeadV2` matches this conditioning split at source level. Materialized Truth V2 corpus, real V8→V2 checkpoint migration, focused training, and held-out PASS remain future work.

## Stale self-hosted forensic record

Run `39121291171` was triggered by stale branch `research/iris-vnext-single-carrier-graphics@90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6`.

Locally sealed values from its log:

```text
teacher receipt sha256 = 7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27
manifest value sha256  = 37a5fccbdcb2d98735291c7963e75e852e260e4d4eecfc2ef9f60ff8dbcb732b
preparation receipt    = fdd72430a9e6573b51b98eddf181d159cca08079654ceed08de92a25ae840082
source commit          = 90c9c2f35ffc77aeea10cd99391bbe9a4d4a31c6
```

Drive persistence started at 12:52:04Z and cancellation landed at 12:54:34Z before the round-trip verifier printed PASS. Forensic GitHub artifact: `iris-vnext-fit8-preparation-39121291171`, artifact id `12137271848`.

This event changes **remote trust**, not current architecture. It does not authorize a new experiment and does not invalidate historical V8.

## Current replacement chain

```text
V5/V6 shared-carrier rewriting
        ↓ falsified as safe continuation
V8 M1 carrier seal
        ↓ supported architecture baseline
post-V8 quality program
        ├─ explicit truth/abstention
        ├─ matching conditioning
        ├─ head-separated optimizer telemetry
        ├─ relation support/court eligibility
        ├─ compiler/runtime consumption audit
        ├─ independent topology court
        └─ only then capacity/LoRA/extraction changes
```

## External evidence pins

- StdGEN public source inspected: `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766`.
- InstantMesh public source inspected: `TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.
- Exact StdGEN/StdGEN++ full training loop is not public in the inspected StdGEN repository; paper training details remain `PAPER_EXACT`, not `CODE_EXACT`.

## Current authority separation

- `main`: merged production source authority.
- V8 completed artifact: immutable measurement baseline bound to exact source + receipts/checkpoints.
- active post-V8 branch: temporary review source; verified pre-docs code head `b38feee7...`, workflow `39126288809` green.
- draft PR #116: stale until synchronized to the final green active head.
- Attempt / Artifact Registry / ProductRevision: durable research/product authority.

## Update rule

Any architectural claim changing this ledger must record exact source/commit/apparatus, artifact/receipt or test evidence, claim scope, falsification, surviving result, replacement path, and whether earlier artifacts remain valid or merely historical.
