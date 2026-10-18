# RealSaS Structural System Index

## 2026-10-18 post-V8 review state

> Navigation and ownership index. `CURRENT_STATE.md` and `CURRENT_PLAN.md` define the active program. Historical recovery/runtime-audit documents remain provenance only.

## Current continuation details

Read first:

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
4. `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`
5. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
6. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
7. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
8. `AGENTS.md`

`main` remains merged production-code authority until reviewed vNext merges. The active post-V8 source line is `research/iris-vnext-post-v8-quality-plan-20261018`; it is temporary review source, not research/product authority. Draft PR #116 is kept fast-forward synchronized to green active-review heads. The completed V8 artifact remains bound to source `5083c9915533ec8ec6530c7d229c635887abfea7` plus exact checkpoint/receipt lineage.

Current V2/product continuity is preserved alongside the narrower post-V8 IRIS program. The machine-readable readiness pointer remains `canonical/V2_IMPLEMENTATION_READINESS.json`; the existing product/compiler control plane remains a dependency DAG rather than branch history. Post-V8 research changes do not silently supersede current canonical mesh, appearance or runtime authority.

## Current architecture summary

```text
source observations + cameras
        ↓
      IRIS
 shared scene carrier H
   ┌────┼──────────────┐
   ↓    ↓              ↓
Geometry Mechanics  Graphics/Structure evidence
   │    │              │
   └────┴──── M1 SEAL ─┘
             ↓
     specialist heads only
             ↓
       Evidence Package
             ↓
          Compiler
     ┌───────┼────────┐
     ↓       ↓        ↓
 fragments  order  contact/attachment
     └───────┼────────┘
             ↓
       one canonical M/G/W
             ↓
          Runtime
```

There is no mandatory named body-part taxonomy in the inference contract and no independent deformable visual mesh.

## Product / research ownership map

| Domain | Canonical source area | Current rule |
|---|---|---|
| Observation / cameras | observation + camera authority | source evidence remains exact; source cardinality and output-direction authority are separate |
| IRIS shared carrier | `models/iris/vnext/scene_core_v1.py` | shared scene representation; post-M1 sealed for downstream specialist training by current baseline |
| IRIS geometry/extraction | `models/iris/vnext/` + pinned FlexiCubes | learned field/extraction; fail closed on invalid surface; future high-res changes require court |
| Mechanical demand | `models/iris/vnext/heads_v1.py`, `teacher_truth_v1.py` | training evidence only; authored mechanics never inference input |
| Appearance evidence | `AppearanceEvidenceHeadV1` | RGBA/confidence/descriptor + alpha/appearance boundaries; evidence, not source override authority |
| Graphics relations V1 | `GraphicsRelationHeadV1` | historical role-free relation path; projected-boundary conditioning debt recorded |
| Graphics relations V2 | `models/iris/vnext/heads_v2.py` | static same/cut/attachment separated from camera-conditioned front/occlusion/visibility/projected boundaries; real migrated checkpoint pending |
| Relation truth V2 | `graphics_truth_v2.py`, `training_v2.py`, `build_fit8_relation_truth_v2.py`, `relation_support_v2.py` | explicit authority/provenance/validity + producer + support audit source implemented; actual corpus materialization/seal pending |
| Specialist optimization V2 | `models/iris/vnext/optimization_v2.py` | future head-separated optimizer/telemetry; `scene_core=0` fail-closed; historical V8 optimizer remains immutable |
| Boundary truth | `boundary_causality_v1.py` | provable causes + unknown/abstention; populated corpus still pending |
| Topology / TESSA | `structure_v1.py`, `structure_v2.py`, TESSA modules | learned proposal/grammar; no topology PASS until independent held-out court |
| Canonical mechanical carrier | candidate selection + canonical M/G/W path | exactly one product mechanical carrier after deterministic selection |
| AXIS | `models/axis/` + Compiler bindings | rig proposal/qualification on exact canonical carrier |
| MIRA | `models/mira/` + Compiler bindings | skin on exact canonical carrier; no semantic transfer to another mesh |
| Graphics Compiler | `compiler/realsas_compiler_core/graphics_v1.py` | globally resolves role-free evidence into fragments/order/attachments on canonical M |
| Graphics runtime materialization | `graphics_runtime_v1.py`, `runtime_graphics_binary_v1.py` | graph content is serialized, not hash-only |
| Native Graphics behavior | `runtime/realsas_cpp/src/graphics_v1.hpp`, `runtime_v2_graphics_reference.cpp` | visibility and explicit order are behaviorally consumed; attachment/support-band still have consumption debt |
| Appearance material | source-wins single-M material path | observed art remains source-wins; learned evidence cannot silently rewrite observed source |
| Motion | motion compile/proof modules | exact admitted motion semantics on qualified mechanical carrier |
| Runtime | C++ runtime / package consumers | playback/render only; no corrective generative inference in ordinary runtime |
| Orchestration | Go Platform + Engine activities | Platform owns durable lifecycle; Engine owns scientific/compiler semantics |
| Research state | Go `Attempt` | immutable research execution truth, not branch chronology |
| Production state | Go `ProductRevision` | qualified production truth |
| Artifacts | Artifact Registry/store | immutable typed bytes + dependency identities |
| Closure | proof/qualification | promotion eligibility is explicit and fail-closed |

### Canonical mesh domain

`compiler/realsas_compiler_core/surface_addressing_v1.py` provides the current `SurfaceAddressing` / `SurfaceAddressingIR` implementation for the stable canonical-M mechanical addressing domain. Existing Stage18/19 product continuity still treats mesh birth/qualification as a Compiler-owned authority boundary; post-V8 IRIS proposals/evidence do not bypass it. `QualifiedMeshIR` remains a product-authority type after its own qualification.

### Appearance / CAA

`compiler/realsas_compiler_core/appearance_authority_v2.py` remains the current product appearance authority home for Complete Appearance Authority / CAA semantics. `compiler/realsas_compiler_core/runtime_authority_v2.py` remains the qualified runtime-consumption authority boundary. Post-V8 learned Graphics evidence may feed Compiler structure but may not silently rewrite source-preserving appearance or runtime authority.

The existing V2 implementation remains governed through its current dependency DAG/readiness records. These product navigation anchors coexist with the narrower post-V8 IRIS experiment program; they are not scientific PASS claims for that program.

## IRIS training authority

Historical curriculum families G1/G2/M1/A1/R1/T1/J1 remain useful identifiers, but current post-V8 law is not “all stages may rewrite the shared trunk”.

Canonical post-M1 stage IDs:

```text
A1_APPEARANCE_EVIDENCE
R1_ROLE_FREE_RELATIONS
T1_MECHANICAL_STRUCTURE
J1_JOINT_FINETUNE
```

`quality_policy_v1.py` fails on unknown post-M1 aliases and future V8-line trainer execution checks the actual `scene_core` optimizer-group LR. Reopening the carrier requires an explicit falsification court.

Historical V8 already enforced post-M1 `scene_core=0` by stage policy but used physical groups `scene_core`, combined `evidence_heads`, and `structure`. Post-V8 `optimization_v2.py` splits future specialist groups into mechanical/appearance/relation/structure for independent LR/gradient/update telemetry. Do not rewrite V8 history as if this physical split existed in V8.

`structure_v2.py` preserves the established TESSA sequence-position default: omitted/explicit-`None` offset means zero, while explicit batched offsets remain supported.

## Graphics evidence / truth status

Current output families include:

```text
same_drawable_affinity
cut_probability
attachment_affinity
alpha_boundary_probability
appearance_boundary_probability
front_probability
occlusion_probability
visibility_probability_a/b
fragment embeddings
confidence
```

Important distinction:

- model output availability does not imply teacher correctness;
- serialization does not imply runtime behavior;
- low training loss does not imply scientific authority.

Do not conflate historical relation-truth producers:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

Thus V8's cut metric concerns the historical **owner-or-material boundary proxy**, not pure `1 - same` and not authoritative drawable-cut truth.

### V2 source state

Implemented and focused-CI green:

- explicit Truth V2 target authority/provenance/validity;
- validity-aware loss;
- versioned Fit8 relation sidecar/manifest producer;
- support/class-balance eligibility audit;
- `GraphicsRelationHeadV2` with projected boundaries camera-conditioned;
- deterministic V1→V2 migration semantics;
- future head-separated optimizer telemetry.

Still pending:

- materialized/sealed actual Fit8 Truth V2 corpus;
- real V8 checkpoint→V2 migrated artifact/receipt;
- focused training and held-out PASS.

Static same/cut/attachment abstain without independent authority. Current valid occlusion rows are single-class positive support and are binary-court ineligible until authoritative negative support exists.

## Graphics Compiler/runtime path

```text
IRIS relation evidence
        ↓
GraphicsEvidenceV1
        ↓
compile_graphics_v1
        ├─ adjacent fragment union
        ├─ per-view visibility
        ├─ directed order edges + DAG cycle handling
        └─ attachment candidates + mechanical contact
        ↓
CompiledGraphicsV1
        ↓
RuntimeGraphicsV1
        ↓
graphics.bin in RSS
        ↓
C++ Graphics-native runtime
```

Current behavioral proof:

- native `graphics_v1_contract_test.cpp` proves explicit fragment precedence can override contradictory physical depth;
- `test_vnext_graphics_native_e2e.py` proves Graphics visibility content changes final native RGBA behavior.

Current consumption debt:

- attachment data is serialized but not used by the reference raster path;
- support-band pixels are serialized/validated but not applied by the reference renderer.

## Topology / extraction status

V8 showed topology objective learnability from sealed H but did not seal an independent topology PASS.

The next topology authority must come from a held-out court over actual topology/surface outcomes, not training loss.

StdGEN++ ideas under conditional investigation:

- coarse-to-fine sparse active-grid high-resolution extraction;
- thin-structure sign/hole regularization;
- specialist LoRA/branches if clean-target capacity court fails.

None is current authority merely because it appears in the paper.

## StdGEN++ evidence discipline

`docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md` labels evidence as:

```text
CODE_EXACT
PAPER_EXACT
REALSAS_INFERENCE
UNAVAILABLE
```

Pinned public code for this audit:

```text
StdGEN      hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766
InstantMesh TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c
```

Public StdGEN confirms constrained LoRA transformer adaptation, shared triplane features and separate SDF/RGB/deformation/FlexiCubes/semantic decoders. StdGEN++ paper supplies the three-stage, dual-branch, multilayer, coarse-to-fine and hole-regularization claims. Exact full StdGEN/StdGEN++ training-loop source is not available in the inspected public repository.

Transfer method discipline, not a fixed semantic vocabulary.

## Foundation research salvage

Provenance source: `agent/unified-iris-stdgenpp-foundation@d90c82e24ccbf43081ca19f105bc4e4c7dfe9531`.

Selective reuse candidates are recorded path-by-path in `CURRENT_PLAN.md` and the decision ledger. They cover source geometry/topology, material/skin provenance, all-hit rays, authored-motion fixed-frame evidence, residual boundary diagnostics and local continuation courts.

Owner/amodal/presentation/semantic-partition claims are not generic FIT8 authority.

## Self-hosted preparation status

Known runner: `realsas-wsl-1660ti` with labels `[self-hosted, linux, x64, realsas]`.

Run `39121291171` / prepare #22 on stale PR head `90c9c2f...` started when the runner came online. It locally sealed stale-source preparation, then entered persistence to `gdrive:RealSaS_Fit8_IRIS_vNext_20261016` and was cancelled before round-trip SHA verification.

Consequences:

- runner remains offline until the deliberate read-only quarantine-audit workflow has been queued on the exact final review SHA;
- the affected Drive path is quarantined/unverified;
- stale local/GitHub artifact `12137271848` is forensic evidence only;
- current experiments must not consume the Drive path based on file presence;
- the legacy self-hosted preparation writer is retired; no current write workflow is authorized.

## Canonical code-line policy

`main` remains the single merged continuation line. Temporary review branches may exist for code review, but must not become Attempt/ProductRevision authority.

The active post-V8 branch is the current review source. Draft PR #116 is fast-forward synchronized only after an active-review head passes focused gates.

## Mechanics boundary

Historical carrier-native mechanics remain:

```text
canonical M
  -> AXIS G
  -> MIRA W_M on the same M
  -> Compiler qualification
  -> motion/runtime proof
```

Knight FIT1 evidence remains scoped historical/regression evidence. It does not establish Fit8/unseen/product generalization.

## Platform boundary

```text
Go Platform   = durable product/research state, Artifact Registry metadata,
                Attempt/ProductRevision lifecycle, promotion, workflow state
Python Engine = compiler semantics, model inference/training, geometry/ML,
                proof logic and typed failures
C++ Runtime   = playback, deformation/render hot paths, package consumption
```

Legacy direct workflows may remain for CI/reproducibility but do not mint durable research/product authority by themselves.

## Platform execution entrypoints

- `docs/platform/EXECUTION_LANES.md`
- `platform/cmd/realsasctl`
- `tools/platform_release_snapshot.py`
- `compiler/realsas_compiler_services/platform_worker/stage_inputs.py`
- `canonical/PLATFORM_MAIN_AUDIT_20261008.json`

## Historical compatibility vocabulary

`CAA` remains compatibility/provenance terminology for historical material paths. The active vNext decomposition/composition domain is **Graphics**. Old independent visual-pipeline continuation is historical only and must not be revived.
