# RealSaS Structural System Index

## 2026-10-18 post-V8 review state

> Navigation and ownership index. `CURRENT_STATE.md` and `CURRENT_PLAN.md` define the active program. Historical recovery/runtime-audit documents remain provenance only.

## Current continuation

Read first:

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
4. `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`
5. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
6. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
7. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
8. `AGENTS.md`

`main` remains merged production-code authority until reviewed vNext merges. The active post-V8 source line is temporary review source, not research/product authority. Draft PR #116 is the canonical vNext review vehicle and its head is required to match the active review source before review/merge. The completed V8 artifact remains bound to exact V8 source `5083c9915533ec8ec6530c7d229c635887abfea7` plus its checkpoint/receipt lineage.

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
| Graphics relations | `GraphicsRelationHeadV1` | role-free same/cut/attachment/front/occlusion/visibility evidence; historical relation targets have explicit truth debt |
| Relation truth V2 | `graphics_truth_v2.py`, `training_v2.py` | explicit authority/provenance/validity; missing truth abstains; producer wiring pending |
| Boundary truth | `boundary_causality_v1.py` | provable causes + unknown/abstention; producer/corpus wiring in progress |
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

## IRIS training authority

Historical curriculum families G1/G2/M1/A1/R1/T1/J1 remain useful identifiers, but current post-V8 law is not “all stages may rewrite the shared trunk”.

Canonical post-M1 stage IDs are:

```text
A1_APPEARANCE_EVIDENCE
R1_ROLE_FREE_RELATIONS
T1_MECHANICAL_STRUCTURE
J1_JOINT_FINETUNE
```

`quality_policy_v1.py` fails on unknown post-M1 aliases and future V8-line trainer execution checks the actual `scene_core` optimizer-group LR. Reopening the carrier requires an explicit falsification court.

`structure_v2.py` preserves the established TESSA sequence-position default: omitted/explicit-`None` offset means zero, while explicit batched offsets remain supported.

## Graphics evidence / truth status

Current learned outputs include:

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

Do not conflate the two historical relation-truth producers:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

Thus V8's cut metric concerns the historical **owner-or-material boundary proxy**, not pure `1 - same` and not authoritative drawable-cut truth.

`graphics_truth_v2.py` is the replacement contract for new work: every scalar target has explicit authority/provenance/validity and may abstain. Historical V8 shards remain immutable.

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

See the machine-readable lineage report before making stronger claims.

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

Public StdGEN code confirms constrained LoRA transformer adaptation, shared triplane features and separate SDF/RGB/deformation/FlexiCubes/semantic decoders. StdGEN++ paper supplies the three-stage, dual-branch, multilayer, coarse-to-fine and hole-regularization design claims. Exact StdGEN/StdGEN++ training-loop source is not currently available in the public StdGEN repository.

Transfer method discipline, not a fixed semantic vocabulary.

## Canonical code-line policy

`main` remains the single merged continuation line. Temporary review branches may exist for code review, but must not become Attempt/ProductRevision authority.

The vNext draft PR #116 is the intended visible review vehicle. Its head must track the current post-V8 review source by fast-forward rather than leaving multiple divergent vNext narratives.

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

## Machine-readable architecture / impact

Architecture discovery remains available through `platform/internal/architecture/registry.go` and `tools/realsas_architecture.py`. Use source/manifest impact tools for dependency discovery. Unknown impact must be investigated; it never means zero consumers.

## Platform execution entrypoints

- `docs/platform/EXECUTION_LANES.md`
- `platform/cmd/realsasctl`
- `tools/platform_release_snapshot.py`
- `compiler/realsas_compiler_services/platform_worker/stage_inputs.py`
- `canonical/PLATFORM_MAIN_AUDIT_20261008.json`

## Historical compatibility vocabulary

`CAA` remains compatibility/provenance terminology for historical material paths. The active vNext decomposition/composition domain is **Graphics**. Old independent visual-pipeline continuation is historical only and must not be revived.
