# IRIS vNext Decision / Falsification Ledger — 2026-10-18

Status: **ACTIVE RESEARCH LEDGER**

This file exists to prevent conversational or branch-history drift. Every major design idea is recorded with its evidence, what it proves, what it does not prove, and what replaces it when falsified.

## Status vocabulary

- `SUPPORTED_BASELINE`: current evidence supports keeping the design as baseline.
- `FALSIFIED`: experiment or source audit disproved the claimed use.
- `PARTIALLY_SUPPORTED`: useful subset survives; stronger claim is false/unproven.
- `TRUTH_DEBT`: target authority is inadequate.
- `CONDITIONING_DEBT`: target depends on information unavailable to the prediction path.
- `OPTIMIZATION_DEBT`: target is potentially valid/conditioned but convergence is inadequate.
- `CONSUMPTION_DEBT`: evidence exists but downstream behavioral use is incomplete/unproven.
- `PENDING_COURT`: implementation/idea awaits an independent court.
- `HISTORICAL_ONLY`: preserved for provenance; must not drive current decisions.

## Ledger

| Item | Current status | What created it | What it proved | What it falsified / failed to prove | Replacement / next action |
|---|---|---|---|---|---|
| V2/V3 long G2 surface continuation | FALSIFIED | Fit8 G2 runs from step1601 lineage | non-empty surface telemetry alone is insufficient | long G2 preserved quality; pre-optimizer telemetry was adequate | surgical migration + post-optimizer gates + bounded G2 |
| V5 single-objective downstream training on shared H | FALSIFIED | A1→R1 V5 run | appearance head can learn while shared H changes | later task optimization safely preserves mechanics/geometry | V6 cumulative retention experiment, then V8 structural isolation |
| V6 cumulative retention | PARTIALLY_SUPPORTED | V6 A1/R1/T1 run | cumulative prior objectives can preserve mechanics | loss balancing alone preserves geometry/extraction | V8 scene-core seal; no return to loss-only carrier protection |
| V8 post-M1 scene-core seal | SUPPORTED_BASELINE | completed 12k V8 run | geometry/extraction can remain invariant while specialist heads keep learning | downstream carrier rewriting is necessary for current heads | fail-closed runtime policy; reopening requires falsification court |
| V8 completion as scientific Fit8 PASS | FALSIFIED | completed V8 receipt | training completed and architecture experiment is interpretable | scientific quality thresholds/topology courts existed | retain as measurement baseline only |
| Frozen H lacks graphics information | FALSIFIED under current teacher definitions | V8 A/R/T learning after M1 seal | current appearance/relation/topology targets remain decodable from sealed H | generic adapters are immediately required | no capacity expansion before clean truth + conditioning + optimization courts |
| Near-zero relation loss means real-world graphics solved | FALSIFIED as claim | V8 relation metrics + teacher audit | current teacher targets are decodable | teacher semantics, held-out balance, conditioning and runtime behavior are automatically correct | truth/conditioning audit + compiler/runtime lineage court |
| Full semantic/amodal owner teacher from generic FIT8 3D | FALSIFIED as generic program | `agent/unified-iris-stdgenpp-foundation` census | geometry can produce strong visibility/depth/material/mechanics evidence | drawable-owner, hidden support, intentional 2D presentation override are uniquely identifiable | salvage measurable producers; abstain on unidentifiable truth |
| Foundation branch measurable 3D census/provenance tools | PARTIALLY_SUPPORTED | foundation branch | ray/depth/visibility/material/rig evidence can be measured | whole branch should become current authority | port source-bound tools selectively with provenance |
| Named semantic taxonomy required by IRIS | NOT SUPPORTED | StdGEN++ comparison + V8 | no evidence currently requires body/hair/cloth labels | role-free evidence is inadequate | keep ontology-free model; revisit functions only from courts |
| Boundary Causality Teacher V1 | PENDING_COURT | post-V8 truth cleanup | contract can abstain and separate provable boundary causes | it is already a populated authoritative corpus | wire measurable producers; report support/class stats; held-out court |
| `graphics_truth_v1.py` cut helper | TRUTH_DEBT / HISTORICAL | V1 helper | anonymous owner-equivalence proxy is deterministic | cut is independently identified | preserve provenance only; do not use as new authority |
| V8 Fit8 shard `relation_cut_probability` | TRUTH_DEBT | active shard builder inherited by V8 | owner-or-material boundary proxy is decodable from sealed H | metric proves authoritative drawable-cut learning | replace with explicit-validity/provenance Graphics Truth V2 |
| V1 relation alpha/appearance boundary branch | CONDITIONING_DEBT + TRUTH_DEBT | shard builder + `GraphicsRelationHeadV1` + trainer source audit | V8 compared raster-view boundary targets against camera-free static predictions | residual can be interpreted as pure LR/capacity debt | split static structural causes from camera-conditioned projected boundary outputs before optimization |
| Graphics Truth V2 explicit validity | PENDING PRODUCER | `graphics_truth_v2.py` + `training_v2.py` | training can abstain per family/sample instead of inventing negatives | corpus producer already supplies authoritative V2 rows | wire identifiable producers + conditioning schema + support statistics before training |
| AppearanceEvidenceHead V1 boundary | OPTIMIZATION_DEBT + TRUTH_DEBT | V8 appearance diagnostics | view-conditioned head receives view-conditioned evidence | relation static-conditioning mismatch explains its debt | separately audit truth, class balance, gradients and capacity after corpus cleanup |
| `~0.05` p95 as global PASS threshold | FALSIFIED as global rule | historical engineering expectations | useful planning reference for comparable normalized residuals | heterogeneous metrics share one scientific threshold | preregister metric-specific held-out gates |
| TESSA low training loss = topology PASS | FALSIFIED as claim | V8 T1/J1 loss | topology objective is optimizable from sealed H | actual connectivity/hole/manifold/surface quality passed | independent TESSA topology court |
| StructureHeadV2 omitted offset default | FALSIFIED IMPLEMENTATION | focused branch CI | existing direct-call regression test correctly caught `None` reaching TESSA decoder | V2 preserved StructureHeadV1 zero-default semantics | normalize omitted/None offset to zero; keep regression in CI |
| StdGEN++ fixed semantic classes should be copied | REJECTED | external architecture comparison | their semantics actively support decomposition/extraction | RealSaS needs their exact vocabulary | translate decomposition functions into evidence + Compiler authority |
| StdGEN++ specialization/LoRA discipline | PARTIALLY_SUPPORTED TRANSFER | paper + public StdGEN code + V8 | constrained specialization/shared-representation protection is valuable | RealSaS should add adapters immediately | adapters only after specialist-capacity court |
| StdGEN++ coarse-to-fine high-res extraction | PENDING_COURT | StdGEN++ paper | plausible route around dense-grid resolution/memory bottleneck | current IRIS quality problem is proven to be resolution | evaluate after topology/surface court identifies resolution limit |
| StdGEN++ hole/sign regularization | PENDING_COURT | StdGEN++ paper | explicit thin-structure continuity objective can help topology | IRIS should mutate SDF signs or add loss blindly | add only after topology court; preserve fail-closed surface authority |
| StdGEN forced empty-SDF sign repair | REJECTED AS AUTHORITY | public StdGEN code | useful diagnostic that empty implicit fields occur | silent forced signs are acceptable scientific truth | retain fail-closed IRIS collapse handling; consider pre-failure viability telemetry |
| Compiler role-free graphics graph | PARTIALLY_SUPPORTED | `graphics_v1.py`, runtime graph, native renderer | visibility/order/fragments affect runtime state; native visibility behavior proven | every serialized signal is behaviorally consumed | complete machine-readable audit, especially attachment/support band |
| Runtime attachment consumption | CONSUMPTION_DEBT | graphics compile/runtime serialization | attachment records survive into binary | native reference renderer uses them behaviorally | identify owning runtime/animation consumer or mark transport-only |
| Runtime support-band consumption | CONSUMPTION_DEBT | support-band serialization | value survives transport | native reference renderer applies it | wire authorized 2D consumer or remove behavioral claims |

## Relation truth + conditioning correction

Do not conflate historical producers:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    cut = (owner differs) OR (material id differs)
```

Also do not conflate **static structural** and **view-conditioned projected** relations:

```text
V1 static_relations(Ha, Hb):
    same, cut, attachment, alpha boundary, appearance boundary

V1 view_relations(Ha, Hb, camera):
    front, occlusion, visibility

V8 shard relation boundary target:
    partly/current-view 3x3 raster alpha/RGB neighborhood
```

The projected boundary target therefore lacks matching camera conditioning in the V1 static relation branch.

Replacement requirement:

```text
static branch -> authoritative view-invariant structural relations only
view branch   -> front/occlusion/visibility + projected view-dependent boundaries
```

See `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`.

## Current replacement chain

```text
historical shared-carrier curriculum
        ↓ falsified by V5/V6
V8 M1 carrier seal
        ↓ supported architecture baseline
post-V8 quality program
        ├─ truth cleanup / abstention
        ├─ target-conditioning cleanup
        ├─ relation truth V2 validity masks
        ├─ head-specific convergence courts
        ├─ compiler/runtime consumption audit
        ├─ independent topology court
        └─ only then specialist capacity changes
```

## Current authority separation

- `main`: canonical merged source-code continuation line; behind active review work until merge.
- V8 completed artifact: measurement baseline bound to exact V8 source + external receipts/checkpoints; not rewritten by later hardening.
- active review source: post-V8 branch/PR until reviewed and merged.
- `Attempt` / Artifact Registry / ProductRevision: durable research/product authority; Git chronology is not a substitute.

## Update rule

Any future architectural claim that changes this ledger must record:

1. exact source/commit and apparatus;
2. exact artifact/receipt or test evidence;
3. claim scope;
4. what was falsified;
5. what survives;
6. replacement path;
7. whether prior artifacts remain valid or merely historical.
