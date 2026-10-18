# IRIS vNext Decision / Falsification Ledger — 2026-10-18

Status: **ACTIVE RESEARCH LEDGER**

This file exists to prevent conversational or branch-history drift. Every major design idea is recorded with its evidence, what it proves, what it does not prove, and what replaces it when falsified.

## Status vocabulary

- `SUPPORTED_BASELINE`: current evidence supports keeping the design as baseline.
- `FALSIFIED`: experiment or source audit disproved the claimed use.
- `PARTIALLY_SUPPORTED`: useful subset survives; stronger claim is false/unproven.
- `TRUTH_DEBT`: implementation exists but target authority is inadequate.
- `OPTIMIZATION_DEBT`: target is potentially valid but convergence/quality is inadequate.
- `CONSUMPTION_DEBT`: evidence exists but downstream behavioral use is incomplete/unproven.
- `PENDING_COURT`: implementation/idea awaits an independent court.
- `HISTORICAL_ONLY`: preserved for provenance; must not drive current decisions.

## Ledger

| Item | Current status | What created it | What it proved | What it falsified / failed to prove | Replacement / next action |
|---|---|---|---|---|---|
| V2/V3 long G2 surface continuation | FALSIFIED | Fit8 G2 runs from step1601 lineage | non-empty surface telemetry alone is insufficient | long G2 preserved quality; pre-optimizer telemetry was adequate | surgical migration + post-optimizer gates + bounded G2 |
| V5 single-objective downstream training on shared H | FALSIFIED | A1→R1 V5 run | appearance head can learn while shared H changes | later task optimization safely preserves mechanics/geometry | V6 cumulative retention experiment, then V8 structural isolation |
| V6 cumulative retention | PARTIALLY_SUPPORTED | V6 A1/R1/T1 run | cumulative prior objectives can preserve mechanics | loss balancing alone preserves geometry/extraction | V8 scene-core seal; no return to loss-only carrier protection |
| V8 post-M1 scene-core seal | SUPPORTED_BASELINE | completed 12k V8 run | geometry/extraction can remain invariant while specialist heads keep learning | downstream carrier rewriting is necessary for current heads | make seal fail-closed in runtime policy; reopening requires falsification court |
| V8 completion as scientific Fit8 PASS | FALSIFIED | completed V8 receipt | training completed and architecture experiment is interpretable | scientific quality thresholds/topology courts existed | retain as measurement baseline only |
| Frozen H lacks graphics information | FALSIFIED under current teacher definitions | V8 A/R/T learning after M1 seal | current appearance/relation/topology targets remain decodable from sealed H | generic adapters are immediately required | no capacity expansion before clean truth + optimization courts |
| Near-zero relation loss means real-world graphics solved | FALSIFIED as claim | V8 relation metrics + teacher audit | current teacher targets are decodable | teacher semantics, held-out balance, runtime behavior are automatically correct | teacher authority audit + compiler/runtime lineage court |
| Full semantic/amodal owner teacher from generic FIT8 3D | FALSIFIED as generic program | `agent/unified-iris-stdgenpp-foundation` census | geometry can produce strong visibility/depth/material/mechanics evidence | drawable-owner, hidden support, intentional 2D presentation override are uniquely identifiable | salvage measurable producers; abstain on unidentifiable truth |
| Foundation branch measurable 3D census/provenance tools | PARTIALLY_SUPPORTED | `agent/unified-iris-stdgenpp-foundation` | ray/depth/visibility/material/rig evidence can be measured | whole branch should become current authority | port source-bound tools selectively with provenance |
| Named semantic taxonomy required by IRIS | NOT SUPPORTED | comparison with StdGEN/StdGEN++ | no evidence currently requires body/hair/cloth labels | role-free evidence is inadequate | keep ontology-free model; capacity/teacher courts may revisit functions, not names |
| Boundary Causality Teacher V1 | PENDING_COURT | post-V8 truth cleanup | contract can abstain and separate provable boundary causes | it is already a populated authoritative corpus | wire measurable producers; report support/class statistics; held-out court |
| `cut_probability = 1 - same_drawable` in `graphics_truth_v1.py` | TRUTH_DEBT | historical role-free relation teacher | deterministic proxy is learnable | cut is an independent causal boundary target | preserve V8 provenance; replace with validity-aware/abstaining cut truth |
| Appearance boundary target | OPTIMIZATION_DEBT + TRUTH_DEBT | V8 A/R diagnostics | current head can receive signal | boundary causality is cleanly identified or quality is adequate | Boundary Causality corpus first, then head-specific optimizer court |
| `~0.05` p95 as global PASS threshold | FALSIFIED as global rule | historical engineering expectations | useful planning reference for comparable normalized residuals | heterogeneous metrics share one scientific threshold | preregister metric-specific held-out gates |
| TESSA low training loss = topology PASS | FALSIFIED as claim | V8 T1/J1 loss | topology objective is optimizable from sealed H | actual connectivity/hole/manifold/surface quality passed | independent TESSA topology court |
| StdGEN++ fixed semantic classes should be copied | REJECTED | external architecture comparison | their semantics actively support decomposition/extraction | RealSaS needs their exact vocabulary | translate decomposition functions into evidence + Compiler authority |
| StdGEN++ specialization/LoRA discipline | PARTIALLY_SUPPORTED TRANSFER | paper + public StdGEN code + V8 | constrained specialization protects/shared structure better than indiscriminate rewriting | RealSaS should add adapters immediately | adapters only after specialist-capacity court |
| StdGEN++ coarse-to-fine high-res extraction | PENDING_COURT | StdGEN++ paper | plausible route around dense-grid resolution/memory bottleneck | current IRIS quality problem is proven to be resolution | evaluate after topology/surface court identifies resolution limit |
| StdGEN++ hole/sign regularization | PENDING_COURT | StdGEN++ paper | explicit thin-structure continuity objective can help topology | IRIS should mutate SDF signs or add loss blindly | add only after topology court; preserve fail-closed surface authority |
| StdGEN forced empty-SDF sign repair | REJECTED AS AUTHORITY | public StdGEN code | useful diagnostic that empty implicit fields occur | silent forced signs are acceptable scientific truth | retain fail-closed IRIS collapse handling; consider pre-failure viability telemetry |
| Compiler role-free graphics graph | PARTIALLY_SUPPORTED | `graphics_v1.py`, runtime graph and native renderer | visibility/order/fragments affect downstream runtime state; native visibility behavior proven | every serialized signal is behaviorally consumed | complete machine-readable consumption audit, especially attachment/support band |
| Runtime attachment consumption | CONSUMPTION_DEBT | graphics compile/runtime serialization | attachment records survive into binary | native reference renderer uses them behaviorally | identify owning runtime/animation consumer or leave explicitly non-behavioral |
| Runtime support-band consumption | CONSUMPTION_DEBT | fragment support-band serialization | support-band value survives transport | native reference renderer currently uses it | either wire an authorized 2D support consumer or remove from behavioral claims |

## Current replacement chain

```text
historical shared-carrier curriculum
        ↓ falsified by V5/V6
V8 M1 carrier seal
        ↓ supported architecture baseline
post-V8 quality program
        ├─ truth cleanup / abstention
        ├─ head-specific convergence courts
        ├─ compiler/runtime consumption audit
        ├─ independent topology court
        └─ only then specialist capacity changes
```

## Current authority separation

- `main`: canonical merged source-code continuation line; currently behind the active review work.
- V8 completed artifact: historical scientific/measurement baseline bound to exact V8 source and external receipts/checkpoints; not rewritten by later source hardening.
- active review source: post-V8 research branch/PR until reviewed and merged.
- `Attempt` / Artifact Registry / ProductRevision: durable research/product authority; Git branch chronology must not substitute for them.

## Update rule

Any future architectural claim that changes this ledger must record:

1. exact source/commit and apparatus;
2. exact artifact/receipt or test evidence;
3. claim scope;
4. what was falsified;
5. what survives;
6. replacement path;
7. whether prior artifacts remain valid or are merely historical.
