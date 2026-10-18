# IRIS vNext Current State — 2026-10-18

Status: **POST-V8 MEASUREMENT BASELINE / NO SCIENTIFIC FIT PASS**

This note freezes the engineering interpretation of the completed V8 Fit8 run before the next quality program begins. It does not retroactively create numeric scientific thresholds.

## Authority

- V8 source head: `5083c9915533ec8ec6530c7d229c635887abfea7`
- Training reached global step `12000`.
- Completion schema: `RealSaS.IRISVNextFit8TrainingComplete.v8`.
- `scene_core_sealed_post_m1 = true`.
- `scientific_fit_pass_claimed = false`.
- `official_numeric_quality_thresholds_present = false`.

The completed V8 run is therefore an architectural/measurement baseline, not a scientific PASS artifact.

## What V8 established

### 1. M1 carrier sealing is the current architectural invariant

After M1, the shared `scene_core` remained sealed while the downstream specialist heads continued training. Geometry/extraction diagnostics remained stable through the remainder of training. This closes the earlier failure mode in which downstream objectives rewrote a carrier that had already acquired useful geometry/mechanics structure.

Current rule:

```text
G/M establish the shared carrier
        ↓
M1 seal
        ↓
A/R/T/J train specialist heads against the sealed carrier
```

Reopening `scene_core` after M1 is forbidden by default. It requires a separately preregistered falsification court proving that the sealed carrier is the limiting factor and specifying non-regression gates for already-qualified structure.

This is a structural freeze, not loss balancing and not a temporary learning-rate trick.

### 2. Frozen H contains substantial downstream graphics structure

With the carrier sealed, several relation heads reached very small residuals while appearance/topology objectives continued to improve. The correct interpretation is capacity evidence for the frozen representation, not proof that the corresponding semantic ontology is scientifically correct.

Particularly strong decoded relations include `same_drawable`, `attachment`, `front`, `occlusion`, and `visibility` under the current teacher definitions.

### 3. Boundary-family quality is not acceptable yet

The completed run does **not** justify normalizing 0.1–0.2-class residuals as acceptable merely because training completed. Boundary/cut families remain material optimization/truth debt, with appearance-boundary behavior especially weak.

A historical engineering expectation around the ~0.05 scale may be used only as a planning reference for genuinely comparable normalized residual metrics. It is **not** an official global threshold and must never be applied across heterogeneous metrics by unit-blind comparison.

Future PASS claims require metric-specific, preregistered courts.

### 4. Training loss is not topology qualification

TESSA/topology objectives improved during V8, but no independent held-out topology-validity court was sealed with the run. Therefore V8 does not establish a topology PASS.

### 5. Low relation loss is not semantic authority

Near-zero loss proves that the current targets are decodable from the sealed carrier. It does not prove that subject-local drawable ownership, intentional presentation override, or hidden/amodal structure were correctly identified by current FIT8 3D evidence.

Teacher correctness, class balance, trivial-negative prevalence, held-out sampling, and compiler consumption remain separate obligations.

## StdGEN++ foundation branch disposition

The branch `agent/unified-iris-stdgenpp-foundation` is retained as research evidence and a source of measurable tools, not adopted wholesale as current authority.

### Salvage into the current line

- exact source geometry/topology provenance;
- rig/skin/deformation evidence;
- authored material-region provenance when present;
- static all-hit ray stacks;
- dynamic fixed-frame visibility/order evidence;
- geometric `front` / `occlusion` / `visibility` evidence;
- geometric-component census;
- residual-boundary analysis;
- local geometric continuation as evidence, never hidden-state truth;
- subject-local partition contracts where their authority is explicit.

### Do not promote from FIT8 3D alone

- authoritative drawable-owner truth;
- hidden/amodal support truth;
- intentional 2D presentation override;
- global semantic class identities;
- inferred hidden continuation as ground truth.

When the source cannot identify a target, the correct result is `UNKNOWN`/abstention rather than a proxy silently promoted to truth.

## Current priority

The next program is not a generic model-capacity expansion. It is:

1. strict post-V8 convergence/quality courts;
2. boundary-causality truth cleanup;
3. compiler consumption/behavior audit for learned evidence;
4. independent held-out topology qualification;
5. only then, evidence-driven specialist-capacity changes if the sealed carrier is shown insufficient.
