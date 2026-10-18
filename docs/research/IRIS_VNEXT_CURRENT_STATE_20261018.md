# IRIS vNext V8 Baseline State — 2026-10-18

Status: **IMMUTABLE INTERPRETATION OF COMPLETED V8 / NO SCIENTIFIC FIT PASS**

> This file records what the completed V8 artifact established. It is intentionally not the live source-work state. For current code, truth replacements, CI and next actions read root `CURRENT_STATE.md`, `CURRENT_PLAN.md` and the decision ledger.

## Authority

- V8 source head: `5083c9915533ec8ec6530c7d229c635887abfea7`
- Training reached global step `12000`.
- Completion schema: `RealSaS.IRISVNextFit8TrainingComplete.v8`.
- `scene_core_sealed_post_m1 = true`.
- `scientific_fit_pass_claimed = false`.
- `official_numeric_quality_thresholds_present = false`.

Later source hardening does not retroactively mutate this artifact or its target semantics.

## What V8 established

### 1. M1 carrier sealing is a supported architectural baseline

After M1, shared `scene_core` remained sealed while downstream specialist heads continued training. Geometry/extraction diagnostics remained stable through the rest of the run.

```text
G/M establish the shared carrier
        ↓
M1 seal
        ↓
A/R/T/J train specialist heads against sealed H
```

This experiment closed the observed V5/V6 failure mode in which later objectives rewrote a carrier that already contained useful geometry/mechanics structure. It supports structural isolation over loss-only retention for this apparatus.

It does **not** prove that the carrier should never be reopened under any future apparatus; reopening requires a separate preregistered falsification court with non-regression gates.

### 2. Frozen H contains substantial downstream-decodable structure

With H sealed, appearance, relation and topology objectives continued improving. Several relation residuals became very small under the historical Fit8 teacher targets.

Correct claim:

> current targets are substantially decodable from sealed H.

Incorrect stronger claim:

> the corresponding real-world semantic/presentation relations are scientifically solved.

Teacher correctness, balance, held-out sampling and runtime behavioral consumption remain separate obligations.

### 3. Historical relation-target semantics must remain exact

Do not retroactively simplify the V8 relation producer.

Two historical code paths exist:

```text
graphics_truth_v1.py helper:
    cut = 1 - anonymous-owner-equivalence

Fit8 shard builder used by V8:
    relation_cut_probability = (owner differs) OR (material id differs)
```

V8's recorded cut residual therefore belongs to the **owner-or-material boundary proxy** written by its shard builder. It is not authoritative drawable-cut truth.

The post-V8 `Graphics Truth V2` replacement does not rewrite V8 shards; it exists only for future work.

### 4. Boundary-family quality is not acceptable yet

The completed run does not justify normalizing `0.1–0.2` residuals into success. Boundary/cut families remain quality/truth debt; appearance-boundary behavior was especially weak.

Historical `~0.05` expectations are a planning reference only for genuinely comparable normalized metrics. No global official threshold existed for V8.

### 5. Training loss is not topology qualification

TESSA/topology objectives improved, but no independent held-out topology-validity court was sealed with V8. Therefore V8 does not establish topology PASS.

## Foundation branch disposition learned before/around V8

`agent/unified-iris-stdgenpp-foundation` remains research evidence and a source of measurable tools.

Salvageable:

- exact source geometry/topology provenance;
- rig/skin/deformation evidence;
- authored material-region provenance when present;
- static all-hit ray stacks;
- fixed-frame dynamic visibility/order;
- geometric front/occlusion/visibility;
- geometric-component census;
- residual-boundary analysis;
- local geometric continuation as evidence;
- subject-local partition contracts where authority is explicit.

Not promotable from generic FIT8 3D alone:

- authoritative drawable-owner truth;
- hidden/amodal support truth;
- intentional 2D presentation override;
- global semantic class identities;
- inferred hidden continuation as ground truth.

The correct result for an unidentifiable target is unknown/abstention.

## What V8 handed to the post-V8 program

V8 did **not** authorize another generic capacity expansion or broad rerun. It handed off these questions:

1. clean/abstaining boundary and relation truth;
2. strict specialist convergence/quality courts;
3. Compiler→runtime behavioral consumption;
4. independent topology qualification;
5. only then capacity/extraction/topology interventions if evidence requires them.

For the live implementation status, do not continue from this file alone; return to root `CURRENT_STATE.md` and `CURRENT_PLAN.md`.
