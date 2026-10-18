# RealSaS-OPT — Current Plan

**Date:** 2026-10-19  
**Status:** POST-V9 RELATION EVIDENCE QUALIFICATION / V10 TRAINING BLOCKED  
**Active source line:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Production authority:** `main` until reviewed merge  
**Last completed scientific attempt:** V9 focused relation `ATTEMPT_001` at source `7029ee3f48787aa9f7e27eecef89155eb9368a96`  
**Next optimizer policy:** **NO V10 OPTIMIZER STEP** until richer teacher evidence, Truth V3 support, preregistration, materialization and readiness are all sealed.

## 0. Non-negotiable rules

### One canonical carrier

```text
K x candidate M*
        ↓
static admission + candidate mechanics
        ↓
deterministic selection
        ↓
ONE canonical M/G/W_M
        ↓
Graphics + Motion + Runtime
```

A second deformable visual mesh remains forbidden.

### Post-M1 carrier seal

```text
Geometry + Mechanics -> shared H -> M1 seal -> A/R/T/J specialist learning
```

`scene_core` remains sealed unless a separate preregistered falsification court authorizes reopening it.

### Truth / conditioning

- unknown is first-class;
- unidentifiable owner/amodal/presentation targets abstain;
- view-conditioned truth requires matching prediction conditioning;
- authored-motion evidence is **not** promoted to supervision unless the prediction path has sufficient conditioning or an identifiability court proves it is decodable without added pose state;
- Compiler/runtime may consume admissible evidence but may not manufacture missing truth.

## 1. V9 completed result — scoped scientific PASS only

V9 focused relation attempt completed at:

```text
source:
  7029ee3f48787aa9f7e27eecef89155eb9368a96
attempt:
  ATTEMPT_001
optimizer steps:
  1600
run state:
  SCHEDULE_STOP_REACHED
formal court:
  SCIENTIFIC_PASS
```

Exact result record:

`docs/research/IRIS_V9_FOCUSED_RELATION_RESULT_20261019.json`

Earned V9 claims:

- cleaned camera-conditioned projected alpha boundary is decodable from sealed V8 H;
- projected alpha held-out court passed:
  - global balanced accuracy `0.7599126696586609`;
  - global balanced BCE `0.5106152445077896`;
  - minimum per-view aggregate balanced accuracy `0.7362911105155945`;
- projected appearance boundary held-out court passed:
  - global balanced accuracy `0.8208722770214081`;
  - global balanced BCE `0.42430220544338226`;
  - minimum per-view aggregate balanced accuracy `0.7912296056747437`;
- frozen parameter domains remained exact;
- static same/cut/attachment abstention was preserved.

V9 does **not** prove full relation quality:

- front held-out balanced accuracy `0.5587652325630188`;
- visibility A `0.5658917725086212`;
- visibility B `0.5533536672592163`;
- some individual views remain at or below chance;
- positive-only occlusion support regressed from step-0 accuracy `1.0` to final `0.4809832274913788` and was not protected by the V9 court;
- no binary occlusion discrimination is proven;
- no global IRIS Fit8 PASS or TESSA topology PASS is claimed.

Classification:

`V9 = SCOPED CLEAN-TRUTH BOUNDARY EXPERIMENT__NOT FINAL RELATION EXPERIMENT`

## 2. Post-V8 sequencing correction

The original Post-V8 plan ordered:

```text
A carrier seal
→ B truth cleanup / abstention
→ C richer source-defensible teacher evidence salvage
→ D focused specialist optimization
```

V9 substantially completed B before optimization, but C was only source-audited, not fully consumed/materialized as relation evidence before training.

That sequencing miss is now explicit. V10 training is blocked until Workstream C is materially closed for the relation scope.

## 3. Workstream C — relation evidence qualification in progress

### 3.1 Authored-motion relation conditioning court

Source:

`tools/iris_vnext/audit_authored_motion_relation_conditioning_v1.py`

Purpose:

- compare rest vs authored poses under identical admitted Stage01 cameras;
- measure same-pair order flips, pair-membership drift, two-hit support loss/gain;
- prove whether motion-derived relation labels are identifiable by the current relation path before admitting them as supervision.

Current prediction-path fact:

`GraphicsRelationHeadV2` receives sealed-H point features + camera features, but no explicit pose/motion state.

Therefore authored-motion evidence is currently:

`COURT_ONLY__NO_TRAINING_AUTHORIZATION`

No motion label becomes V10 target merely because it is measurable.

### 3.2 Complete static all-hit relation evidence

Source:

`tools/iris_vnext/build_fit8_relation_all_hit_evidence_v1.py`

This corrects a structural limitation of Truth V2.

Truth V2 used only first/second same-ray hits and therefore enforced:

```text
visibility_a = front
visibility_b = 1 - front
```

That equivalence is not valid for a complete stack. For ray stack `A,B,C`, pair `B,C` has:

```text
front(B,C) = B
visibility(B) = 0
visibility(C) = 0
```

The all-hit evidence producer therefore emits:

- complete deterministic ordered ray stacks;
- every unordered stack pair in both orientations;
- pairwise `front` from depth order;
- global visibility from rank-zero only;
- positive same-ray occlusion support only;
- no fabricated occlusion negatives;
- no hidden-pair RGB/alpha boundary labels;
- no new same/cut/attachment authority.

Pair coordinates are bound to the exact legacy IRIS `center_xyz` and `normalization_half_extent` used by frozen H.

### 3.3 Exact evidence materialization lane

Workflow:

`.github/workflows/iris_vnext_relation_evidence_self_hosted_v1.yml`

Policy:

- self-hosted read-only input lane;
- exact Stage01 bytes;
- exact legacy shard bytes;
- exact sealed V9 teacher bytes verified against V9 round-trip receipt;
- no optimizer/trainer;
- no Drive mutation;
- Actions artifact output only;
- `training_authorized=false`.

Current materialization run:

`39173687686`

Do not design V10 truth thresholds from provisional local probes. Use the exact Actions artifact from this lane.

## 4. V9 trainer efficiency debt — separate from model capacity

V8 used the larger active training/optimizer GPU footprint. V9 was a much narrower relation specialist.

Nevertheless V9 ran around `~1.0 s/step`, versus historical V8 relation-stage median around `~0.204 s/step`.

This is **not evidence of a larger V9 model or a need for more capacity**. The dominant implementation debt is the full frozen-domain SHA court executing inside every optimizer step.

Required future behavior:

```text
full frozen hash court:
  step 0
  checkpoint boundaries
  final court

NOT:
  every optimizer step
```

Optimizer parameter whitelisting and `requires_grad=False` remain active continuously.

## 5. V10 Truth V3 admission — blocked on measured evidence

Do not implement a V10 trainer first.

After the exact Workstream C artifact completes, quantify globally + per subject + per view:

- all-hit stack depth distribution;
- 2+ and 3+ hit-ray support;
- complete oriented-pair count;
- front positive/negative support;
- visibility A/B positive/negative support;
- rows where pairwise front and global visibility are decoupled;
- authored-motion same-pair order flip rate;
- authored-motion pair-membership drift;
- authored-motion overlap-support loss/gain;
- occlusion positive-support preservation requirements.

Then define Relation Truth V3.

Expected constraints, subject to the measured court:

- `front`: deterministic full-stack pairwise order;
- `visibility_a/b`: deterministic global stack visibility, independent of pairwise front;
- projected alpha/appearance boundary: preserve V9 current-view RGBA authority only where applicable;
- hidden deep pairs: no invented RGB/alpha boundary label;
- occlusion: positive same-ray support remains measurement/support unless defensible negative authority is established;
- static same/cut/attachment: abstain absent independent authority.

## 6. V10 model/init decision

Do **not** increase model capacity by default.

`GraphicsRelationHeadV2` already exposes separate front and visibility outputs; the immediate limitation was corpus truth, not necessarily head expressivity.

V9 final relation checkpoint is retained as a candidate initialization:

```text
file:
  step_00001600_45e3995f70a9489882650312acfdd147.pt
sha256:
  393b113f0ca3b7a2c23da61022692d15bd0ca8d37167426f546266702df5da05
```

It may be used only after a V10 preregistration explicitly authorizes:

- exact V8 + HeadV2 bootstrap lineage;
- exact V9 relation-state load;
- zero-step reproduction of V9 boundary gains;
- frozen-domain non-regression;
- new Truth V3 support/court binding.

If zero-step V9-state migration does not reproduce the V9 boundary court, fail closed and do not train.

## 7. V10 training prerequisites

Before optimizer step 1, all must exist and PASS:

1. exact richer relation evidence artifact + batch receipt;
2. Truth V3 producer + manifest with explicit authority/provenance/validity;
3. split-support / class-balance / abstention audit;
4. authored-motion conditioning disposition;
5. exact V9-state-or-neutral initialization receipt;
6. preregistered held-out gates;
7. projected-boundary non-regression gates;
8. ordering/visibility absolute gates, not merely weak step-0 non-regression;
9. occlusion positive-support preservation gate;
10. frozen-domain hash policy moved out of the per-step hot path;
11. exact candidate byte round-trip/materialization seal;
12. readiness status explicitly authorizing the fixed V10 experiment.

No thresholds may be relaxed after observing held-out results.

## 8. Later work remains unchanged

After the relation evidence/truth program:

- independent TESSA topology court;
- Compiler/runtime attachment/support-band behavioral closure;
- Fit8 E2E generic integration analogous to Knight;
- only then unseen evaluation;
- capacity/LoRA/high-resolution extraction only if a clean-target court demonstrates a real capacity/resolution bottleneck.

## Execution order from current state

```text
A. finish exact Workstream C materialization run
B. inspect evidence artifact and seal support findings
C. define Relation Truth V3 from measured authority only
D. build V10 support / split / abstention courts
E. qualify V9-final relation-state initialization vs neutral alternative
F. lock V10 preregistration and absolute relation courts
G. materialize exact V10 candidate + byte round-trip
H. readiness PASS at optimizer step 0
I. deliver V8-style A100 Run-All notebook
J. run fixed V10 schedule
K. inspect Drive artifacts directly; no manual output handoff
```
