# SkinTokens ↔ Arachne V6 Coverage Forensic Diff — 2026-09-29

## Status

`EVIDENCE_SEALED__NO_MODEL_MUTATION`

## Question

For the Knight smear / catastrophic skinning failure class, determine from exact source code whether SkinTokens solves an analogous problem by an uncertainty head, by a different loss, by sampling/coverage, by representation, or by downstream motion refinement; then identify what the exact Knight Arachne V6 training/qualification path omitted.

## Sources

### SkinTokens public authority

- Repository: `VAST-AI-Research/SkinTokens`
- Inspected commit: `273b691d35989d71cd17ff2895fdc735097b92d1`
- Relevant files:
  - `src/data/sampler.py`
  - `src/model/skin_vae_model.py`
  - `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py`
  - `README.md`
- Paper: Zhang et al., *Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging*, arXiv:2602.04805.

The public repository exposes the sampler, model, codec and inference path. It does **not** expose the complete paper training/GRPO implementation: `SkinVAEModel.get_loss_dict()` and `training_step()` are abstract / `NotImplemented` in the public source. Loss and GRPO details below are therefore attributed to the paper, while sampling and architecture claims are source-code verified.

### Exact Knight Arachne V6 authority

- Architecture: `RealSaS.Arachne.A1.RawSurfaceK4AttentionDirectSimplex.v6`
- Exact source: `run_knight_arachne_v6_demo_closure_v1.py`
- Exact source SHA-256: `39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01`
- Exact result: `ARACHNE_KNIGHT_V6_RESULT.json`
- Run: `SUBJECT2_KNIGHT_DEMO_V2_20260924`
- Imported / sealed evidence already recorded by:
  - `canonical/KNIGHT_ARACHNE_V6_UNCERTAINTY_SOURCE_AUDIT_V1_20260929.json`
  - `canonical/KNIGHT_SKIN_EVIDENCE_TRANSPORT_AUDIT_V1_20260929.json`
  - `canonical/KNIGHT_TOPOLOGY_WEIGHT_COUPLING_COURT_V1_20260929.json`
  - `canonical/KNIGHT_INVALID_ROW_WEIGHT_ORACLE_TOPOLOGY_CEILING_V1_20260929.json`
  - `canonical/KNIGHT_RELIABILITY_MASK_ONLY_TOPOLOGY_CEILING_V1_20260929.json`

## Verdict

The primary missing mechanism is **not** Dice loss, a deformation consequence loss, or an uncertainty head.

The exact Knight V6 already has a strong sparse-aware / hard-tail / articulated objective, but it trains and evaluates only the subset of rows marked teacher-valid, while Compiler qualification admits all rows based on numerical simplex legality.

The strongest source-level difference from SkinTokens is therefore:

> **SkinTokens explicitly changes the training distribution so sparse active bone-support regions are represented in training, whereas Knight Arachne V6 uniformly samples only teacher-valid surface rows and provides no total-coverage learning contract for the rows outside that mask.**

For RealSaS this is stricter than ordinary sparse class imbalance: there is a **supervision coverage hole**. Importance sampling can rebalance labeled rows, but it cannot create missing labels for rows that are excluded by `teacher_valid_mask`.

The current failure is best classified as:

`SUPERVISION_COVERAGE_AND_SAMPLING_GAP + QUALIFICATION_EVIDENCE_GAP`

rather than:

`MISSING_UNCERTAINTY_HEAD`

or

`MISSING_DICE_OR_DEFORMATION_LOSS`.

---

## 1. SkinTokens: what the public code actually does

### 1.1 Per-bone skin-field representation

SkinTokens does not primarily regress the complete (N × J) matrix row-by-row.

For each bone, it represents a sparse scalar skin field and compresses it through an FSQ-CVAE into discrete SkinTokens. The decoder is geometry conditioned.

This creates a bone-centric representation in which the model is explicitly exposed to each influence field rather than asking a shared row decoder to discover all rare supports from uniform vertex batches.

### 1.2 Active-region importance sampling is explicit in code

In `src/data/sampler.py`:

1. a normal global / uniform mesh sample is produced;
2. for a selected bone `indice`, the ground-truth scalar field is read:
   `_s = asset.vertex_groups['skin'][:, indice]`;
3. `sample_on_skin()` identifies faces for which at least one vertex has nonzero skin:
   `face_has_skin = np.any(skin[faces] > 0, axis=-1)`;
4. that support is expanded to nearby geometry using a cKDTree and bounded distance;
5. `num_skin_samples` samples are then drawn from this bone-active / near-active mask.

`src/model/skin_vae_model.py` carries both:
- `uniform_cond / uniform_skin`, and
- per-bone `dense_cond / dense_skin / dense_indices`.

So active deformation support is not merely a loss weighting trick: it is explicitly overrepresented in the decoder's training inputs.

### 1.3 Sparse objective

The paper specifies SkinTokens CVAE reconstruction using BCE + MSE + Dice. Dice is specifically motivated by extreme skin-matrix sparsity / support coverage.

### 1.4 Motion / geometric refinement

The paper then applies GRPO with four rewards including:
- skinning coverage/sparsity, and
- deformation smoothness under LBS/random poses.

This is post-training consequence pressure, not an uncertainty head.

### 1.5 No explicit per-row reliability head found

The inspected public SkinTokens source does not expose a per-vertex/per-row `confidence`, `uncertainty`, `sigma`, or `reliability` output as the central solution.

Therefore a RealSaS reliability head would be an additional compiler-facing mechanism, not a faithful copy of SkinTokens.

---

## 2. Arachne already copied more of SkinTokens than expected

### 2.1 Historical Arachne V4 objective is already sparse-aware

`models/arachne/v4/loss_v4.py` contains:

- BCE
- MSE
- Dice
- normalized row L1
- top-10% hard-tail / CVaR-style row L1
- blend-boundary weighted row L1
- articulated LBS deformation-ratio loss

Default weights:

- BCE = 1.0
- MSE = 0.1
- Dice = 1.0
- row L1 = 1.0
- hard tail = 0.5
- blend boundary = 0.5
- articulated deformation = 0.25

Therefore `MISSING_DICE` and `MISSING_MECHANICAL_CONSEQUENCE_LOSS` are falsified as general Arachne explanations.

### 2.2 Exact Knight V6 keeps the same important objective family

Exact source `run_knight_arachne_v6_demo_closure_v1.py`, function `_objective()`, computes:

- categorical cross entropy against soft skin targets;
- MSE;
- Dice with epsilon in numerator and denominator;
- row L1 mean;
- top 10% hard-tail row L1;
- blend-boundary loss;
- articulated deformation ratio.

Total:

`CE + 0.1*MSE + Dice + RowL1 + 0.5*HardTail + 0.5*Boundary + 0.25*Articulated`.

So the Knight failure does **not** come from accidentally dropping Dice or articulated consequence pressure in V6.

---

## 3. Exact Knight V6 coverage gap

### 3.1 Training schedule

Exact V6 source:

```python
clean = np.flatnonzero(np.asarray(valid, bool))
schedule = _sample_schedule(clean)
```

and:

```python
def _sample_schedule(clean):
    rng = np.random.default_rng(DECODER_SEED)
    n = min(ROWS_PER_STEP, len(clean))
    for i in range(MAX_STEPS):
        out[i] = rng.choice(clean, size=n, replace=False)
```

with:

`ROWS_PER_STEP = 384`.

Every optimization step then computes the entire objective only on:

`truth_t[idx], world_t[idx]`

where `idx` comes from this `clean` set.

There is no bone-active / support-stratified branch in the schedule.

### 3.2 The excluded population is large

Exact V6 result:

- total Stage15 surface rows: **12,090**
- teacher-valid / clean rows: **10,611**
- coverage: **0.8776674938**
- training-excluded rows: **1,479 = 12.23%**

These 1,479 rows receive no V6 supervised readout gradient.

### 3.3 Evaluation repeats the same coverage boundary

Exact V6 final science metrics:

- `rows_evaluated = 10,611`
- `rows_total = 12,090`
- `teacher_coverage = 0.8776674938`
- row-L1 p95 = `0.046917...`
- articulated deformation ratio = `0.004111...`

Thus the attractive closure numbers are metrics over the same teacher-valid population used by training.

They do not prove correctness of the 1,479 uncovered rows.

### 3.4 Compiler nevertheless admits all rows

The same final V6 evaluation calls `qualify_skin()` on the dense prediction for **all 12,090 rows**.

The sealed result reports:

- Compiler row count = **12,090**
- corrected rows = 0
- total correction L1 ≈ `4.42e-13`
- Compiler pass = true.

At this historical Knight authority point, qualification proves numerical skin-row legality, not supervised/mechanical reliability.

Therefore the exact contract was effectively:

`train/evaluate 87.77% -> predict 100% -> simplex-qualify 100% -> ship 100%`.

That is the core coverage leak.

---

## 4. Downstream identity: 1,479 -> 1,474 is the same population, not a coincidence

The current product mesh candidate has **12,049** vertices rather than the original **12,090** Stage15 surface rows.

In `tools/audit_knight_weight_outlier_signal_court_v1.py`:

`surface_ids(candidate)` requires every candidate vertex to have an **identity support binding**:

- exactly one source surface coefficient;
- coefficient exactly 1.0.

`teacher_matrix()` then indexes `teacher_valid_mask` by those exact source `surface_id` values.

Current product court result:

- product candidate vertices = **12,049**
- teacher-valid = **10,575**
- teacher-invalid = **1,474**

Hence the 1,474 current invalid product vertices are the exact downstream subset of the original teacher-valid-mask population carried by source surface identity.

The cardinality difference is explained by candidate survival:
- 1,479 Stage15 rows were invalid;
- 1,474 of those invalid source rows survive as identity-bound product candidate vertices;
- five invalid Stage15 surface rows are not present in the product candidate.

This seals the lineage between the V6 supervision hole and the current catastrophic population.

---

## 5. Causal evidence that this coverage hole matters

### 5.1 Failure concentration

`KNIGHT_TOPOLOGY_WEIGHT_COUPLING_COURT_V1_20260929.json`:

- every one of **209/209** actual >10× false-negative cross-region faces contains a teacher-invalid vertex;
- every one of **99/99** unsafe catastrophic false-negative cross-region faces contains a teacher-invalid vertex;
- **73/77 = 94.8%** false-positive seam faces contain a teacher-invalid vertex;
- only **66/1034 = 6.38%** true-positive cross-region faces contain one.

This is not a weak correlation.

### 5.2 Fixing only invalid-row weights closes the frozen topology gates

`KNIGHT_INVALID_ROW_WEIGHT_ORACLE_TOPOLOGY_CEILING_V1_20260929.json` replaces only the 1,474 invalid product rows with teacher weights, while keeping topology inference teacher-label free.

Best arm:

- actual >10×: **1**
- actual >4×: **26**
- p99 edge: **1.5635**
- worst edge: **15.0187**
- synthetic unsafe faces: **387**
- mixed precision: **0.9960**
- unsafe cross-region recall: **0.9606**
- no face deletion
- region-pure holeless output.

So the existing downstream topology machinery becomes sufficient when those weights become correct.

### 5.3 Merely detecting bad rows is insufficient

`KNIGHT_RELIABILITY_MASK_ONLY_TOPOLOGY_CEILING_V1_20260929.json` supplies the invalid mask but does not replace the bad weights.

It still leaves approximately:

- >10×: **200**
- >4×: **232**
- worst: **185.45**
- unsafe: **590**

Therefore the product needs better values / completion, not only a reliability flag.

---

## 6. What SkinTokens solved vs. what RealSaS still has to solve

### SkinTokens solves

**Sparse active support under labeled meshes.**

Its bone-centric field representation + active-region importance sampling ensure rare nonzero deformation support is not drowned by the much larger zero-weight region.

Its Dice objective reinforces support reconstruction.

Its RL stage further pushes global rig validity and deformation behavior.

### Knight Arachne's stricter problem

**A real supervision hole.**

Rows with `teacher_valid_mask=false` are excluded entirely from V6 optimization and science evaluation, yet they receive product predictions and are admitted downstream.

SkinTokens-style importance sampling cannot directly supervise a row for which no trusted target is available.

Therefore copying only the SkinTokens sampler is necessary as a sparse-support control, but is not sufficient to solve the Knight owner.

---

## 7. Training experiment order

### Court A — forensic coverage diff

**Completed by this report.**

Conclusion:
- missing Dice: falsified;
- missing articulated consequence loss: falsified;
- missing uncertainty head as SkinTokens mechanism: falsified;
- active-region sampling mismatch: supported;
- total-supervision coverage mismatch: strongly supported;
- compiler qualification evidence gap: already independently supported.

### Court B — sampler-only matched causal retrain

Keep frozen:
- exact V4 backbone checkpoint;
- exact V6 readout architecture;
- optimizer family;
- objective;
- total sampled rows / optimizer budget;
- training horizon and evaluation gates.

Change only the schedule.

Compare:
1. exact V6 `UNIFORM_VALID` baseline;
2. `SKINTOKENS_STYLE_HYBRID`: matched-budget mixture of uniform valid rows plus per-joint active / near-active valid rows.

Do not select a winner from Knight after tuning thresholds. Pre-register arms and sampling budgets before optimizer step 1.

Purpose:
- test whether rare/support-starvation within the **valid** 87.77% population materially contributes;
- measure per-joint active-support exposure;
- measure valid-row tail and mechanical consequence.

Boundary:
- this court cannot claim to solve the 12.23% unlabeled population.

### Court C — total-coverage completion learning

This is the court capable of attacking the actual uncovered-row owner without treating invalid bank rows as trustworthy GT.

Use rows with trusted artist/teacher weights from the training corpus, then create **synthetic missing-coverage patches**:

- isolated rows;
- connected islands;
- thin straps;
- joint/blend boundaries;
- mechanically coherent regions;
- patch-size distribution matched to observed coverage holes, without using Knight answers as targets.

Hide their truth from the predictor and require reconstruction from product-available evidence:
- surface geometry / graph;
- qualified skeleton;
- neighboring trusted skin evidence during completion training;
- Arachne surface/joint latent state;
- mechanical probes where appropriate.

The hidden truth remains available only to the training loss / validation oracle.

This directly trains the capability we currently lack:

`supported skin context + structure -> recover missing skin field`.

Real Knight invalid rows remain evaluation-only / unlabeled.

### Court D — reliability head only if B + C are insufficient

Only after the field/completion mechanism is tested should a per-row reliability head be introduced.

A reliability head remains useful for compiler abstention and qualification, but current causal evidence says it cannot substitute for actually repairing the row values.

---

## 8. Architecture decision

Current evidence does **not** justify replacing Arachne's V4/K4 backbone with SkinTokens FSQ wholesale.

Why:

- valid-region Arachne performance is already strong;
- the invalid-row weight oracle shows that changing only ~12.23% of product rows is enough to close the downstream topology gate;
- therefore a global representation-capacity failure is not yet established.

The lowest-risk scientific order is:

`coverage/sampling -> learned completion -> reliability/qualification -> representation redesign only if still necessary`.

## Final owner statement

The Knight skinning failure uncovered a contract error that the FIT metrics hid:

> **Arachne V6 was allowed to declare closure from metrics computed on 10,611 supervised rows while Compiler admitted 12,090 predicted rows. The 1,479 unsupervised rows were not given a total-coverage learning mechanism; 1,474 of that same source-ID population survive into the product mesh, and they dominate the catastrophic failure class.**

SkinTokens' relevant lesson is therefore not “add uncertainty.”

It is:

> **make sparse deformation support impossible to disappear from training, and make the skin representation/training contract cover the field that will actually be shipped.**

RealSaS must go one step further because its teacher coverage is incomplete: it needs a learned, auditable completion path or an equivalent total-coverage source of trustworthy targets before uncovered rows can be admitted as product skin.


---

## 9. Sealed follow-up courts

### 9.1 Exact V6 sampler exposure court

Authority:

- `canonical/KNIGHT_ARACHNE_V6_SAMPLER_EXPOSURE_COURT_V1_20260929.json`
- workflow run `36532697901`
- exact V6 replay: 8,192 steps × 384 teacher-valid rows, seed 20261128.

After excluding six genuinely non-skinned target columns, the uniform-valid V6 schedule materially starves rare deforming supports.

Examples:

- target column 19, a `SKIN_SUPPORTED_ARTICULATION`:
  - only 18 teacher-valid rows have weight > 0.10;
  - exact V6 uniform sampling has **51.44%** of steps with zero >0.10 example for that target;
  - exposure-balanced K4 counterfactual reduces this to **0.95%** under the same row budget.
- at threshold >0.05, exact V6 has 20/28 columns with at least one zero-active step; after accounting for the six true zero-skin columns, 14 deforming targets suffer at least one zero-active step.
- exposure-balanced K4 at >0.05 leaves only the six true zero-skin columns with zero-active steps.

This proves a genuine active-support exposure mismatch between SkinTokens-style training and Knight V6. It does **not** prove performance improvement until a matched retrain is run.

### 9.2 Teacher support coverage court

Authority:

- `canonical/KNIGHT_ARACHNE_TEACHER_SUPPORT_COVERAGE_COURT_V1_20260929.json`
- exact teacher-bank ↔ Geppetto target positional alignment error: 0.

Result at weight >0:

- target columns with any valid positive support: **22**
- true zero-support target columns: **6**
- invalid-only-support target columns: **0**

The six true zero-skin columns are exactly:

- 1 × `ROOT_MOTION_ANCHOR`
- 5 × `TERMINAL_TIP_SYNTHETIC`

Therefore the sampler court's six permanently-zero columns are legitimate non-skinned controls, not Arachne failures.

The remaining 22 positive-support targets correspond exactly to the mechanically relevant skin-bearing side of the 28-target rig (20 skin-supported articulations + 2 terminal extension sources). Sparse-support starvation is therefore a real issue on deforming targets, not an artifact of helper controls.

### 9.3 The 1,479-row hole is cross-topology teacher-transfer coverage

`ARACHNE_KNIGHT_TEACHER_PROJECTION_REPORT_FIX2.json` seals:

- rule: `EXACT_SOURCE_TRIANGLE_CLOSEST_POINT_BARYCENTRIC_WITH_COMPONENT_MECHANICAL_ATTACHMENT`;
- Stage15 rows: 12,090;
- clean / teacher-valid rows: 10,611;
- coverage: 0.87766749;
- status: `PASS_WITH_COVERAGE_WARNING`;
- `coverage_is_teacher_supervision_mask_not_product_gate = true`.

The bank actually contains a weight vector for every Stage15 row. The mask rejects rows whose closest-source-surface distance is beyond the frozen clean-distance rule. V6 then removes those rows from optimization and science metrics but still predicts and compiler-qualifies all rows.

This means the strongest training-side owner is more precisely:

`CROSS_TOPOLOGY_TEACHER_TRANSFER_COVERAGE_HOLE + RARE_ACTIVE_SUPPORT_STARVATION`.

SkinTokens' native weighted-mesh setup does not require this RealSaS-specific source-topology → product-surface supervision transfer, so copying its sampler alone cannot close our 12.23% hole.

### 9.4 Updated causal order

The next matched training court should separate two changes rather than combine them:

1. `UNIFORM_VALID` — exact V6 baseline.
2. `ACTIVE_BALANCED_VALID` — change sampling only; retain the 10,611-row supervision mask.
3. `UNIFORM_ALL_PROJECTED` — change coverage only; train on all 12,090 projected teacher rows.
4. `ACTIVE_BALANCED_ALL_PROJECTED` — combine total projected coverage with active-support balancing.

This 2×2 court identifies whether the catastrophic Knight failure is principally caused by:
- support starvation within trusted rows,
- the supervision mask itself,
- or both.

Because the invalid-row oracle already shows that those projected bank weights close downstream topology when substituted at inference, `UNIFORM_ALL_PROJECTED` is a particularly important causal arm. It remains demo/FIT evidence; it must not be generalized into a product teacher-transfer policy without cross-subject validation.

Only after this 2×2 court should we add a SkinTokens-like stronger random-pose max-edge consequence objective or a new reliability head.


### 9.5 Exact V6 masked-row error regime

The exact V6 final canonical weights were aligned back to the exact teacher bank by:

1. exact `surface_id` lookup, and
2. the sealed `canonical_joint_to_target_index` mapping from `ARACHNE_KNIGHT_V6_RESULT.json`.

This reproduces the published V6 valid-row p95 to numerical precision and exposes the hidden excluded-row regime.

Teacher-valid 10,611 rows:
- mean row L1: **0.0100**
- p95: **0.046917**
- p99: **0.15736**
- max: **1.99398**
- row L1 > 1: **3**
- dominant-joint accuracy: **0.998775**

Teacher-invalid 1,479 rows:
- mean row L1: **0.20628**
- p95: **1.99959**
- p99: **~2.0**
- row L1 > 1: **151**
- dominant-joint accuracy: **0.90264**

All 12,090 rows:
- p95: **0.06353**
- p99: **1.65054**
- row L1 > 1: **154**

For simplex skin rows, L1=2 is the maximum possible distance between two one-hot assignments. Therefore the excluded set contains a distinct near-maximal catastrophic tail that the historical V6 science gate cannot observe because `base._metrics(pred, truth, valid, ...)` is explicitly evaluated through the same teacher-valid mask.

This is consistent with the downstream 12,049-vertex product court where 1,474 surviving invalid rows own essentially all catastrophic missed seams.
