# IRIS V-next — Controlled Geometry-Prior Experiment Plan V4

This plan incorporates ObservationContract V1 and supersedes V3.

Read first:
`OBSERVATION_CONTRACT_V1.md`

The research authority remains the controlled Mode-G observation contract:
8 exact orthographic views, native 1024 RGBA, exact cameras, fixed framing.

---

# A. What the current program covers

The current IRIS program covers:

`ObservationContract V1 / Mode G`
`-> visual/geometry prior`
`-> fixed 8-view fusion`
`-> authoritative depth d`
`-> analytic P`
`-> delivered N_d`
`-> calibrated task-agnostic depth risk`
`-> SurfaceBuilder/persistence`
`-> VisualHullEnvelope + conservative depth refinement`
`-> Geppetto`
`-> Compiler`
`-> Arachne`
`-> Compiler/proof/runtime`.

External uncertain-camera artwork is explicitly NOT silently absorbed into this guarantee.
It is a versioned ObservationPreflight / best-effort problem upstream of IRIS.

---

# PHASE 0S — Sacrificial residual-shape pilot

Unchanged from V3.

Use permanently excluded `SAC_TRAIN` / `SAC_SHAPE` to freeze `CORRUPTION_GENERATOR_V1` before any candidate evaluation.

The generator may model:
- edge concentration;
- grazing concentration;
- occlusion correlation;
- spatial spectra;
- cross-view covariance;
- view asymmetry.

No product semantics.

---

# PHASE 0A — Consumer profiles

Unchanged core rule:

A safety object belongs to:
`substrate contract + exact Geppetto/Arachne checkpoints + compiler/proof policy`.

Freeze C0 clean and prospective C1 robust-substrate.

Any downstream retrain creates a new consumer profile and new safety object.

---

# PHASE 0B — Residual-field safety bridge

The authority is direct replay of the complete family residual field.

`SAFE_C = {R : Q_C(R) is safe}`

The low-dimensional corruption manifold is a chart/training instrument, not product safety authority.

`theta_Nd` and projected error are derived observables, not independent Cartesian sweep axes.

---

# MODEL-LEVEL PASS OPERATOR

Atomic cell:
`one family / all 8 views`.

For every family:
`actual predicted residual field R_f -> exact frozen consumer replay Q_C(R_f)`.

Hard gate:
`UNSAFE_ACCEPT_COUNT = 0`.

Coverage gate:
`>= 95% QUALIFIED family cells`.

`SAFE_ABSTAIN` counts against coverage.

Candidate ordering among PASS models:
1. higher qualified-family coverage;
2. lower across-family P95 projected error diagnostic;
3. lower across-family P95 delivered `N_d` angular error;
4. lower across-family P95 along-ray error;
5. smaller/faster model.

No later summary substitution.

---

# NEW: dual risk-band reporting

Phase 1A asks a representation question.
A candidate-specific risk head can confound that question.

Therefore every candidate is replayed twice.

## Replay R_native

Use the candidate's own calibrated depth-risk field.

Report:
- `QUALIFIED_COVERAGE_NATIVE_RISK`
- `SAFE_ABSTAIN_NATIVE_RISK`
- `UNSAFE_ACCEPT_NATIVE_RISK`.

This is product-like behavior.

## Replay R_common
Replace the candidate-specific risk field with one frozen candidate-agnostic common band policy.

Define before Phase 1:

`COMMON_BAND_V1(f,u) = kappa_common * D_hull_f`

where:
- `D_hull_f` is the frozen family visual-hull diagonal;
- `kappa_common` is derived only from the primary Phase-0 consumer bridge / OBS_CAL geometry authority;
- candidate residuals are not used to choose `kappa_common`.

The exact `kappa_common` derivation rule and value are sealed before candidate results.

All candidates use identical `COMMON_BAND_V1`.

Report:
- `QUALIFIED_COVERAGE_COMMON_BAND`
- `SAFE_ABSTAIN_COMMON_BAND`
- `UNSAFE_ACCEPT_COMMON_BAND`.

## Scientific interpretation

Primary representation-capacity interpretation uses `COMMON_BAND_V1`.

Product-readiness interpretation uses `NATIVE_RISK`.

If the rankings differ, report the disagreement explicitly:
the ladder is separating geometry representation quality from risk-calibration behavior.

A model cannot product-PASS with unsafe native-risk behavior merely because common-band replay looks good.

---

# PHASE 1A — Frozen DINOv2 representation-capacity ladder

Rungs:
- S / 384-d
- B / 768-d
- L / 1024-d
- g / 1536-d

Claim boundary remains:

Phase 1A measures frozen representation accessibility under a fixed RealSaS system.

`g<=B` MUST NOT be called capacity-hypothesis falsification.

---

# Foundation interface

## Frozen raster/grid

`FOUNDATION_RASTER = 518 x 518`

`37 x 37` patch-14 grid.

`NATIVE_DETAIL_RASTER = 1024 x 1024`.

The foundation path and native-detail path have different responsibilities.

## Framing

Use ObservationContract V1 framing:
- full native 1024 canvas;
- no adaptive crop/zoom;
- no per-view recrop;
- deterministic whole-canvas antialiased resize to 518.

Thus every rung sees the same framing and patch grid.

---

# Trainable-capacity / scale control

Per rung:

`token`
`-> fixed isometric Q_k`
`-> same non-affine post-lift normalization in 1536-d`
`-> identical trainable fusion`.

Trainable parameter count after extractor must be identical.

Seal:
- Q_k;
- normalization;
- tensor shapes;
- trainable parameter count.

---

# NEW: structure-scale diagnostics

The 518 foundation path may discard information that remains visible only to the shared native-1024 path.

Therefore freeze native-thickness strata using the ObservationContract V1 operator:

- `SUBPATCH_THIN`: `< 28 px`;
- `ONE_TO_TWO_PATCH`: `28–55 px`;
- `BROAD`: `>= 56 px`.

For every rung report:
- projected residual;
- delivered N_d angular residual;
- depth residual;
- high-frequency surface-energy retention

separately for all three structure scales.

Interpretation firewall:

> No rung difference on sub-patch-thin structures does not establish that prior capacity is irrelevant; those structures may bypass the discriminating foundation path and be recovered primarily by the identical native-1024 branch.

Also report where S/B/L/g differences actually concentrate.

---

# Frozen-feature cache

Foundation raster is 518 and each family has 10,952 patch tokens across 8 views.

Approximate raw FP16/BF16 pre-Q cache:
- S: ~8.0 MiB/family
- B: ~16.0 MiB/family
- L: ~21.4 MiB/family
- g: ~32.1 MiB/family.

Run/cache one rung at a time.
Seal manifests/hashes.
Cached-vs-online preflight occurs at scientific optimizer step 0.

---

# PHASE 1A appearance-domain reporting

Primary scientific PASS remains on the in-domain controlled corpus.

Add the sealed ObservationContract probes:

## OOD-GT

Known geometry, exact cameras, unseen appearance family.

Role:
- separately reported OOD axis;
- preregistered tiebreaker among primary-PASS models;
- cannot rescue a primary FAIL.

Tie order is extended only after the primary in-domain direct-replay criteria:
if candidates remain tied after the fixed in-domain ordering, use higher OOD-GT qualified coverage, then lower OOD-GT projected residual.

## OOD-REAL

Artist/AI external views without authoritative geometry.

Exploratory only:
- ObservationPreflight;
- cross-view consistency;
- abstention;
- qualitative downstream behavior.

No scientific model ranking authority.

---

# NEW: cross-view input quality reporting

For every candidate and every observation set, compute the frozen family score from ObservationContract V1:

`Q_XV(f)`.

For Mode-G research:
report it as an input/model-consistency diagnostic.

For future Mode-E product use:
the model-versioned threshold rule is:

`tau_XV = Q99(Q_XV on OBS_CAL_V1 Mode-G families)`.

The threshold numeric value is not invented before calibration data is legally opened.

This rule is sealed prospectively.

Do not interpret Q_XV alone as proof of input inconsistency because model error also contributes.

---

# PHASE 1B — Fine-tuning

Same as V3.

Phase 1A frozen readout and Phase 1B adaptation are separate scientific claims.

---

# PHASE 2 — Full geometry-system comparison

Compare:
- full MapAnything Apache stack;
- DA3-Base Apache stack;
- selected fixed-fusion DINO rung(s).

This remains a SYSTEM comparison, not encoder ablation.

All systems map to the same RealSaS contract:
`d -> analytic P -> frozen N_d -> calibrated band`.

## Perspective → orthographic transfer is explicit

This item MUST remain visible in both plan and architecture diagram.

Pretrained systems are perspective-heavy.

Canonical product geometry uses the ObservationContract's exact orthographic authority.

No unvalidated long-focal approximation becomes canonical geometry.

Any promoted uncertain-camera/external mode is versioned separately upstream.

---

# MapAnything encoder-only note

Same as V3:
do not count it as a clean fifth DINO capacity rung.

If tested:
`GEOMETRY-ADAPTED / TRUNCATED DINOv2-g-FAMILY INTERVENTION`.

Compare tensors against vanilla DINO before scientific interpretation.

---

# Visual hull / refined rigging volume

`VisualHullEnvelope`:
- native alpha + exact cameras only in Mode G;
- IRIS bypass;
- immutable evidence object.

`RefinedRiggingVolume`:
- hull minus only calibrated-certain front free-space;
- carve with `[d_lo, d_hi]`, never point `d_hat`.

Remember:
because `L_S` trains silhouette exclusion, model-vs-hull agreement is training-aligned and not independent validation.

---

# ObservationContract V1 firewall

Current guaranteed claims are only for Mode G:
- exact camera metadata;
- fixed 8-view order;
- exact 1024 native canvas;
- fixed 518 whole-canvas foundation resize;
- native authoritative alpha;
- versioned cross-view quality rule.

41B�/51° artist/AI inputs are NOT silently "approximately known cameras".

They require Mode E:
`ObservationPreflight -> CameraQualification -> uncertainty-aware hull -> best-effort/abstain`.

Future robust camera hull:

`V_hull_robust = intersection_v union_(c in U_v) backproject(S_v+, c)`.

This is a future ObservationContract version, not part of the current exact-camera qualification.

---

# PREDECLARED NO-PASS TREE

Same scientific rule:

1A no-pass -> 1B fine-tune, same consumer/operator.
1B no-pass -> Phase 2 full systems, same consumer/operator.
Phase 2 no-pass -> freeze NO_PASS_V1.

A new version may change exactly one major lever:
- consumer robustness;
- ObservationContract/render/camera/view contract;
- IRIS geometry contract.

ObservationContract changes invalidate automatic transfer of prior PASS claims.

---

# Final firewall

The system now distinguishes five authorities:

1. Observation authority — alpha + camera contract.
2. Learned geometry authority — depth d.
3. Derived geometry — P and N_d.
4. Geometric risk — calibrated residual band.
5. Product authority — Compiler/qualified downstream objects.

No lower layer may silently absorb the uncertainty/semantics of an upper or upstream layer.
