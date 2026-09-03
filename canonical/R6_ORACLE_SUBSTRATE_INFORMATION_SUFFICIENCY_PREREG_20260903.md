# RealSaS — R6 Oracle-Substrate Information Sufficiency Prereg — 2026-09-03

**Status:** `FROZEN_BEFORE_FIRST_U0_U1_RESULT__NO_REFREEZE__NO_REAL_FAMILY_EVIDENCE`

## 1. Scientific question

The remaining clean-room P0 is not whether RealSaS reconstructs a closed mesh. It does not and must not acquire that objective.

The question is:

> Does the information available from the frozen eight-view observation contract suffice for the admitted rigging task when perception error is removed, compared with a reference-class full-surface geometry upper bound?

The causal arms remain those frozen in `GEPPETTO_ARACHNE_EXTERNAL_REFERENCE_AUDIT_CLOSURE_20260831.md`:

- `U0_REFERENCE_FULL_SURFACE` — diagnostic rich-surface upper bound;
- `U1_OBSERVATION_ORACLE_SUBSTRATE` — shipping-observable oracle substrate;
- `U2_PREDICTED_IRIS_SUBSTRATE` — forbidden until U1 closes and predicted IRIS is separately qualified.

This prereg implements the synthetic behavioral precursor of U0/U1 before any fresh real-family evidence.

## 2. Non-goals and leakage firewall

Forbidden:

- closed-mesh reconstruction as a RealSaS product target;
- hidden/back-surface completion inside U1;
- source mesh / teacher topology entering Geppetto conditioning;
- direct teacher `P` or teacher `N` injection into U1 `RiggingSurfaceIR`;
- teacher joint identity becoming final Compiler graph authority;
- changing witness geometry, cameras, visibility policy, thresholds, optimizer or stability rule after seeing results.

U0 may use authoritative full-surface `P+N` because it is explicitly a diagnostic upper bound. U1 may use authoritative geometry only to render exact observation evidence; consumer geometry must then be rebuilt through the shipping deterministic observation route.

## 3. Synthetic mechanical witnesses

Reuse the already preregistered mechanical families and seeds from the Arachne behavioral panel:

- `chain_blend_3` — seed `20260921`;
- `branch_blend_4` — seed `20260922`;
- `sharp_fork_5` — seed `20260923`.

The joint positions and parent vectors are unchanged.

A closed reference shell is generated deterministically around the same family centerline using a union-of-capsules carrier. This shell exists only to define reference-class surface information and exact eight-view rendering; it is not a product mesh.

Frozen centerline graph edges:

- chain: `(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)`;
- branch: `(0,1),(1,2),(2,3),(3,4),(4,5),(2,6),(6,7),(7,8),(2,9),(9,10),(10,11)`;
- fork: `(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(4,10),(10,11),(11,12),(12,13),(13,14)`.

Frozen shell parameters:

- capsule radius: `0.075` object-frame units;
- cylinder longitudinal samples per edge: `3` interior levels;
- cylinder azimuth samples: `8`;
- joint-sphere latitude samples: `4` non-polar levels plus poles;
- joint-sphere azimuth samples: `8`;
- candidate points strictly inside another capsule by more than `1e-5` are culled;
- duplicate positions are canonicalized on a `1e-5` coordinate grid;
- if more than `160` full-surface samples remain, deterministic farthest-point sampling selects exactly `160`, beginning at lexicographically smallest coordinate and resolving equal distances lexicographically.

No shell parameter may be changed in response to U0/U1 result.

## 4. Frozen eight-view camera and visibility contract

Eight orthographic cameras are placed at equally spaced azimuths around the object Y axis:

`0°,45°,90°,135°,180°,225°,270°,315°`.

For every view:

- camera forward points from camera plane toward the object origin;
- world up is `+Y`;
- right is the normalized camera-right basis;
- one common object framing is derived once from U0 full-surface bounds and reused unchanged for U1;
- raster resolution: `64 x 64`;
- framing margin: `10%` beyond the maximum projected U0 extent;
- projection coordinates are continuous pixel-center coordinates; raster z-buffer ownership uses nearest integer pixel;
- front-facing requirement: `dot(outward_normal, ray_forward) < -1e-6`;
- among front-facing samples mapping to one raster pixel, the smallest positive orthographic depth owns that pixel;
- a sample is U1-visible in a view iff it owns at least one pixel in that view.

A U0 sample absent from all eight view-visible sets is **not** available to U1.

No visibility dilation, hidden completion, back-fill or source-surface neighbor propagation is allowed.

## 5. U0 construction

`U0_REFERENCE_FULL_SURFACE` uses the complete sampled capsule-union shell.

Each full-surface node carries:

- authoritative shell `P`;
- authoritative analytic outward normal `N`;
- all eight exact camera projections as diagnostic raster bindings;
- a distinct U0 full-surface normal/operator provenance label.

It is converted only into the existing `RiggingSurfaceIR` consumer format. No source-rig identity is included.

## 6. U1 construction — actual deterministic product route

For every U1-visible full-surface sample and every view in which it is visible, construct an exact `ObservationSample`:

- exact orthographic `ray_origin` from the frozen camera plane and projected sample coordinate;
- exact `ray_forward`;
- `depth` satisfying `P = O + dF` exactly up to floating precision;
- `support=True`;
- raster binding from the frozen camera projection;
- oracle provenance identifies only synthetic observation/sample/view identity.

All observations of the same physical shell sample form one oracle hypothesis group. This uses oracle correspondence only to remove IRIS prediction error; it does not provide hidden geometry.

U1 is then compiled through the actual shipping deterministic route:

`ObservationEvidenceIR`
`-> build_persistence_groups_v2`
`-> build_surface_from_persistence / analytic P=O+dF`
`-> attach_observed_local_relations_v2 when available`
`-> attach_dtb_nd1_from_evidence`
`-> RiggingSurfaceIR S_U1`.

Common-frame persistence tolerance for exact synthetic oracle observations: `1e-6`.

Teacher/full-surface normals are forbidden from being copied into `S_U1`.

Normal availability, support-view count, visible/full-surface coverage and relation count are telemetry; product PASS is downstream mechanical behavior.

## 7. Geppetto consumer gate — first executable P0

Use **default shipping Geppetto**, not the historical tiny behavioral surrogate:

- `GeppettoCandidateConfigV2()` defaults;
- model dim `192`;
- kNN `16`;
- local layers `2`;
- global layers `2`;
- decoder layers `2`;
- heads `6`;
- feedforward dim `576`;
- position modes `3`.

Training protocol is frozen from the existing Geppetto hardening harness:

- AdamW LR `3e-4`;
- weight decay `1e-4`;
- maximum steps `2048`;
- evaluation every `32` steps plus step `1`;
- deterministic CPU algorithms;
- three consecutive PASS evaluations required.

Teacher graph equality remains diagnostic only.

Geppetto product PASS per arm requires:

1. generated control count equals the deliberately minimal synthetic control count;
2. anonymous shipping joint-locus p95 is below `0.5 * nearest_teacher_joint_separation`;
3. actual `model.propose()` succeeds;
4. actual `Compiler.qualify_skeleton_v2()` succeeds with the expected control count;
5. at least one deform root exists;
6. no illegal parent reference exists;
7. every qualified joint has non-empty admitted surface support;
8. criteria 1–7 hold for three consecutive checks.

No teacher-parent equality is required for product PASS.

## 8. Execution order and causal interpretation

Frozen order:

### R6-G-SYN-1

One-family ceiling uses `branch_blend_4` only because it is preregistered before results and contains nontrivial branching.

1. U0 `branch_blend_4`;
2. U1 `branch_blend_4`.

Interpretation:

- U0 FAIL -> shipping Geppetto apparatus/representation remains inadequate; stop U1 interpretation;
- U0 PASS + U1 FAIL -> observation-limited substrate information insufficiency is demonstrated on this gate;
- U0 PASS + U1 PASS -> hidden/full surface is not necessary for this synthetic branch task.

### R6-G-SYN-HET

Only if R6-G-SYN-1 U0 and U1 both PASS:

Run U1 on all three frozen families (`chain_blend_3`, `branch_blend_4`, `sharp_fork_5`) with the same protocol and acceptance criteria.

All three must PASS to close the synthetic Geppetto observation-substrate gate.

## 9. Arachne U0/U1 follows Geppetto, not in parallel

Arachne/Codec U0/U1 is not interpreted until the Geppetto synthetic substrate gate above closes.

When opened, it must use:

- Compiler-qualified skeleton input, not source-rig identity;
- shipping Codec config hash `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`;
- shipping Arachne config hash `ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`;
- raw W and Compiler-qualified W separately;
- verified LBS deformation as product authority;
- no hidden surface W required for U1: only admitted U1 surface nodes are evaluated.

Its exact execution fixture will be a separate prereg amendment before its first result, after Geppetto U1 closes.

## 10. What a U1 failure means

A U1 failure does **not** authorize closed-mesh reconstruction.

It opens a causal information audit. The missing quantity must be classified first, for example:

- insufficient observed surface coverage;
- inadequate deterministic normal/local differential recovery;
- insufficient multi-view support/persistence information;
- missing occlusion/discontinuity evidence;
- consumer sensitivity to sampling density;
- another explicitly identified mechanical information class.

Only the demonstrated missing function may be repaired.

## 11. Freeze relationship

Architecture refreeze remains blocked while this gate is unresolved.

Repository/runtime historical source restoration remains separately deferred until after architecture freeze, per user-approved sequence. It may not interrupt or change this scientific gate.
