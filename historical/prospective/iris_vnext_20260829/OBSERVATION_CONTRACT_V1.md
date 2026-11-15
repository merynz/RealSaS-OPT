# RealSaS — ObservationContract V1

**Status:** PREREGISTERED_INPUT_BOUNDARY  
**Purpose:** Make explicit what the current IRIS research/product guarantee assumes about incoming 8-view observations.

This contract is upstream of IRIS.

IRIS is NOT responsible for estimating semantic parts, canonical identity, skeleton meaning, or camera pose inside its depth head.

---

# 1. Two product input modes

## Mode G — GUARANTEED / CONTROLLED

This is the scientific authority for the current IRIS V-next program.

Requirements:

- exactly 8 ordered views;
- native raster: `1024 x 1024 RGBA`;
- known exact orthographic camera basis for every view;
- camera metadata originates from the RealSaS-controlled renderer or another source that can provide the same exact camera contract;
- no unmodelled per-view yaw/pitch/roll uncertainty;
- authoritative alpha/support is available on the native raster;
- no adaptive per-asset crop/zoom is applied before the foundation path;
- the full authoritative 1024 canvas is deterministically resized to the frozen foundation raster;
- observation-set quality passes the frozen cross-view preflight.

For Mode G:

`camera_uncertainty_set U_v = {c_v}`

is a singleton.

The standard visual hull is therefore:

`V_hull = intersection_v backproject(S_v, c_v)`.

Current research claims apply to Mode G only unless a later contract explicitly promotes another mode.

### Camera tolerance in V1

For guaranteed claims:

`UNMODELLED_CAMERA_ERROR_TOLERANCE = 0`.

This does NOT mean the future product must reject a user who drew 41° instead of 45°.
It means that such an input is not silently treated as if its camera were exact.

Any nonzero camera uncertainty is routed to Mode E / a future versioned camera-uncertainty contract.

---

## Mode E — EXTERNAL / BEST-EFFORT

Examples:

- artist-drawn 8-view sheets;
- AI-generated 8-view sheets;
- exported views whose camera metadata is missing or approximate;
- nominal 45° views that are actually 41°, 51°, etc.;
- stylized/asymmetric views that may not admit one exact common 3D explanation.

Mode E does NOT inherit Mode G guarantees.

It enters:

`ObservationPreflight`
`-> CameraQualification`
`-> CrossViewConsistencyQualification`
`-> IRIS only if admissible`.

Possible outcomes:

- `QUALIFIED_WITH_CAMERA_UNCERTAINTY`
- `BEST_EFFORT`
- `ABSTAIN_INPUT_INCONSISTENT`

The input mode and camera uncertainty must be carried in provenance.

No external set may be silently relabelled as Mode G.

---

# 2. CameraQualification is a separate upstream contract

Do not add camera pose estimation as another IRIS geometry authority.

Camera qualification may output:

`U_v = admissible set/distribution of camera parameters for view v`.

For a controlled exact view:

`U_v` is a singleton.

For an uncertain external view:

`U_v` has nonzero width.

The camera estimator/qualifier is versioned independently from IRIS.

Jointly training camera estimation and depth is NOT part of ObservationContract V1.

---

# 3. Robust visual hull under camera uncertainty

Using one estimated camera point estimate for uncertain input can over-carve and destroy the one-way outer-envelope property.

Therefore an uncertainty-aware hull must be conservative.

Let:
- `S_v+` be the conservative foreground silhouette after alpha/boundary dilation required by the observation uncertainty contract;
- `U_v` be the admissible camera set.

Define the per-view conservative backprojection:

`B_v^robust = union_(c in U_v) backproject(S_v+, c)`.

Then:

`V_hull^robust = intersection_v B_v^robust`.

If the true camera is contained in every `U_v` and the true silhouette is contained in every `S_v+`, this construction preserves an outer-envelope interpretation.

Do NOT use:

`intersection_(c in U_v) backproject(S_v, c)`

because it is anti-conservative and can cut away true volume.

The exact dilation/uncertainty calibration is NOT part of Mode G V1 because `U_v` is a singleton there.
Any promoted nonzero uncertainty policy is a new versioned observation contract.

---

# 4. Alpha / silhouette authority

Native 1024 alpha remains observation authority.

Use typed support:

- `CERTAIN_BACKGROUND`
- `BOUNDARY_UNCERTAIN`
- `CERTAIN_FOREGROUND`

Hard free-space exclusion may use only `CERTAIN_BACKGROUND`.

A false-positive foreground pixel expands the hull.
A false-negative foreground pixel can remove true admissible volume.

Therefore the uncertainty policy is intentionally conservative.

---

# 5. Cross-view consistency as input qualification

A set of 8 views can be internally contradictory even when every individual image is high quality.

Examples:
- a hand changes shape between views;
- an accessory appears/disappears;
- left/right proportions drift;
- AI-generated views do not correspond to any single common object;
- artist asymmetry exceeds what one coherent 2.5D/3D substrate can explain.

This is not ordinary sensor noise.

## Frozen score definition

After a candidate depth system is available, compute the family-level post-IRIS directed consistency score:

`Q_XV(f) = max_i median_(j != i) Q95( e_reproj(i->j) / image_diagonal )`

using:
- frozen directed visibility logic;
- frozen boundary sampling rule;
- native support masks;
- the same definition for every candidate.

This score is model-version-aware: a bad depth model can also make it large.
Therefore it is an **input-admissibility diagnostic**, not independent proof that the input itself is contradictory.

Also report model-independent silhouette/support contradictions separately where available.

## Threshold derivation rule

Do not invent a numeric threshold before data exists.

The threshold operator is frozen now:

`tau_XV(model_version) = Q99 of Q_XV on OBS_CAL_V1 Mode-G families

using a calibration split that is not used to train the candidate.

The numeric threshold is filled once that calibration split is legally opened.

Changing the score definition or threshold derivation rule creates a new ObservationContract version.

For Mode :
- `Q_XV <= tau_XV` is required for normal best-effort continuation;
- persistent violation triggers `BEST_EFFORT_LOW_CONFIDENCE` or `ABSTAIN_INPUT_INCONSISTENT` according to the frozen product policy.

---

# 6. Appearance domain

## In-domain appearance for scientific PASS

Current primary scientific selection is based on the frozen RealSaS-controlled render/corpus domain.

This is intentional and does not claim open-world artwork robustness.

## OOD-GT appearance probe

Create a small, sealed, family-disjoint probe where exact geometry/cameras remain known but appearance is outside the training style family.

Examples:
- unseen cel-shading pipelines;
- ink/line-art variants;
- palette/texture regimes not used in training;
- stylized contour treatments;
- controlled synthetic appearance transformations that preserve geometry.

Requirements:
- never used for training or hyperparameter selection;
- exact geometry retained;
- same 8-view camera contract;
- sealed before Phase 1 candidate results.

Role:
- separate OOD axis;
- preregistered tiebreaker among primary-PASS candidates;
- never allowed to rescue a primary FAIL.

## OOD-REAL exploratory probe

Separately maintain external artist/AI-generated 8-view sets that may not possess authoritative geometry.

These can measure:
- ObservationPreflight quality;
- Q_XV;
- abstention rate;
- qualitative downstream behavior.

They do NOT rank scientific candidates when geometry truth is unavailable.

---

# 7. Foundation raster and framing

The foundation extractor receives:

`518 x 518 RGB`.

Reason:
`518 = 37 x 14`

so all DINOv2 patch-14 rungs use the exact same `37 x 37` patch grid.

Framing is frozen:

- start from the full authoritative `1024 x 1024` canvas;
- NO per-asset adaptive zoom;
- NO per-view recrop;
- deterministic antialiased whole-canvas resize to `518 x 518`;
- all 8 views retain the same canvas geometry.

This keeps the effective framing identical across DINO rungs.

The native-detail/support path remains `1024 x 1024`.

Any future adaptive crop/zoom is an ObservationContract change.

---

# 8. Structure-scale reporting

Because the 518 foundation path can suppress sub-patch details while the native 1024 path remains identical across capacity rungs, capacity conclusions must be stratified by structure scale.

Define local native-1024 foreground thickness from the frozen alpha distance-transform operator.

One 14-pixel patch at 518 corresponds to approximately:

`14 * 1024 / 518 ≈ 27.7 native pixels`.

Freeze scale strata:

- `SUBPATCH_THIN`: local diameter `< 28 px` at native 1024;
- `ONE_TO_TWO_PATCH`: `28 <= diameter < 56 px`;
- `BROAD`: `diameter >= 56 px`.

Report the fixed geometry metrics separately for these strata.

Interpretation firewall:

> little/no Sₒg improvement on `SUBPATCH_THIN` cannot by itself show that foundation capacity is irrelevant, because those structures may be primarily carried by the shared native-1024 path.

The primary model PASS operator remains family-level direct consumer replay.

Scale strata are causal diagnostics, not alternate PASS rules.

---

# 9. Versioning rule

ObservationContract V1 is the controlled exact-camera research contract.

Any of the following requires a new contract version:

- nonzero admitted camera uncertainty;
- camera estimator promotion;
- robust-hull dilation policy;
- changed view count/placement;
- changed native raster;
- adaptive framing/cropping;
- changed alpha/support authority;
- changed cross-view admissibility rule;
- expansion of the claimed appearance domain.

No result obtained under V1 automatically transfers to V2.
