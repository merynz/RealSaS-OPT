# IRIS vNext Relation Truth Audit — 2026-10-18

Status: **CURRENT CORRECTION / REPLACEMENT AUTHORITY FOR RELATION-TRUTH INTERPRETATION**

This note records both historical producer semantics and a newly confirmed conditioning mismatch in the V1 relation-boundary path.

## 1. `models/iris/vnext/graphics_truth_v1.py`

For adjacent faces, this helper derives:

```text
same_drawable_affinity = 1 iff anonymous_drawable_id[a] == anonymous_drawable_id[b]
cut_probability        = 1 - same_drawable_affinity
```

Within this helper, `cut` is a tautological complement of anonymous owner-equivalence. It is not independent causal cut truth.

## 2. Fit8 shard builder used by the V8 lineage

`tools/iris_vnext/build_fit8_vnext_shards_v1.py::_build_view_banks`, inherited by the active v2 builder for relation rows, uses a different target construction.

For two-hit rows:

```text
same = owner_a == owner_b
cut proxy = (owner_a != owner_b) OR (material_id_a != material_id_b)
```

For adjacency fallback rows it uses the same owner-or-material boundary proxy.

Therefore **V8 cut supervision must not be described as simply `1 - same_drawable`**.

The active target is still not clean causal drawable-cut truth because it mixes:

- anonymous source-object ownership change;
- authored material-region change.

A material boundary is not necessarily a drawable cut; a source-object boundary is not automatically presentation ownership.

## 3. Newly confirmed conditioning mismatch in V1 relation boundaries

The V8 shard builder also derives relation boundary targets from the selected **render view**.

`_boundary_maps_v2()` computes local 3×3 raster evidence:

```text
alpha boundary      = local alpha range > 0.20
appearance boundary = max local RGB distance > 0.20
```

For two-hit relation rows, `_build_view_banks()` samples these flags at the current view pixel and writes them into:

```text
relation_alpha_boundary_probability
relation_appearance_boundary_probability
```

Material boundary is also OR-ed into relation appearance-boundary truth.

However `GraphicsRelationHeadV1.static_relations(feature_a, feature_b)` predicts:

```text
same_drawable_affinity
cut_probability
attachment_affinity
alpha_boundary_probability
appearance_boundary_probability
```

without any camera/view input.

Only `GraphicsRelationHeadV1.view_relations(...)` receives camera features, and that branch predicts front/occlusion/visibility.

`IRISVNextFit8TrainerV1.one_step()` confirms the mismatch: static boundary predictions are merged with camera-conditioned view predictions, then compared against relation targets sampled from one specific current render view.

### Consequence

A projected alpha or RGB-neighborhood boundary is not generally a view-invariant function of `(H_a, H_b)` alone. The same local 3D relation can project differently under another camera because silhouette, alpha and nearby image structure change.

Therefore relation-boundary residuals cannot be treated purely as an LR/capacity problem until this target/conditioning mismatch is removed.

This is particularly relevant to V8's pattern:

- front / occlusion / visibility: very low residuals;
- cut / alpha boundary / appearance boundary: materially worse.

This does **not** fully explain appearance-head boundary debt, because `AppearanceEvidenceHeadV1` is already camera-conditioned. It does, however, invalidate the assumption that the V1 relation static-boundary branch was correctly conditioned for its raster-derived targets.

## 4. Correct interpretation of V8 relation results

V8 `cut` demonstrates decodability of the historical owner-or-material boundary proxy from sealed H.

V8 relation alpha/appearance-boundary residuals combine at least two debts:

1. target causality/authority debt;
2. for raster-derived components, view-conditioning mismatch in the V1 relation architecture.

`same_drawable` remains interpretable only within the anonymous-owner proxy authority of the historical source.

## 5. Replacement contracts already landed

New source:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

V2 requires every scalar relation target to carry:

- value;
- evidence authority;
- provenance;
- `valid_for_training`.

Unavailable targets abstain. Diagnostic-only evidence cannot be trainable. The V2 relation loss consumes an explicit validity mask per family/sample and never converts missing truth into a negative target.

Historical V1/V8 shards remain immutable provenance.

## 6. Boundary Causality relationship

`BoundaryCausalityTeacherV1` supplies provable boundary **causes**:

- material boundary;
- alpha boundary;
- geometric termination;
- occlusion/visibility boundary;
- source primitive/component seam;
- residual unknown.

These causes do not automatically imply drawable cut.

Future target design must also separate **view-invariant structural causes** from **view-conditioned projected evidence**.

Recommended decomposition for the next relation/graphics contract:

```text
static / structural pair evidence
    same-component / qualified ownership when identifiable
    structural cut when identifiable
    authored attachment/contact
    material/geometric boundary causes

view-conditioned pair evidence
    front
    occlusion
    visibility
    projected alpha boundary
    projected appearance boundary
```

The existing view-conditioned `AppearanceEvidenceHeadV1` remains a separate per-point graphics evidence path and should be evaluated independently.

## 7. Next implementation rule

Do **not** modify historical `GraphicsRelationHeadV1` in place; it is part of V8 provenance.

Before focused relation retraining:

1. populate Graphics Truth V2 with explicit authority/provenance/validity;
2. split static structural boundary truth from view-conditioned projected boundary truth;
3. create a versioned relation-head/trainer path whose view-conditioned outputs receive camera features;
4. publish support/class-balance/abstention statistics;
5. preregister held-out courts;
6. only then tune LR/schedule/capacity.

This ordering prevents an optimization experiment from compensating for a supervision-contract error.
