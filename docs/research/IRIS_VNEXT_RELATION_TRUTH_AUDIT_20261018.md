# IRIS vNext Relation Truth Audit — 2026-10-18

Status: **CURRENT CORRECTION / REPLACEMENT AUTHORITY FOR RELATION-TRUTH INTERPRETATION**

This note corrects an over-broad shorthand that appeared during the first post-V8 audit. There are **two distinct historical relation-truth producer paths** and they must not be conflated.

## 1. `models/iris/vnext/graphics_truth_v1.py`

For adjacent faces, this helper derives:

```text
same_drawable_affinity = 1 iff anonymous_drawable_id[a] == anonymous_drawable_id[b]
cut_probability        = 1 - same_drawable_affinity
```

Within this helper, `cut` is therefore a tautological complement of the anonymous owner-equivalence proxy. It is not independent causal cut truth.

## 2. Active Fit8 shard builder used by the V8 lineage

`tools/iris_vnext/build_fit8_vnext_shards_v1.py::_build_view_banks`, inherited by the active v2 builder for relation rows, uses a different target construction.

For two-hit rows it computes:

```text
same = owner_a == owner_b
material_boundary = (owner_a != owner_b) OR (material_id_a != material_id_b)
relation_cut_probability = material_boundary
```

For adjacency fallback rows it similarly uses:

```text
boundary = (owner_a != owner_b) OR (material_id_a != material_id_b)
relation_cut_probability = boundary
```

Therefore **V8 cut supervision must not be described as simply `1 - same_drawable`**.

However the active target is still not clean causal cut truth. It mixes at least:

- anonymous source-object ownership change;
- authored material-region change.

A material boundary is not necessarily a drawable cut, and a source object boundary is not automatically presentation ownership.

## 3. Correct interpretation of V8 relation results

The V8 `cut` metric demonstrates decodability of the historical **owner-or-material boundary proxy** from sealed H. It does not establish that IRIS learned authoritative drawable cuts.

Likewise `same_drawable` under the current teacher should be interpreted only within the authority of the anonymous owner proxy that produced it.

This strengthens—not weakens—the case for a validity-aware truth replacement.

## 4. Replacement contract

New source:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

V2 requires every scalar relation target to carry:

- probability value;
- evidence authority;
- provenance;
- `valid_for_training`.

Unavailable targets abstain. Diagnostic-only evidence cannot be marked trainable. The V2 relation loss requires an explicit validity mask per family and refuses to turn missing truth into negative supervision.

Historical V1/V8 shards remain immutable provenance. No migration rewrites their meaning.

## 5. Boundary Causality relationship

`BoundaryCausalityTeacherV1` supplies provable boundary **causes** such as:

- material boundary;
- alpha boundary;
- geometric termination;
- occlusion/visibility boundary;
- source primitive/component seam;
- residual unknown.

These causes do not automatically imply `drawable cut`.

Future cut truth is valid only when an independent source contract actually identifies the structural relation being trained. Otherwise `cut` must abstain even if another kind of boundary is known.

## 6. Current decision

- preserve historical targets exactly for V8 interpretation;
- stop describing the active V8 cut target as pure `1 - same`;
- classify both historical producers as proxy truth with different semantics;
- use Graphics Truth V2 for all new relation-target work;
- do not connect V2 to a new training run until its concrete producer supplies explicit provenance/validity statistics and held-out courts.
