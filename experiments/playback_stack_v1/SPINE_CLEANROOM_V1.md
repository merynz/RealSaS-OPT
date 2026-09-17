# Spine clean-room runtime contract v1

Status: ACTIVE DESIGN INPUT — NOT PRODUCT PASS

## Goal

Define a clean-room behavioural reference from public Spine documentation and observable exported/runtime behaviour, then use it to simplify the RealSaS compiler/runtime boundary before rerunning the full Mage closure.

This is not a Spine-compatible reimplementation and does not use Spine Runtime source code. The reference is intentionally limited to public concepts and externally documented behaviour.

## Clean-room boundary

Allowed reference material:

- public Spine User Guide / Runtimes Guide documentation;
- public API documentation describing observable concepts and state;
- exported-data behaviour that can be observed without inspecting implementation internals;
- RealSaS-owned artifacts, code and measurements.

Explicitly excluded:

- copying or translating Spine Runtime source code;
- decompilation or binary inspection;
- dependence on private/internal algorithms or data structures;
- reproducing non-public implementation details by inspection.

Public documentation consulted for this contract:

- https://esotericsoftware.com/spine-runtimes
- https://esotericsoftware.com/spine-basic-concepts
- https://esotericsoftware.com/spine-slots
- https://esotericsoftware.com/spine-clipping
- https://esotericsoftware.com/spine-api-reference

## Public behavioural observations used

1. Attachments are associated with slots; slots are associated with bones.
2. A slot provides attachment-selection state and contributes to draw order.
3. Skeleton draw order is a slot sequence, distinct from the bone hierarchy, and may change during animation.
4. Attachments are effectively asset payloads; per-instance state lives on skeleton/bone/slot state rather than requiring copies of the attachment asset.
5. Clipping is bounded by explicit slot ranges and its cost grows with clipped geometry complexity, so runtime geometry size matters.
6. Runtime playback consumes already-authored/exported skeleton and animation state; it is not the stage that rediscovers rigging, topology, weights, ownership or layer order.

These observations define behaviour only. They do not prescribe Spine's internal implementation.

---

# RealSaS clean-room conclusions

## CR-1 — Separate scientific truth from runtime representation

The dense compiler substrate is proof/qualification material, not automatically the shipping mesh.

For the current Mage path:

- dense zero-surface BODY (~500k faces) may remain a sealed scientific substrate;
- runtime/export BODY must be a separately qualified compact mesh;
- compact runtime geometry must retain provenance to the dense sealed truth;
- no compact mesh may silently replace the scientific authority.

Target runtime triangle count is product/platform dependent. For the current mobile-oriented demo, the intended order of magnitude is thousands to low tens of thousands of triangles, not hundreds of thousands.

## CR-2 — Immutable assets are stored once

Runtime package assets are immutable objects referenced by stable IDs:

- mesh topology;
- UVs / atlas bindings;
- static attachment payloads;
- bind/skin data;
- source/provenance hashes.

A view/frame/clip must not deep-copy these payloads merely to express pose, visibility or ordering.

## CR-3 — Explicit slot-like indirection

RealSaS uses a neutral `RuntimeSlot` abstraction derived from its own product contract.

A RuntimeSlot contains only runtime state and stable references:

- `slot_id`;
- `parent_joint_id` or deformable-owner binding;
- `attachment_id | null`;
- `setup_order` / current order key;
- visibility;
- optional clipping relation;
- compact per-frame overrides when qualification authorizes them.

Current Mage minimum slot/component set:

- BODY_UNDERLAY
- CAPE_FOREGROUND
- HAT_FOREGROUND
- BOOK_FOREGROUND
- STAFF_FOREGROUND

Rigid foreground uses canonical-parent carry. BODY remains deformable.

## CR-4 — Draw order is compiled runtime state

Per-view/per-frame draw order is persisted as a compact ordered sequence of slot IDs/indices.

Runtime must never infer draw order from:

- filenames;
- component names;
- mesh depth alone;
- geometry heuristics;
- fallback ordering.

Depth/z-buffer may still be used for qualification and rendering where explicitly required, but it is not a replacement for qualification-owned layer order.

## CR-5 — Validation is boundary-oriented, not recursively replayed

A sealed immutable artifact is fully validated once when admitted to a compiler/package boundary.

Downstream immutable composition verifies references and sealed hashes rather than recursively traversing and re-hashing the same 500k-face graph at every layer.

Allowed downstream checks include:

- expected artifact SHA / lineage hash;
- expected schema/status;
- expected parent authority;
- exact component ID set;
- exact view set;
- exact motion-state binding;
- fail-closed policy flags.

Forbidden product-path behaviour:

- repeatedly constructing multi-million-entry coverage sets for an already-qualified immutable mesh;
- recursively serializing the same dense mesh to establish every parent object's identity;
- re-running identical child validation at component -> direction -> set -> continuity -> product boundaries;
- rebuilding a full intermediate product that is immediately discarded by the next stage.

This is an optimization of proof composition, not a relaxation of proof requirements.

## CR-6 — Hash composition uses sealed child identities

Once a child artifact is sealed, parent identity should be composition-based.

Conceptually:

`H(schema, policy, child_sha_1, child_sha_2, ..., local_small_state)`

not:

`H(recursively_serialized_entire_child_object_graph_again)`

Full-content hashes remain authoritative at the sealing boundary. Parent composition hashes bind those immutable identities.

## CR-7 — Pose/frame state must stay compact

Runtime animation state should contain only data that changes:

- joint transforms or qualified baked deform samples;
- current slot attachment selections;
- draw-order deltas or ordered indices;
- visibility / clipping state;
- qualification bit and source hash references.

It must not duplicate topology, atlas payloads or static appearance bindings per frame.

## CR-8 — Clipping/visibility work is explicitly bounded

Because clipping cost scales with affected geometry, RealSaS must:

- keep clipping scopes explicit;
- keep runtime meshes compact;
- avoid clipping the dense scientific substrate in shipping playback;
- prefer prequalified visibility/order state when equivalent;
- fail closed rather than synthesizing hidden surfaces at runtime.

## CR-9 — Incremental compilation is part of the product contract

A change in one authority should invalidate only dependent stages.

Examples:

- motion change must not rebuild geometry/rig/skin;
- appearance change must not retrain/recompute mechanics;
- rigid attachment change must not rebuild dense BODY;
- export-format change must not re-run scientific qualification when sealed inputs are unchanged.

## CR-10 — Runtime does playback; compiler does reasoning

The RealSaS native runtime remains intentionally simple:

- no topology discovery;
- no rig inference;
- no weight inference;
- no ownership inference;
- no draw-order inference;
- no scientific solver replay.

It consumes a compact product contract exactly.

---

# Performance contract

Optimization is a first-class product requirement, not a cosmetic follow-up.

For a sealed current-Mage chain, the closure/export stages should be bounded by metadata composition and compact runtime conversion, not by repeated dense-graph traversal.

Preregistered engineering targets for the current workstation class (8 GB RAM class, GTX 1660 Ti; CPU-bound closure):

- no closure stage may intentionally require multi-hour runtime;
- no final composition stage may require O(full_dense_graph) repeated serialization more than once per newly sealed child artifact;
- peak RAM must remain bounded enough to avoid swap-thrash on the reference workstation;
- every stage longer than 10 s must emit progress telemetry;
- every persisted stage must be restartable from already sealed predecessor artifacts;
- interrupted final closure must not force regeneration of earlier sealed body/derivation/assembly stages.

These are architecture gates. Exact wall-clock thresholds will be tightened after profiling the first corrected run.

# Required instrumentation

Each major compiler stage must emit:

- stage name;
- monotonic elapsed time;
- input sealed hashes;
- output hash when available;
- resident-memory snapshot where practical;
- explicit indication of whether full-content validation, cached validation or seal-chain validation was used.

A performance regression must be visible in canonical reports rather than inferred from a silent terminal.

# Mage v1 application

The next full Mage rerun must use this clean-room contract as the design constraint:

1. preserve dense BODY as scientific authority;
2. do not treat dense BODY as the eventual shipping representation;
3. bind BODY + CAPE + HAT + BOOK + STAFF through explicit runtime slots/components;
4. persist qualification-owned per-view/per-frame order;
5. keep rigid carry parented to canonical joints;
6. preserve `UNSEEN` fail-closed behaviour — do not hallucinate missing appearance;
7. avoid recursive dense validation/hash replay in V4 closure;
8. make the final product closure restartable and observable;
9. after the full product is materialized, render the full Mage and separately measure residual BODY holes against the previous ABC/D0 visibility diagnosis;
10. only after that visual test decide whether visibility/selection architecture is actually fixed.

# Non-claims

This clean-room contract does not claim:

- Spine implementation equivalence;
- Spine source compatibility;
- product visual pass;
- semantic motion truth;
- unseen/generalized input performance;
- that the current dense BODY residual-hole problem is fixed.

It is a behavioural design contract for the next implementation and proof pass.
