# IRIS → TESSA Scene-Memory Intervention V1

Date: 2026-10-16

This note records the narrow pre-Fit8 architecture change. It does not reactivate the deferred broad Unified-IRIS program.

## Problem

The legacy topology path discards rich IRIS scene information and reduces the scene to 17-D per GSA node before training a second TESSA surface encoder.

## V1 change

Use the existing IRIS `scene_planes` directly:

```text
IRIS scene_planes
  → explicit triplane position/plane-id tokens
  → bounded topology-memory adapter
  → TESSA decoder-only topology role
  → learned topology token sequence / proposal evidence
```

No GSA 17-D tensor is accepted by the new topology head.

## Compiler authority

The model-side proposal remains non-authoritative. The signed-field branch still goes through deterministic zero-surface, geometry substrate, GSA and partition courts. Stage18 remains the meeting point for learned proposal + qualified Compiler evidence; Stage19 remains canonical M authority.

The existing GSA/partition support path therefore survives as post-proposal binding/qualification evidence rather than neural scene conditioning.

## Fit8 control

The existing IRIS core source is intentionally unchanged by this first code step so the stopped Fit8 IRIS checkpoint can be evaluated against its exact resume binding. Initial topology-head training should freeze the sealed IRIS representation. Joint fine-tuning is a separate future experiment, not implicit in V1.
