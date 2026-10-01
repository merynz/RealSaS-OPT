# Timeout-Era Forensic Index — 2026-09-30 Lineage

> NON-AUTHORITATIVE agent reconstruction. This file records what was mechanically observed, not what should be promoted.

## Executive finding

The current `main` and the 30 September normalization candidate are **not the same implementation**.

Current main:
- substantive base `e6c91184...`
- current head `a8002b20...`
- only one commit ahead of `e6c91184...`, and that commit changes generated authority/cache files only.

Primary normalization candidate:
- `integration/generic-main-finalization-20260930`
- head `92de05292...`
- diverged from current main at `e6c91184...`
- 68 commits ahead / 2 behind current main at inspection
- 168 changed files

Therefore the large normalization effort was not substantively promoted into current main.

## Candidate family

The following are mostly one family rather than separate architectures:

```text
generic-main-promotion
       |
       +--> generic-main-finalization
                 |
                 +-- 2 --> final-main-closure
                           |
                           +-- 1 --> ops/main-knight-render
                                      |
                                      +-- 2 --> backup/main-normalized
                 |
                 +-- 35 --> ops/current-main-knight-render
```

`integration/generic-main-normalization-v2-20260930` descends from the promotion lineage but later diverges from finalization. It must be treated as a sibling evidence branch, not “the final version”.

`audit/knight-rest-visual-owner-20260930` is a 1,958-commit research lineage relative to current main and is not a merge candidate as a whole.

## What the full PASS actually proves

Run `36724501505` on tested head `484c01f...` succeeded:
- repository/governance: 59 passed
- learned-source ownership: 51 passed
- compiler regressions: 434 passed, 5 skipped
- synthetic proof export: 1 passed
- native runtime CTest: 9/9 passed
- plan validated with SHA `be0067d7...`
- implementation closure `1cb60bb3...`

The workflow explicitly describes itself as rerunning after a **runtime fail-closed seam**.

It did **not** run `tests/repository/test_source_owned_visual_runtime_wiring_v1.py` as part of its repository-governance set.

So its correct interpretation is:

> the normalized candidate is mechanically coherent under its declared fail-closed boundary; the source-owned visual runtime seam is not implemented.

It is **not** evidence that the visual presentation path reaches Stage42/native playback.

## Source-owned visual ownership split

Current main and candidate differ in architecture scope. Only Stage18's plan record changes, but the candidate has substantial implementation changes downstream.

Candidate Stage18:
- takes observation authority directly;
- builds mechanical mesh plus source-owned visual mesh;
- establishes a distinct visual presentation geometry ownership domain.

Candidate Stage37/38 lineage:
- carries `visual_mesh_set_binding_hash`;
- asserts `mechanical_mesh_render_authority=False` in source-owned visual mode;
- verifies exact visual-mesh binding identity.

Candidate Stage42:
- does not yet execute the visual mesh;
- sees source-owned mode and raises `RUNTIME_V2_SOURCE_OWNED_VISUAL_PRESENTATION_BINDING_REQUIRED`;
- legacy runtime package remains mechanical `faces + face_uv + mesh.bin` oriented.

This is an explicit fail-closed P0 boundary, not silent fallback.

## Stale audit reference

The normalization handoff claims an exact audit file:

`canonical/V2_PRODUCT_STATE_WIRING_AUDIT_V1_20260928.json`

That path is absent from both:
- current `main`;
- `generic-main-finalization` head.

Do not use that missing file as authority. The seam is independently supported by:
- `tests/repository/test_source_owned_visual_runtime_wiring_v1.py`;
- `.github/workflows/source_owned_visual_runtime_wiring_gate_v1.yml`;
- candidate product-state adapter source;
- candidate runtime adapter source.

This stale reference is itself evidence that documentation and exact artifact lineage drifted during the timeout-era work.

## Current classification

- Current main: **A authority / incomplete relative to candidate research**
- Generic finalization candidate: **C/B mixed evidence** — extensively tested, non-authoritative, known P0 fail-closed seam
- Visual ownership split existence on candidate: **A observed**
- Stage42 visual consumer implementation: **F absent / D unresolved engineering target**
- Wholesale promotion decision: **D forbidden until commit-level lineage and gate scopes are reconstructed**
- Huge audit branch: **D evidence mine only**

## Next forensic target

1. reconstruct the candidate's substantive commit slices from `e6c91184...` to tested head `484c01f...`;
2. classify each slice by subsystem + test evidence + supersession;
3. identify which slices are independently promotable and which depend on the unresolved Stage42 visual presentation executor;
4. inspect gate failures that coexisted with the full PASS rather than assuming every workflow was part of the closure;
5. only after this build a proposed canonical promotion set.
