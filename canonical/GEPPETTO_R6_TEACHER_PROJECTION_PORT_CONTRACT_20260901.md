# RealSaS — Geppetto R6 Teacher Projection Selective Port Contract — 2026-09-01

**Status:** `FROZEN_SELECTIVE_PORT__TRAINING_EVALUATION_ONLY__NO_MODEL_ARCHITECTURE_AUTHORITY`

## Purpose

Selectively rescue the still-valid skeleton teacher/evaluator projection from historical branch `geppetto-arachne-v0-1-20260830` without importing the branch's unsealed G0.1 neural architecture or its authority-sensitive surface+interior token contract.

Historical source authorities:

- `experiments/geppetto_arachne_v0_1_20260830/teacher_projection_v1.py` blob SHA `3f82dd5559ab3620ed9d47005316642037d9d856`;
- teacher projection dataclass definitions originated inside `contracts_v0_1.py` blob SHA `9a25832eb669f055d4abdcb0f69cd78a730e89b1`.

## Ported semantics

The current R6 training/evaluation adapter must preserve:

- source `bone_heads`, `bone_tails`, `parents`, `deform_mask` are teacher/evaluator authority only;
- exactly one anonymous teacher control per deform bone;
- teacher control position is source bone head;
- helper/non-deform bones do not become controls;
- parent relation skips helper chains to nearest deform ancestor;
- multiple deform roots remain multiple teacher roots;
- no synthetic super-root;
- source bone index and source tail remain teacher provenance only;
- deterministic BFS serialization remains an audit/training representation;
- optional sibling randomization is training-only and may not change depth/ancestry/control set;
- source skin-column map remains explicit teacher/evaluator provenance;
- no canonical `J:*` product IDs are created.

## Explicitly not ported

This port must not bring in or authorize:

- `GeppettoG01` neural architecture;
- old `max_controls` as product authority;
- the 27D combined surface/interior `ConsumerTokenV1` contract;
- interior sampling or volume-state hidden completion semantics;
- Arachne predictor architecture;
- optimizer steps;
- product skeleton root/tree authority.

Compiler remains the sole current canonical root/parent/ID authority after Geppetto proposal emission.

## Target namespace

Current code lives under:

`experiments/geppetto_arachne_r6_20260901/`

It is training/evaluation apparatus, not runtime product code.

## Required regressions

At minimum test:

1. helper-chain nearest deform ancestor;
2. multiple-root preservation;
3. arbitrary source-parent cycle rejection;
4. duplicate-free/full-control deterministic BFS;
5. sibling randomization preserves exact control set and parent depth;
6. source skin-column map remains source-bone-index provenance;
7. no product canonical IDs are minted.

Passing this port does not authorize Geppetto training; it only restores a clean teacher/evaluator adapter needed to construct R6 truth without source-rig identity leakage into product inference.
