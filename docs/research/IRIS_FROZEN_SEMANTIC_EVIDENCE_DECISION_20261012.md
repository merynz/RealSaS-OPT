# IRIS frozen semantic evidence — decision and Knight experiment

**Date:** 12 October 2026  
**Base:** `main@82e31b566cdb811e5b1c41fd4f84a6eaeeee7ce3`  
**Research branch:** `research/iris-semantic-behavior-v1-20261012`

## Decision

Do **not** modify or retrain the current IRIS geometry model for the semantic experiment.

The canonical geometry line remains exactly the existing `SceneFirstSignedGeometryV4`/sealed geometry checkpoint and the canonical single-M product path.  Instead of first building a large semantic 3D training corpus, use frozen external foundation models as **non-authoritative semantic evidence producers**, then lift/fuse that evidence onto the existing canonical M and let the Compiler qualify downstream relations.

This supersedes only the immediate plan to train a new semantic decoder from hand-authored/GSA-derived targets.  It does not revoke the longer-term option to distill qualified pseudo-labels into an IRIS semantic head once enough data exists.  RGB-head work remains parked and is not part of this experiment.

## Why

StdGEN-style semantic supervision is attractive because a shared 3D field can carry body/clothing/hair identity, but its clean supervision comes from semantically decomposable 3D source assets.  RealSaS does not currently possess a sufficiently broad, independent semantic 3D truth corpus.  Treating GSA component/persistence IDs as semantic teacher truth would be circular and incorrect: those IDs are addresses/geometry structure, not independent artist semantics.

RealSaS already has a useful advantage: eight source views, canonical cameras and canonical M addressing.  A frozen 2D/open-vocabulary model can provide modal semantic evidence per source view; Compiler geometry can fuse that evidence across views onto the same M without introducing a second visual mesh or a second authority.

## Architecture

```text
SOURCE V0..V7
      |
      +--> frozen open-vocabulary detector ----+
      |                                        |
      +--> frozen prompted segmenter ----------+
                                               |
                             modal mask proposals + confidence
                                               |
                                               v
                         FrozenSemanticMaskEvidenceV1
                                               |
                             canonical cameras + canonical M
                                               |
                                               v
                        lift/fuse to M-face evidence
                                               |
                 +-----------------------------+--------------------+
                 |                             |                    |
                 v                             v                    v
          subject-local owner           grip/contact          overlap/order
          components on M               candidates            evidence
                 |                             |                    |
                 +-----------------------------+--------------------+
                                               |
                                               v
                                   Compiler qualification
```

External models do not assign canonical owner IDs.  Connected same-label regions on M create subject-local **owner candidates**.  Unknown/low-confidence areas abstain.

## Frozen research providers

The first inexpensive oracle uses:

- `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71`
  - `model.safetensors` SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`
  - open-vocabulary class/box proposal evidence.
- `facebook/sam2.1-hiera-small@ee5bba1d82bb8749febdf90f45e84b687142ba03`
  - `model.safetensors` SHA-256 `0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60`
  - box-prompted modal masks.

These exact providers are replaceable research adapters.  Their outputs are never silently promoted to truth.

## What is implemented in this arm

- `models/iris/semantic_evidence_v1.py`
  - provider-neutral, hashable semantic-mask evidence bundle;
  - no geometry-model import;
  - no trainable parameters.
- `compiler/realsas_compiler_core/semantic_relation_evidence_v1.py`
  - visible source mask evidence -> canonical M-face semantic distribution;
  - explicit abstention;
  - same-label connected M components -> subject-local owner candidates;
  - class-to-behavior prior evidence (`RIGID`, `PRIMARY_ARTICULATED`, `SECONDARY_DEFORMABLE`, `UNKNOWN`);
  - hand/held-object proximity -> `GRIP_CANDIDATE` evidence;
  - source-observed semantics + M fragment layers -> `FRONT_OF_SOURCE_EVIDENCE`;
  - recorded posed M -> `POSE_ORDER_CONFLICT` diagnostic when supported rest/source order reverses.
- `tools/research/run_frozen_semantic_evidence_v1.py`
  - external provider execution only.
- `tools/research/compile_knight_semantic_relations_v1.py`
  - reuses sealed Knight M/rest/poses and emits diagnostics;
  - does not execute geometry inference, refit, reskinning or retargeting.

## Geometry invariants

This arm MUST NOT:

1. edit `models/iris/v4/scene_first_signed_v4.py`;
2. alter any sealed IRIS geometry checkpoint;
3. add semantic loss to the current geometry fit;
4. regenerate or replace M merely to make semantic evidence easier;
5. add an independent visual geometry carrier;
6. make external provider masks canonical authority.

The Knight compiler diagnostic stores explicit `rest_geometry_byte_equal_after` and `recorded_pose_byte_equal_after` checks.

## First Knight questions

This experiment is intentionally diagnostic before runtime modification.

1. Do cape/cloak vs arm/body masks map to stable, connected M owner candidates across views?
2. Do hand and sword/weapon/shield owner candidates appear with enough confidence to propose the intended grip relation and useful M anchors?
3. Does source-supported semantic overlap produce a cape/front relation in the relevant views?
4. Does the existing recorded RUN motion then report a `POSE_ORDER_CONFLICT` at the known elbow/cape failure interval?
5. Are rigid vs secondary-deformable regions plausible enough to justify a later secondary-motion consumer experiment?

If these fail, do not change geometry to rescue the experiment.  First adjust prompt vocabulary/provider capacity/cross-view semantic association and measure again.

## What this arm does NOT yet claim

- a qualified grip contract;
- a qualified draw-order graph;
- runtime semantic order override;
- collision/cloth simulation;
- authorized secondary motion;
- trained IRIS semantic head;
- RGB prediction;
- product qualification.

The point of this arm is to determine whether frozen semantic evidence plus existing M is sufficient to support those later consumers safely.

## Promotion path if Knight evidence is strong

Only after the diagnostic succeeds:

1. define typed `QualifiedGripRelationIR`/equivalent using semantic + geometric + motion evidence;
2. define typed presentation overlap/order qualification with view/pose scope and cycle/conflict handling;
3. wire those qualified relations into the existing single-M runtime consumer;
4. add independent Stage45 relation residual/occlusion proofs;
5. separately evaluate secondary motion on semantic regions while preserving canonical M and primary motion authority;
6. accumulate high-confidence multi-view/compiler-qualified pseudo-labels; only then consider distillation into an IRIS semantic head.

The intended long-term pattern is:

```text
same canonical pipeline
+ richer frozen/learned evidence
+ explicit qualification
+ additional typed consumers
```

not a parallel image system and not a second geometry authority.
