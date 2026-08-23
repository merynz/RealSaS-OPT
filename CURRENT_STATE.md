# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Status:** `IRIS_ARCHITECTURE_AUDIT_IN_PROGRESS__ALL_TRAINING_FORBIDDEN`

## Read this first

This file is the single continuation authority during the audit. The preserved pre-audit branch is `g0-g1/single-pose-geometry`.

## Training authorization

**NO optimizer run is authorized.** This includes mini, scale, CAL/DEV/EXTERNAL, native-1024 qualification and production training.

Training may resume only after:

1. one canonical single-pose IRIS architecture is frozen;
2. corpus -> tensor -> output coordinate/resolution semantics are explicit and tested;
3. the 2026-08-21 multi-paper descriptor/correspondence transfer report is reconciled with the subsequent D1/D2/G2 experiments;
4. executable code matches that architecture role-by-role;
5. evaluator measures the intended quantity in explicit native-pixel/object-space units;
6. syntax, shape, coordinate, role-separation and synthetic preflight all PASS;
7. `audit/IRIS_ARCHITECTURE_AUDIT_20260824.md` closes with no unresolved mini-gate mismatch.

## Canonical problem boundary

Shipping input remains `ONE neutral pose x 8 ordered views`.

IRIS owns observable geometric evidence sufficient for a deterministic `SurfaceBuilder`; it does not own hidden mechanical owner IDs, source-rig exact partition, skeleton hierarchy, skinning or mandatory GFDR.

The current evidence family remains revisionable under the canonical substrate contract:

- P: common/object-frame surface position evidence;
- N: local orientation/normal evidence;
- U_geo: geometric predictive risk;
- coarse persistence capability;
- fine local correspondence evidence;
- direct visibility/alpha and derived support;
- provenance;
- set-valued ambiguity when singleton evidence is insufficient.

## Why the previous M256 result cannot authorize the next scale step

The completed audited M256 run is retained as diagnostic evidence only.

It did show a real learner signal:

- P error reduction: ~85.4%;
- N error reduction: ~83.5%;
- coarse persistence top8 gain: ~+0.691;
- fine persistence top8 gain: ~+0.674.

But its consumer/evaluator did not match the intended final architecture:

- native corpus authority was 1024, while model input was 256 and matcher candidate lattice was fixed 128x128;
- matcher tolerances were normalized-grid constants rather than explicit native-pixel criteria;
- `Z_fine` was used in global rank fusion even though the preserved D2 result falsified it as a global ranking authority and supported it only as retained-top-k local evidence;
- singleton/ambiguity heuristics were not a completed calibrated ambiguity contract;
- checkpoint selection did not constitute the final frozen observable metric panel.

Therefore the run is **not** an information-limit result and does not authorize architecture scale-up.

## Research lineage that must be preserved

- `canonical/PRODUCT_CONTRACT_V1.md`
- `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`
- `experiments/g0_g1_single_pose_geometry/FRONTEND_NATIVE_1024_CORPUS_CONTRACT_V1.md`
- `experiments/g0_g1_single_pose_geometry/G0_CONTRACT_FREEZE.json`
- `experiments/g0_g1_single_pose_geometry/legacy_reserve/D2_RESULT.md`
- historical `experiments/iris_controlled_v1/` package and M256 outputs as diagnostic lineage only.

The 2026-08-21 paper transfer report and 2026-08-23 post-corpus closure matrix are external Drive research authorities and are being mirrored into the audit conclusions.

## Active implementation path

New active code is being built under:

`experiments/iris_single_pose_v2/`

The old `experiments/iris_controlled_v1/` folder is historical/diagnostic during this audit and must not be executed.

Target V2 architecture:

```text
native/control RGBA x 8
      |
shared high-resolution encoder
      |
+---------------- local/detail path ----------------+
| f2/f4/f8/f16 at resolution-proportional scales    |
+----------------------------------------------------+
      |
within-view axial reasoning @ full f16
      |
fixed-size pooled global multiview context
(known yaw, resolution-independent positional encoding)
      |
upsample/fuse context back into local f16
      |
coarse Zc @ R/8 -------- global high-recall search
      |                         |
P/N/U_geo @ R/2                 | retain top-k basins
Zf @ R/2 -----------------------+--> LOCAL refinement only
                                      |
                              reciprocal/cycle evidence
                                      |
                              set-valued hypotheses
                                      |
                            deterministic SurfaceBuilder
```

No fixed `max_w`; no fixed 128 matcher lattice; no global `Z_fine` authority; no premature singleton.

## Current preflight status

A local synthetic V2 implementation has already passed:

- Python compile;
- exact align_corners=False coordinate round-trip;
- model forward shape checks at input 256 / 512 / 1024;
- loss forward/backward with finite gradients;
- explicit matcher role-separation test proving a globally perfect far-away `Z_fine` match cannot enter unless its coarse basin was admitted;
- native-pixel metric unit test.

This is code preflight only, not training evidence.

## NEXT EXECUTABLE STEP

**Do not train.** Finish the V2 audit package and repo consolidation:

1. freeze architecture/resolution/corpus/evaluator contracts;
2. commit the preflighted V2 source;
3. mark V1 execution paths historical;
4. run repository-wide syntax/static self-checks;
5. close `audit/IRIS_ARCHITECTURE_AUDIT_20260824.md` only if every mismatch is resolved.

Only after that closure may a new, preregistered mini-training notebook be created.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
