# RealSaS IRIS Single-Pose Frontend — Native 1024 Corpus Contract V1

Status: CANONICAL_FRONTEND_DATA_DIRECTION_FROZEN__IMPLEMENTATION_PENDING
Date: 2026-08-22
Branch: g0-g1/single-pose-geometry

## Decision

The primary single-pose IRIS frontend corpus shall be rendered natively at **1024×1024 per view**.

128×128 remains historical/control-only. 512×512 may be deterministically derived from the 1024 master for matched ablations, but is not the primary observation authority.

Shipping/problem boundary remains:

```text
ONE neutral pose × 8 ordered views
    -> IRIS observable geometry frontend
    -> RiggingSurface
    -> Geppetto / Arachne / Compiler
```

## Why 1024 is primary

The intended product distribution is dominated by user-provided/generated artwork near 1024×1024. The active scientific problem is fine cross-view surface correspondence and geometry recovery, where early destructive downsampling can erase exactly the local evidence needed to resolve thin structures, silhouette-adjacent surfaces, symmetric parts, accessories and other hard-tail cases.

This decision does **not** assert that resolution alone solves the hard tail. It removes resolution as an avoidable information bottleneck and allows later experiments to localize the remaining failure to representation/matching rather than missing pixels.

## Geometry-only authority firewall

The 1024 frontend corpus must contain no rig/mechanics fields.

Allowed training/evaluation truth:

- native 1024 RGBA/raster observation;
- foreground/alpha derived from the same render;
- canonical/object-frame surface XYZ;
- canonical/object-frame surface normals;
- normalized image coordinates XY01;
- visibility/occlusion state;
- geometry-only surface identity/provenance used strictly as teacher/evaluator correspondence authority;
- camera/view metadata required to reproduce the projection.

Forbidden in the frontend corpus:

- joints or joint roles;
- skeleton hierarchy/parents;
- skinning weights or support masks derived from weights;
- deformation probes;
- GFDR/mechanics targets;
- owner identity or compiler IDs;
- Pose B.

Forbidden fields must be physically absent from emitted frontend sidecars/shards, not merely ignored by the learner.

## Dense truth preference

A 1024 raster should not be supervised only by the historical 512 sparse surface samples if denser geometry-only truth can be rendered.

Preferred authority is a geometry G-buffer / dense teacher representation containing, for foreground pixels where defined:

```text
canonical XYZ
canonical normal
surface provenance (e.g. triangle/primitive id + barycentric coordinate, teacher-only)
visibility/depth consistency
```

The provenance channel is teacher/evaluator-only and is not a required IRIS output. It exists to generate exact same-surface cross-view positives and hard negatives without importing rig semantics.

If the existing renderer cannot expose dense primitive/barycentric provenance safely, fallback is a denser persistent surface sampling set (target >=4096 samples) while retaining the historical 512-sample set as an exact lineage/parity anchor subset.

## Resolution hierarchy

Primary master:

```text
1024×1024 × 8 views
```

Matched derived controls:

```text
512×512  <- deterministic downsample from 1024 master
128×128  <- historical/control lineage only
```

XY truth remains normalized in [0,1]^2 and therefore resolution-independent.

## Frontend architecture implication

Full global self-attention over all high-resolution spatial tokens is forbidden as the default design.

The frontend shall be hierarchical:

```text
1024×1024 × 8
      |
shared high-resolution encoder
      |
+----------------------+----------------------+
|                                             |
local/detail path                             global/context path
high-res feature maps                        aggressively pooled tokens
(e.g. 256² / 128²)                           (e.g. 16² or smaller per view)
|                                             |
fine correspondence evidence                 multiview global reasoning
+----------------------+----------------------+
                       |
             coarse-to-fine matching
                       |
         reciprocal/cycle consistency
                       |
          common-frame P/N geometry
```

D1/D2 historical code is architecture lineage only. Two-pose results are not single-pose evidence.

## Experimental ladder

```text
F0-128   historical matched control
F0-512   deterministic-resolution control
F0-1024  primary native-resolution treatment
F1       + reciprocal/cycle consistency
F2       + fine/local reranking within retained top-k
F3       learned matcher only if a reproducible hard tail remains
```

No early global top1 singleton collapse is allowed. The coarse stage must preserve high recall; fine stages may refine only inside retained candidate support unless a later preregistered experiment changes this rule.

## Primary measurements

- same-surface top1 / top4 / top8 / top-k containment;
- reciprocal agreement;
- cycle failure rate;
- localization error in normalized coordinates and native pixels;
- family median / p90 / p95;
- symmetry/thin-structure/accessory hard-tail slices;
- downstream common-frame P consistency;
- normal accuracy where applicable;
- abstention/ambiguity retention when evidence does not support a singleton.

Resolution promotion must be based on matched family-disjoint evidence rather than mean-only improvement.

## Preservation

The frozen G1 baseline remains preserved/unstarted and is not silently rewritten by this contract. The 1024 frontend is a separate research continuation until its own preflight/preregistration authorizes training.

Optimizer steps under this new 1024 contract: **0**.
