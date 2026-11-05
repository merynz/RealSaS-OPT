# Corpus370 V8: original authored-pose source-face partition persistence

**Date:** 2026-11-05. **Status:** Real hosted original-source native teacher run [42008889188](https://github.com/merynz/RealSaS-OPT/actions/runs/42008889188) **SUCCESS 3/3** (plus SHA-gated workflow validator). Public aggregate `canonical/CORPUS370_NATIVE_POSE_PARTITION_REAL_RESULT_V8.json`. PR #327 remains DRAFT; training, drawable/graphics identity and product release remain CLOSED.

## Independent source proof

Unlike V7, which observed only bounded 192² rest-pose RGBA edits, V8 opens the existing exact authored-teacher NPZ from the immutable source-native temporal pilot and compares it against the **independently proven original source rest NPZ**. Source face indices must be equal in the two files, source rest vertices must match within 1e-6×source extent, and every pose must share that original source-vertex/face domain. Exact source, original-rest, authored-teacher, source-hit pair, and nested partition JSON SHA-256 bindings are fail-closed.

The V8 source-pose court sees **actual native source** geometry samples: 32 CAT frames from 8 authored actions, 24 TRICERATOPS frames from 6 actions, and 24 COWFISH frames from 6 actions. Source names / armature / object labels are teacher-only and not emitted as student targets. It replays the 8/16/32/64 geometry-only group addresses from the same preexisting nested proposals, without reassigning source faces per frame.

The direct observed properties are:
- frame-by-frame original-source/rest vertex and face-order equivalence;
- source-face-to-anonymous-group address invariance under the *known* sampled authored poses;
- bounded per-group centroid displacement and a separate best-fitting rigid-transform residual (an estimate of nonrigid geometry, **not** animation-controller identity);
- original rest-degenerate source faces and new posed-face collapse counts reported **separately**;
- group centroid pairwise separation change measured relative to the first authored sampled frame. This is observational and can include root/global and action changes.

This source-face persistence **partly follows from the same indexed native surface**. It is not proof that separately rendered 2D layers can be moved, independently reweighted, safely ordered, attached or concealed across these poses.

## Real source evidence

| Source | Actions | Real sampled native poses | Source faces | Existing degenerate REST faces | New posed-face collapse instances |
|---|---:|---:|---:|---:|---:|
| CAT | 8 | **32** | 684 | 0 | **0** |
| TRICERATOPS | 6 | **24** | 1,334 | **6** | **0** |
| COWFISH | 6 | **24** | 1,390 | 0 | **0** |

All 80 pose frames and 3,408 original source-face addresses were checked; 60 valid within-action temporal edges were observed (24/18/18). The TRICERATOPS rest mesh contained six original degenerate faces. These are not new pose collapse defects and are not admitted to any rig failure/owner-negative label.

Each worker’s original independent court JSON was downloaded separately after the hosted run and verified with SHA-256:
- CAT: `5a195d44940147a3e608fb448f25bcd6e85c74c1f722d08692dc9c49514dbda3`
- TRICERATOPS: `0c77f94094b5b18f9e46db352754112246971233fc9dc7bb78ceed6588fece10`
- COWFISH: `b262594df4458199d49660995a31a0be85bf2fac812040bb912c250705ce420e`

The raw poses, original mesh arrays, source teacher metadata and all per-group observations remain private at `gdrive:corpus370_preparation_v1/hosted_native_pose_partition_private_v8/42008889188-1`. GitHub stores only non-sensitive aggregate measures, program code and restrictive lineage.

## Numerical diagnosis (not a selection objective)

Maximum local best-fit rigid residual over source extent across groups and native poses:

| Original source | 8 groups | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| CAT | 0.08384 | 0.07955 | 0.06351 | 0.05892 |
| TRICERATOPS | 0.15060 | 0.13210 | 0.13210 | unavailable |
| COWFISH | 0.04059 | 0.04059 | 0.04059 | unavailable |

This can indicate whether coarser arbitrary groupings combine different native motions, but **more groups mechanically lower best-fit residual** in many cases, so use neither this metric nor the V7 finest-partition self-reference to auto-select a group count. First sample in a clip need not be the native rest pose. Authored motion is observed behavior, not controlled stimulus.

## Source-authority gates

- **Measured:** source-rest and teacher face order, native sampled geometry and group-address correspondence, group-level best-fit residual and area stability.
- **Not measured:** native animated original-source vs triangle-soup **RGBA parity**, alpha bleed, hidden z/draw order, CAA source-art authority, per-pose independent group removal/isolations, native-resolution renderer fidelity, attachments, mechanical counterfactual controls.
- **Forbidden:** map source owner/mesh/material/action names into IRIS/TESSA labels, infer SAME_DRAWABLE from co-motion, infer DIFFERENT_DRAWABLE from differential deformation, mint graphics owner or model-training admission.
- **Corpus roster:** the 352 observation PASS / 18 observation FAIL universe remains unchanged, and **zero** new training subjects are admitted by V8.

## Next actual science gate: V9 original authored animated image court

At selected **real source-authored action frames**, reimport original source exactly as the teacher did, validate evaluated native posed vertices against the SHA-sealed teacher NPZ, create a triangle-soup view with identical UV/material/alpha and **pose its per-face vertices to those sampled source positions**. Render original animation vs posed triangle soup, no-op and anonymous group isolate/removal across independent cameras. Fail/ABSTAIN when appearance parity, occlusion, seams, or draw order deviate, and keep all real RGBA private. Only after that may Graphics/CAA be assessed separately; mechanical causal intervention remains a different independent gate.

**Operational state:** V8 research closed as bounded source-native geometry correspondence, V9 animation-render court next. PR #327 remains draft; no merge or materialization notebook authorization.
