# RealSaS N1D — Representation Contract V2

**Date:** 2026-08-19  
**Status:** `FROZEN_FOR_OPEN_DEV_ADAPTIVE_H_TEST__NOT_QUALIFICATION`  
**Parent decision:** `REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT`  
**Frozen Stage-B authority remains:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Purpose

This contract turns the sufficiency-battery result into an explicit interface. The central change from the historical route is that plausible geometric states remain set-valued until a later constrained/global decision. Fixed early `top4` collapse is forbidden.

## Canonical representation

```text
P_A
H_i / P_B_geom candidates
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred
U_obs
```

### `P_A`
Observation-derived Pose-A carrier locus in the canonical common world frame.

### `H_i`
A bounded set of feasible Pose-B geometric hypotheses for carrier `i`.

Each hypothesis must retain enough provenance for downstream scoring/audit:

```text
endpoint_xyz
source_view_pair / supporting views
per-view descriptor scores or ranks
reprojection residuals
triangulation conditioning where available
candidate/search tier
```

`H_i` is the sole physical XYZ authority for Pose-B motion. A learned motion head may score members of `H_i`; it may not create an unconstrained replacement XYZ endpoint.

### `N_A`, `N_B`, `V_A`, `V_B`, `Z`
Normals, visibility/support and local descriptor/spatial evidence remain observation evidence. They do not independently override feasible geometry.

### `p_active`
Separate absolute motion/non-motion authority. It must carry calibrated confidence/coverage semantics rather than being inferred implicitly from a normalized amplitude rank.

### `log_amp`
Conditional magnitude/ranking evidence, meaningful after/with motion-state evidence. It is not free radial/XYZ authority.

### `dir`
Direction evidence, separated from activity and magnitude.

### `U_pred` and typed `U_obs`
Uncertainty is operational, not decorative. At minimum the contract supports:

```text
reprojection_error
view_support
triangulation_condition
feasible_set_width
match_margin
candidate_multimodality
cycle_error
```

These fields control hypothesis retention, bounded search expansion, abstention and later compiler confidence/coverage.

## Authority split

```text
bounded H_i                         -> physical XYZ authority
p_active                            -> absolute motion/non-motion authority
log_amp                             -> conditional magnitude/ranking evidence
dir                                 -> direction evidence
U / margin / multimodality/support -> retention / expansion / abstention
compiler/global mechanical solver   -> final collapse after global consistency
```

## Candidate-generation contract

1. Search Pose-B observation evidence from the frozen/common descriptor representation.
2. Preserve a bounded per-view candidate set; do not assume a universal fixed `K=4`.
3. Construct multiview rank-sufficient 3D hypotheses while retaining provenance.
4. Use observation-only uncertainty to decide whether to keep a compact set, widen retention, widen coarse search, or abstain.
5. Collapse to one endpoint only after constrained/global mechanical scoring.

`K=16` is a demonstrated open-development reference ceiling, **not** a permanent product constant.

## Required invariants

- No truth enters inference-time retention/expansion decisions.
- Candidate-set permutation must not alter the represented feasible set.
- Pose/view bookkeeping must remain deterministic and auditable.
- `p_active/log_amp/dir` cannot bypass `H_i` to emit free XYZ.
- Low-confidence/ill-conditioned cases may widen or abstain; they must not silently become confident singletons.
- Aggregate metrics cannot authorize a representation if a preregistered hard-tail/worst-family coverage gate fails.

## Forbidden by V2

- fixed early `top4` candidate collapse;
- unconditional singleton `P_B` representation before global reasoning;
- free XYZ motion head overriding multiview feasible geometry;
- one fused head treated as simultaneous activity/amplitude/direction authority;
- truth-conditioned candidate retention;
- large/end-to-end training before bounded adaptive `H_i` behavior and small factor heads are separately frozen.

## Current development sequence

1. Freeze/test `Adaptive Hypothesis Retention V1` under this contract.
2. If the bounded set contract passes hard-tail coverage and compute gates, freeze it.
3. Only then train/evaluate small strict family/episode-disjoint `p_active + log_amp` heads; keep `dir` separate.
4. After all open-development components are frozen, preregister a new untouched qualification panel.

This file defines the representation/interface contract; it does not change the historical Stage-B qualification result.
