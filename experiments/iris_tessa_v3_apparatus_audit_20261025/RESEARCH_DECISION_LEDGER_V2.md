# IRIS/TESSA Research Decision Ledger V2

**Date:** 2026-10-26  
**Status:** `ACTIVE_RESEARCH_MEMORY__FAST_PROBE_MODE`  
**Scope:** architecture-search evidence only; no Product, ModelRelease, Truth139, FIT8 or genericity authority.

## Purpose

This file is the canonical working memory for zero/low-cost architecture search. Chat history, GitHub Actions green checks, branch names and individual probe logs are not the decision authority.

For every court retain:

`question -> exact treatment/control -> metric -> result -> interpretation -> carry-forward choice -> unresolved dependency`

A GitHub workflow PASS means the probe executed successfully. It does not by itself mean the scientific hypothesis passed. Scientific status comes from the preregistered metric/report and is recorded here.

## Research operations policy

During synthetic / analytic / bounded micro-learning architecture search, use the fast GitHub workflow probe path. Preserve exact preregistration, seeds, frozen dependency identities where relevant, emitted report/artifact, and the resulting decision in this ledger.

Use Platform `Experiment -> Attempt -> Artifact/CAS` authority when the work crosses into real dataset/checkpoint/training/expensive-compute/product-model-authority experiments. PR #218 is intentionally deferred for that phase.

---

# Active carry-forward architecture evidence

## O1 — authored alpha/support must survive observation ingestion

**Status:** `PROVEN_REPAIR_REQUIRED`

Current RGBA -> white-composited RGB can make distinct authored support/alpha states observationally identical before shared-scene construction. Discarding alpha is therefore ineligible.

**Carry forward:** preserve alpha/support through an authoritative aligned path. Exact feature contract remains open.

**Source:** `OPEN_QUESTIONS.md` A1; `rgba_identifiability_probe_v1.py`.

## O2 — keep frozen DINOv2-S/14; expose early + final features

**Status:** `TREATMENT_SELECTED_TINY`

Exact feature-access court showed L2 local recovery = 1.0 across tested 2/4/8/12 px strata while the final-only contract was materially weaker. The matched dual-level court held token count and trainable adapter budget fixed and obtained:

- `CURRENT_FINAL_512`: mean local `0.864626`, min `0.816327`;
- `DUAL_L2_FINAL_MATCHED_512`: local mean/min `1.0/1.0`;
- dual final-feature cosine mean/min `0.997598/0.996222`.

**Carry forward:** frozen DINOv2-S/14 remains the baseline. Use same 37x37 lattice with aligned `L2 + FINAL` information. Do not adopt DINOv3 merely for the observed local-detail problem.

**Still open:** exact production fusion form, including a final residual + projected early-detail form and integration of alpha/support.

**Sources:** `DINOV2_FEATURE_ACCESS_RESULT_V1.md`, `DINOV2_DUAL_LEVEL_VIEW_TOKEN_RESULT_V1.md`.

## O3 — current scene-query -> 2x carrier route is not the first local-detail suspect

**Status:** `MECHANISM_ESTABLISHED`

The bounded subcell-routing court showed distinct subcell states can be preserved perfectly when they are already present in source tokens.

**Carry forward:** do not redesign the 32->64 routing merely to fix the DINO final-only information loss. Reopen only if a harder carrier/query court falsifies it.

## O4 — retain query-mediated cross-view fusion as baseline

**Status:** `MECHANISM_ESTABLISHED`

Current query-mediated fusion and source-self-attention-first both reached perfect correspondence across five seeds; early source self-attention supplied no measurable gain.

**Carry forward:** retain query-mediated fusion; do not pay O(source-token^2) for source self-attention without a harder falsifying court.

**Still open:** deterministic camera/ray/backprojection geometry versus learned factorized camera+XY features.

**Source:** `IRIS_CROSS_VIEW_FUSION_RESULT_V1.md`.

## O5 — endpoint-dependent presentation evidence must be explicitly bound before TESSA compression

**Status:** `TREATMENT_SELECTED_TINY`

Endpoint court:

- current late/unbound mean `0.503223`;
- generic joint/unbound mean `0.507373`;
- endpoint-bound late `1.0` on all five seeds;
- random-label firewall `0.506396`;
- binder has zero trainable parameters.

**Carry forward for endpoint relations:** preserve typed endpoint refs; convert declared coordinate domains explicitly; query `Q(Z,x_a)` and `Q(Z,x_b)`; bind admissible endpoint-local evidence before presentation compression; let TESSA structuralize already-addressable evidence.

Generic joint fusion is not the selected endpoint solution.

**Source:** `TESSA_ENDPOINT_BINDING_RESULT_V1.md`.

## O6 — coordinate domains require deterministic coherence

**Status:** `PROVEN_REPAIR_REQUIRED`

IRIS query/carrier geometry uses approximately `[-1,1]`; TESSA surface/presentation/output domains use `[-0.5,0.5]`. Existing Bridge position conventions also disagree with `align_corners=False` carrier cell centers.

**Carry forward:** typed coordinate domains; `x_iris = 2*x_tessa` for endpoint queries; scene positions aligned to true carrier cell-center geometry. Do not ask a learned module to absorb unit conversion / half-cell offsets.

**Source:** `OPEN_QUESTIONS.md` H1.

## O7 — current positive-only occlusion pair truth is ineligible

**Status:** `PROVEN_REPAIR_REQUIRED`

Validity-masked BCE over only positive same-ray support cannot learn calibrated non-occluding negatives.

**Carry forward:** Truth Foundry must expose qualified positive / negative / N-A / abstention support before pair occlusion training.

## O8 — first-hit binary ray visibility is not final graphics truth

**Status:** `PROVEN_REPAIR_REQUIRED`

Opaque first-hit truth does not represent alpha compositing.

**Carry forward:** preserve complete-stack causal quantities such as incoming transmittance and per-hit contribution, e.g. `w_k = alpha_k * product_{j<k}(1-alpha_j)`. Whether final runtime visibility is learned, deterministic, or hybrid remains open.

## O9 — mechanical tangent supervision needs an anisotropy validity rule

**Status:** `PROVEN_REPAIR_REQUIRED`

Principal tangent direction is undefined as major/minor stretch become isotropic.

**Carry forward:** separate tangent validity / anisotropy gate or continuous equivalent. Do not supervise arbitrary teacher orientation in isotropic regions.

## O10 — current surface-witness index sampler is invalid as the default

**Status:** `PROVEN_REPAIR_REQUIRED`

Vertex-array index sampling is not geometry-order invariant and offers no qualified coverage guarantee.

**Carry forward:** compare against deterministic geometry-native sampling using permutation invariance, geometric coverage radius, and qualified thin/low-area support coverage. Do not blindly allocate one witness per noisy extracted component.

## O11 — extraction resolution is a topology-observability parameter

**Status:** `MECHANISM_ESTABLISHED__NO_RESOLUTION_SELECTED`

Base-grid sign observability is limited before FlexiCubes deformation can help. Measured TESSA-domain phase-independent guaranteed scales:

| Resolution | sphere-like diameter | thin support diameter | interface gap |
|---:|---:|---:|---:|
| 24 | 0.072169 | 0.058926 | 0.041667 |
| 32 | 0.054127 | 0.044194 | 0.031250 |
| 48 | 0.036084 | 0.029463 | 0.020833 |
| 64 | 0.027063 | 0.022097 | 0.015625 |

**Carry forward:** res24 is not automatically retained merely because current product uses it. No replacement resolution is selected yet.

**Next dependency:** normalized real morphology-scale census using silhouette/local-thickness/gap statistics, then fixed-resolution versus adaptive/coarse-to-fine comparison if warranted.

**Source:** `EXTRACTION_TOPOLOGY_OBSERVABILITY_RESULT_V1.md`.

## O12 — relation algebra is not structurally enforced by the current generic pair head

**Status:** `STRUCTURAL_FACT_ESTABLISHED__FINAL_PARAMETERIZATION_OPEN`

Current pair representation `[a,b,a-b,a*b]` can represent directed quantities but does not enforce known invariants such as symmetry or front-order complementarity by construction.

**Carry forward:** every causal relation truth family must be assigned an algebra class before final head selection: `SYMMETRIC`, `DIRECTED`, `COMPLEMENTARY_OR_TIE_AWARE_ORDER`, or `RAY_GLOBAL`. Compare family-specific constrained parameterizations against unconstrained heads under matched capacity after truth quantities stabilize.

**Source:** `RELATION_ALGEBRA_RESULT_V1.md`.

## O13 — Oracle-Z V1 does not select or reject triplanes

**Status:** `APPARATUS_INCONCLUSIVE__ESCALATE`

V1 failed to isolate representation capacity from optimization/shared-query confounds. Crucially, the supposedly favorable dense-3D control also failed the volumetric control, so a negative triplane verdict would be invalid.

Observed examples:

- `TRIPLANE_Q` surface nRMSE ~`0.318-0.319`;
- `DENSE3D_Q` ordinary surface ~`0.231-0.239`;
- both near ~`1.0` on the deliberately volumetric control;
- multires triplane thin-family improvement only ~`4.77%`, below the preregistered 20% gate;
- explicit xyz did not improve the base arm.

**Carry forward:** no carrier choice changes from V1.

**Required V2 factorization before any representation verdict:** carrier-only oracle/per-scene or analytic decoder upper bound; shared-Q court after carriers are proven/frozen/prefit; optimizer calibration on an independent family; frequency/thickness ladder against carrier cell size; negative-control calibration where at least one arm demonstrably solves the easy volumetric case.

**Sources:** `ORACLE_Z_REPRESENTATION_RESULT_V1.md`, `ORACLE_Z_REPRESENTATION_PREREG_V2.md`.

## O14 — TESSA local-window semantics are measured, adequacy remains open

**Status:** `SEMANTICS_MEASURED__ADEQUACY_OPEN`

Current chunk-end-anchored semantics provide about `3841-4096` direct historical tokens across a full post-window chunk and inference intentionally reproduces the same behavior.

**Carry forward:** this is not a learned failure. Measure real dependency span before increasing attention memory/window.

---

# High-priority unresolved courts

1. **Oracle-Z V2 representation factorization** — highest priority; fix apparatus before carrier verdict.
2. **Alpha feature contract** — aligned alpha/support residual without unnecessary token-count growth.
3. **Camera/backprojection geometry** — factorized learned camera+XY vs deterministic orthographic ray/basepoint features.
4. **Surface witness sampler** — index baseline vs geometry-native deterministic coverage treatment.
5. **Morphology-scale census / extraction policy** — choose resolution/adaptive policy from real normalized support scales.
6. **Shared-scene degree (GM/GP sharing)** — independent / late-fusion / shared-trunk-private / full shared Z / family adapters, only with clean matched courts.
7. **Carrier x query interaction** — only after carrier-only and shared-Q adequacy are separated.
8. **Relation algebra x truth parameterization** — after causal truth quantities stabilize.
9. **Presentation schema V2** — richer vector/response/ray quantities may supersede scalar probabilities.
10. **Real topology dependency span** — CPU census before TESSA memory redesign.

Deferred until authored causal truth exists: multi-objective gradient interference, final relation-head objective selection, final presentation truth schema, real joint IRIS/TESSA architecture selection.

---

# Assembly rule

No single tiny-probe winner becomes final architecture by itself.

A component may enter the V3.1/VNext candidate assembly only when its own mechanism court is valid and passed, known deterministic contract defects are repaired, and relevant interaction courts do not overturn it. Authored causal-truth courts then confirm the combined behavior.

Operational sequence:

`fast probes -> Decision Ledger -> interaction courts -> V3.1 architecture freeze -> authored Truth Builder -> real training via Platform -> FIT8 -> LOFO/unseen`.
