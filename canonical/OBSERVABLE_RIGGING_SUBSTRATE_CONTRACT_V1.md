# RealSaS — Observable Rigging Substrate Contract V1

**Date:** 2026-08-22  
**Status:** `CANONICAL_PROBLEM_DEFINITION_FROZEN`  
**Shipping input:** `ONE neutral pose × 8 ordered views`

## 1. Canonical problem

RealSaS does **not** ask IRIS to recover a hidden authored rig or to infer a privileged mechanical ontology from raster input.

The canonical IRIS research problem is:

> **From one neutral 8-view raster observation, recover the minimal observation-grounded geometric substrate that is sufficient for downstream skeleton generation, skinning and compiler verification.**

Equivalently:

```text
ONE neutral pose × 8 ordered views
        ↓
      IRIS
        ↓
rigging-sufficient observable geometry
        ↓
 deterministic SurfaceBuilder
        ↓
 canonical RiggingSurface
        ↓
     Geppetto
        ↓
 skeleton / hierarchy proposal
        ↓
      Arachne
        ↓
 skinning / weight proposal
        ↓
      Compiler
        ↓
 verified editable puppet
```

The research target is the **functional 8-view observational equivalent of the geometric shape input consumed by 3D auto-riggers**. It need not be the hidden source mesh and need not preserve source-rig exactness.

## 2. Formal objective

Let

- `X = {I_v}_{v=1..8}` be the ordered neutral-pose raster observation;
- `O(X)` be the set of quantities legally observable or inferable from `X` without Pose B, hidden rig truth or authored IDs;
- `E_geo = IRIS(X)` be learned geometric evidence;
- `S = SurfaceBuilder(E_geo, X)` be a deterministic canonical rigging surface;
- `G = Geppetto(S)` be a skeleton/hierarchy proposal;
- `W = Arachne(S, G)` be a skinning proposal;
- `Y = Compiler(S, G, W)` be the verified editable puppet.

We seek a minimal evidence contract `E*` such that:

1. **Observability** — `E*` is a function of `X` only at product inference.
2. **Rigging sufficiency** — replacing exact observation-equivalent geometry with `E*` preserves downstream skeleton, skinning and functional deformation quality within frozen non-inferiority criteria.
3. **Minimality** — no learned output head is retained unless matched ablation shows causal downstream value or it is required for safety/abstention.
4. **Derivability discipline** — quantities that can be reconstructed deterministically and robustly from other evidence belong in `SurfaceBuilder`, not automatically in a neural head.
5. **Ambiguity safety** — if `X` does not identify a unique geometric state, IRIS must preserve uncertainty and, where required, multiple supported hypotheses rather than manufacture an unjustified singleton.
6. **Provenance** — downstream decisions must be traceable to supporting views/observations.

The final head list is therefore an **experimental result**, not an architectural assumption.

## 3. Working geometric object

The current best working representation is a dense/common-frame **oriented surfel evidence field**.

For an observation-supported surface sample `i`:

```text
SurfaceEvidence_i
  P_i          common/object-frame position evidence
  N_i          local orientation / normal evidence
  V_i          view-wise visibility / observational support
  U_i          geometric risk / uncertainty evidence
  provenance_i source view / pixel / support metadata
  H_i          optional retained geometric hypotheses when one point is not justified
```

An explicit learned correspondence descriptor `Z` is optional. Cross-view **correspondence/persistence capability is mandatory**, whether represented by `Z`, direct common-frame geometry, reciprocal/cycle evidence, or another verified mechanism.

This list is a **candidate factorization**, not a declaration that every item requires its own neural head. In particular:

- normals may be learned or derived from sufficiently coherent dense geometry;
- visibility/support may be learned, observed or recomputed by reprojection;
- provenance is normally deterministic bookkeeping;
- local adjacency, connected sheets, curvature and surface graph structure should be derived by `SurfaceBuilder` first and become learned outputs only if controlled evidence shows deterministic recovery is insufficient;
- uncertainty may begin as raw predictive risk and becomes calibrated only at the dedicated calibration gate;
- multimodal ambiguity may require a set-valued `H_i`; scalar uncertainty alone is not assumed sufficient.

## 4. Deterministic SurfaceBuilder boundary

`SurfaceBuilder` is not a fourth learned model. Its job is to convert raw observation-grounded evidence into the canonical geometric object consumed by rigging models.

Candidate deterministic responsibilities:

- fuse duplicate cross-view surfels;
- preserve source-view provenance and support counts;
- reprojection and cycle checks;
- construct local neighborhoods / adjacency from image-grid continuity, 3D distance, normal continuity and cross-view support;
- separate discontinuous surface sheets/components;
- compute derived curvature/local differential geometry where stable;
- expose occupancy/thickness or alternate surface hypotheses only when supported by observations or a separately justified learned prior;
- produce fail-closed validity flags when geometry is not sufficiently supported.

No item above is guaranteed necessary. S0 ablations decide the minimum sufficient contract.

## 5. What IRIS does not own

The following are forbidden as mandatory IRIS product authority unless this contract is explicitly revised after controlled evidence:

- Pose B or any second-pose shipping dependency;
- authored mechanical owner identity;
- source-rig exact identity/partition;
- skeleton joints/topology/parents;
- skinning weights;
- mandatory GFDR;
- motion `p_active`, `log_amp`, `dir`, `ΔP` or deformation Jacobians as product-inference primitives.

Historical two-pose/mechanical experiments remain valuable evidence. Their architecture lessons may survive even when their motion-specific outputs do not.

## 6. Preserved lessons from prior research

The following principles remain active because they concern geometric observation quality rather than mandatory motion semantics:

- **high-recall correspondence before irreversible collapse**;
- **reciprocal/cycle consistency** as a cheap cross-view geometric primitive;
- **explicit view/camera geometry** when appearance-only fusion is insufficient;
- **common/object-frame geometry**;
- **geometric grounding and reprojection validation**;
- **visibility/support and uncertainty**;
- **provenance**;
- **set-valued ambiguity preservation** when observations do not support a singleton;
- **relational reasoning**, but translated to static geometry rather than copied as two-pose differential motion.

The following are reserve diagnostics, not core IRIS outputs:

- `p_active`, amplitude and motion direction;
- GFDR / differential mechanics;
- Pose-B response;
- local deformation Jacobians.

## 7. Exact meaning of “2D/8-view equivalent”

The target is **not** “reconstruct the exact hidden 3D mesh.”

A RealSaS `RiggingSurface` is functionally equivalent to the relevant geometric input of a 3D auto-rigger when it preserves the downstream-accessible information needed to generate a clean rig:

- global/common-frame shape organization;
- surface location;
- local orientation where causally useful;
- local surface neighborhood/connectivity where causally useful;
- separation of distinct sheets/components;
- enough coverage of rig-relevant visible/inferrable surface;
- reliability/observability information needed to avoid unsafe decisions.

Any additional field is admitted only if S0 shows that the downstream rigging task materially needs it and that it cannot be recovered safely from the admitted fields.

## 8. Head-derivation rule

A candidate IRIS head `h` is promoted to the product contract only if all are true:

1. the downstream rigging substrate requires the information carried by `h`;
2. that information cannot be reconstructed sufficiently by deterministic `SurfaceBuilder` operations from already-admitted evidence;
3. a matched ablation shows material family-disjoint downstream benefit or a necessary safety benefit;
4. the target is observable from the shipping input or can be represented uncertainty-aware when not uniquely observable.

Therefore:

```text
HEADS = minimal downstream-sufficient observable quantities
```

not

```text
HEADS = historically convenient labels
```

## 9. Canonical research decomposition

```text
S0  Define and prove the minimum rigging-sufficient substrate
    S0-A information/derivability inventory
    S0-B frozen downstream probe ablation
    S0-C freeze minimal RiggingSurface contract

G1  Frozen A×8 baseline: retained IRIS core -> current P/N/V/U candidate factorization

G1.5  Geometry coherence interventions if needed
      reciprocal/cycle, high-recall persistence, deterministic SurfaceBuilder

G2  Camera/ray-aware cross-view fusion if view geometry is causal bottleneck

G3  Representation reformulation if required
      direct common-frame geometry, ambiguity/set-valued hypotheses, richer surface state

G4  Explicit geometric grounding + uncertainty calibration

G5  Geometry-aware local refinement if residual hard-tail remains

G6  Artist-domain robustness / cross-view drawing inconsistency tolerance

G7  IRIS product qualification against the frozen S0 RiggingSurface contract

R1  Geppetto product development
R2  Arachne product development
R3  joint rig quality
C0  Compiler restoration/rebind
P0  end-to-end product gate
```

G1 remains a valid frozen baseline and is not rewritten by S0. S0 determines what the baseline ultimately must be sufficient for.

## 10. Domain-gap rule

The shipping domain includes stylized 2D drawings whose eight views may be only approximately geometrically consistent. Therefore product qualification must eventually test:

- mild cross-view proportion inconsistency;
- silhouette disagreement;
- occlusion and missing detail;
- symmetric/texture-poor ambiguity;
- artist-specific perspective/orthographic deviations.

The response to these cases is robust geometric fusion, explicit uncertainty, grounding and bounded hypotheses — not a return to hidden mechanical labels.

## 11. Change control

This document is the canonical problem-definition authority. Revision is required before:

- changing shipping input away from one neutral 8-view pose;
- declaring a fixed final head list without S0 evidence;
- adding hidden rig/mechanical truth to product inference;
- forcing singleton geometry where observation supports multiple hypotheses;
- replacing downstream rigging sufficiency with source-mesh/source-rig exactness.
