# RealSaS N1D — Functional Quotient Existence Audit V1 — Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_EXISTENCE_AUDIT__NO_SOLVER_CHANGE__NO_RETUNE`

## Question

Does the current geometric hard tail actually imply multiple downstream mechanical/rig-functional states, or are some geometrically distinct M256 solutions equivalent under the frozen observable-mechanics contract?

The target is **not** teacher/geometric singleton:

```text
|Q_geometry(O)| = 1
```

The relevant target is functional singleton:

```text
|Q_functional(O)| = 1
```

where equivalence is defined only by downstream consequences that the current frozen GFDR-V2 observable-mechanics contract can represent.

## Scope / limitation

This V1 audit is an existence test under the frozen **GFDR-V2 mechanical functional contract** (G/F/D/R consequences). The broad e00 sidecars available for the canonical eight-family panel do **not** contain M5 dense skin-weight truth. Therefore:

- a V1 PASS is evidence that exact geometric singleton is **not required by GFDR-V2 mechanics** on the tested hard tail;
- it is **not yet a theorem of dense-skinning equivalence**;
- a V1 FAIL means current functional singleton is not established and research resumes; it does not by itself prove that exact teacher geometry is the only possible solution.

No claim stronger than this scope is authorized.

## Frozen population

Same truth-open broad e00 panel:

```text
09908, 11032, 12772, 13203,
14404, 14702, 14758, 15290
```

Frozen geometry/search/solver state:

- F16 feasible candidate construction;
- deterministic M256 compression/order;
- raw unary;
- graph;
- R_REL_DIS;
- degree weights;
- two-round synchronous cavity min-sum P1.

Primary nodes are unchanged:

```text
reliable
AND M256_contained2x
AND active
AND full_F16_contained2x
```

Expected primary `n=261`.

A **geometric hard-tail witness** is a primary node whose frozen P1 round-2 selected endpoint has normalized teacher-endpoint error `>2`. From the frozen parent aggregate this population is expected to contain 48 nodes; the exact count is a validity guard, not a target to tune.

## Validity guards

Before interpreting functional results, reproduce all of:

```text
P1 round1 candidate-index parity        512/512
P1 round2 candidate-index parity        512/512
primary n                               261
round1 primary contain2                 .7777777778
round2 primary contain2                 .8160919540
round2 worst-family contain2            .5200000000
geometric hard-tail witness count       48
```

Any mismatch => `INVALID_PARENT_PARITY_FAIL`; repair execution only, without changing this preregistration.

## Counterfactual pair for each hard-tail witness

For hard-tail node `i` in family `f`:

1. `x_round2`: the complete frozen P1 round-2 64-node candidate configuration.
2. `k_teacher_near(i)`: evaluator-only M256 candidate with minimum Euclidean distance to the exact Pose-B target of node i; deterministic lowest-index tie break.
3. `x_swap(i)`: identical to `x_round2` except node i is replaced by `k_teacher_near(i)`.

The teacher target is used only to construct this evaluator counterfactual. It never enters the solver or an observation-native score.

The comparison is therefore:

```text
current geometrically-wrong solver state
vs
same global state with only this node repaired to teacher-nearest feasible M256 geometry
```

This isolates whether the geometric discrepancy at that witness changes downstream mechanical function.

## Pose-B normal / visibility attachment

GFDR-V2 requires P/N/V. Candidate positions are M256 3-D points; candidate normals/visibility are attached by this deterministic evaluator-only rule fixed before results:

- map each selected Pose-B candidate point to the nearest exact dense Pose-B `surface_points` sample from that family's e00 observation sidecar;
- lowest dense-surface index breaks ties;
- copy that sample's `surface_normals` and `surface_visibility` as candidate N_B/V_B.

P_A/N_A/V_A remain the frozen carrier-A quantities used by the canonical candidate runner.

This mapping does not choose candidate geometry and is applied identically to round2 and swap states.

## Local functional motif

Global whole-character metrics can dilute a single-node change. Therefore the **primary functional comparison is local**.

For witness i define a frozen motif:

```text
S_i = {i} union N_graph(i)
```

where `N_graph` is the exact frozen solver graph. If `|S_i| < 4`, extend deterministically with the nearest P_A carriers by Euclidean distance until four carriers are present, because GFDR-V2 requires at least four carriers.

No truth is used to define the motif.

For descriptive context only, the same comparison is also computed over all 64 carriers.

## Frozen GFDR-V2 functional descriptor

Use the exact canonical `realsas_gfdr_v2.compute_gfdr_v2` implementation from the N1D canonical proof pack. No field definitions are changed.

For each local motif, compute GFDR-V2 for `x_round2` and `x_swap` and compare the following authoritative blocks directly:

### F finite response

```text
F_delta_normalized
F_response_features
F_coresponse_kernel
```

### D differential response

```text
D_surface_action
D_local_residual_rms
D_J_surface_gradient
```

### R relative response

```text
R_rel_B
R_motion_distance
R_transfer_error_normalized
```

### G articulation-locus response

```text
G_axis_direction
G_axis_point
G_support
```

## Dimensionless block distances

All thresholds below are fixed before inspection.

For any array block A/B, use normalized RMS:

```text
NRMS(A,B) = RMS(A-B) / (RMS(B) + 1e-8)
```

except:

- `G_axis_direction`: support-weighted angular disagreement `1-|dot|`, evaluated only where either state's `G_support >= .25`;
- `G_axis_point`: RMS point difference divided by P_A motif RMS radius;
- kernel block: plain RMS because the kernel is already bounded;
- support block: plain RMS because support is already bounded.

Define block-level local-equivalence gates:

```text
F_delta_NRMS             <= .10
F_response_NRMS          <= .15
F_kernel_RMS             <= .10

D_surface_action_NRMS    <= .15
D_residual_NRMS          <= .20
D_gradient_NRMS          <= .20

R_relB_NRMS              <= .15
R_motion_NRMS            <= .20
R_transfer_NRMS          <= .20

G_direction_disagree     <= .15
G_axis_point_norm_RMS    <= .20
G_support_RMS            <= .15
```

If no supported G axis exists in either state on a motif, G is `ABSTAIN_EQUIVALENT` rather than failed.

A witness is `GFDR_FUNCTIONALLY_EQUIVALENT` only if **all non-abstained local block gates pass**.

These thresholds are deliberately tighter than product qualification gates because this audit asks whether two geometries are effectively the same mechanical state, not merely whether either state is acceptable in isolation.

## Primary decision statistics

Report:

- pooled equivalent fraction among all 48 geometric hard-tail witnesses;
- per-family equivalent fraction and witness count;
- equivalent fraction for classical hard families `11032,13203,15290` pooled;
- each block's pass fraction;
- number of witnesses where geometry is >2 local spacings wrong but all functional blocks are equivalent;
- median/p90 normalized teacher endpoint error separately for equivalent and non-equivalent witnesses.

## Frozen decision tree

### A — geometry singleton not required by frozen GFDR contract

`GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_GFDR_FUNCTIONAL_CONTRACT_V1` only if:

```text
pooled functional-equivalence fraction          >= .80
hard-family pooled equivalence                  >= .75
and every family with >=5 hard-tail witnesses   >= .60
```

This says the dominant geometric hard tail collapses under the existing mechanical-functional quotient. It does **not** establish M5 dense-skinning equivalence.

### B — functional non-singleton is material

`FUNCTIONAL_NON_SINGLETON_MATERIAL__RETURN_TO_RESEARCH` if:

```text
pooled functional-equivalence fraction <= .50
```

or if any family with >=5 hard-tail witnesses has equivalence `<= .30`.

This means geometrically distinct states frequently induce materially different GFDR mechanics. The project returns to research: first determine whether additional raster-observable typed evidence can separate the functional classes; geometric singleton / richer observation is only one possible remedy.

### C — ambiguous middle

Otherwise:

`FUNCTIONAL_QUOTIENT_NOT_YET_SINGLETON__RETURN_TO_RESEARCH`

No architecture promotion is allowed from an ambiguous middle result.

## Descriptive observation-native ambiguity check

For every non-equivalent witness, also report the teacher-nearest candidate's normalized rank in the frozen round-2 belief vector, when available. This is descriptive only.

A low rank together with functional non-equivalence is evidence that the current observation-native objective admits a mechanically distinct alternative near its preferred basin. It does not change the decision tree.

## Forbidden

- no R_REL_DIS retune;
- no solver depth sweep;
- no lambda/damping/softmin tuning;
- no F/D factor injection during this test;
- no free XYZ;
- no truth-conditioned solver choice;
- no threshold changes after results;
- no sealed21/external10;
- no large/end-to-end training.

## Follow-up authority

- If decision A: retain set-valued/canonical geometry; next required proof is M5/dense-skinning functional equivalence on data containing dense weight/deformation truth.
- If decision B or C: research resumes. The first research question is **functional-class observability**, not automatically exact teacher geometry reconstruction.
