# RealSaS — Joint Mechanical Compatibility and Topology Preservation V1

**Date:** 2026-10-04  
**Status:** `CANONICAL_RESEARCH_PRINCIPLE__TRAINING_CONTRACT_PENDING`

## Discovery

Independent model correctness is not sufficient for product mechanical correctness.

A rig teacher and a skin teacher may each be internally valid and may each be matched with arbitrarily low supervised loss, while their composition on the Compiler's actual mechanical substrate still fails deformation compatibility.

The decisive evidence is the Knight oracle result: even teacher-derived rig/skin evidence did not guarantee G3 compatibility on the current compiled topology. Therefore:

```text
L_ATLAS -> 0
and
L_MIRA  -> 0

does NOT imply

G3(Topology, Rig, Skin) = PASS
```

Sequential conditioning does not fix this by itself. MIRA consuming ATLAS output still does not guarantee that the two predictions are jointly admissible on the exact topology evaluated by the Compiler if their supervision was produced from incompatible truth substrates.

## Core principle

The product object is not `rig + skin` as two independent predictions.

The product object is the composition:

```text
MechanicalSubstrate
  + ControlBasis / Rig
  + Skin / Influence Field
  + Motion Probe Family
  -> Deformation
```

All learned supervision and all qualification must refer to the same substrate identity or to a transfer operation whose mechanical equivalence is separately proven.

This is the topology-preservation requirement.

## Why fixed coherent teachers are different

A coherent rigging asset naturally provides one closed mechanical system:

```text
one mesh topology
one skeleton/control hierarchy
one skin field on that exact mesh
(optional) animation on that exact rig
```

Rig and skin are therefore jointly meaningful by construction.

RealSaS historically allowed different learned components and Compiler stages to be trained/evaluated against separately constructed truths. This creates a composition gap even when every local metric is perfect.

## Architectural consequence

Introduce a single canonical mechanical substrate lineage that survives across ATLAS, MIRA and Compiler qualification.

Conceptually:

```text
IRIS / geometry evidence
        |
        v
Canonical Mechanical Substrate / topology identity
        |
        +-------------------+
        |                   |
        v                   v
      ATLAS                MIRA
 control-basis evidence    influence evidence
        |                   |
        +---------+---------+
                  |
                  v
        Joint mechanical evaluation
                  |
                  v
             Compiler proof
```

MIRA may condition on the qualified ATLAS skeleton, but both outputs must remain attached to the same substrate identity.

## Training objective

Per-model supervised losses remain useful but are incomplete:

```text
L_total =
    L_atlas_supervised
  + L_mira_supervised
  + lambda_joint * L_joint_mechanical
```

`L_joint_mechanical` is a differentiable surrogate for the mechanical quantities enforced by G3 on the same topology and probe family.

Candidate terms:

```text
L_joint_mechanical =
    w_cond  * L_dynamic_condition
  + w_area  * L_dynamic_area_range
  + w_edge  * L_dynamic_edge_ratio
  + w_flip  * L_orientation_or_fold
  + w_cont  * L_component_aware_skin_gradient
  + w_seam  * L_seam_transfer_compatibility
```

For face deformation Jacobian `F_f(q)` under motion probe `q`:

```text
L_dynamic_condition =
  mean softplus(log(kappa(F_f(q))) - log(kappa_max))

L_dynamic_area_range =
  mean [
    softplus(A_min - A_f(q))
    + softplus(A_f(q) - A_max)
  ]

L_dynamic_edge_ratio =
  mean softplus(r_edge(f,q) - r_max)
```

Exact formulation remains experimental. G3 remains the non-differentiable fail-closed authority; the learned loss is only a surrogate intended to make its failures rare.

## Compiler consequence

The same compatibility concept should inform discrete topology/partition decisions.

A repartition or seam creation is not successful merely because it improves a local static/topological objective. Candidate cuts should be evaluated against downstream mechanical compatibility before becoming canonical.

Therefore future partition optimization should support:

```text
candidate topology/cut
 -> predicted or measured joint mechanical compatibility
 -> accept/reject
```

rather than topology optimization followed by an unrelated late G3 surprise.

## 2026-10-04 seam locality counterfactual

On the exact Knight V9 final candidate and partition:

| Arm | Production G3 unsafe | All-face unsafe |
|---|---:|---:|
| current component harmonic | 48 | 58 |
| harmonic top-1 | 141 | 143 |
| harmonic top-2 | 72 | 83 |
| harmonic top-4 | 50 | 64 |

The naive support-locality hypothesis is therefore falsified.

Interpretation:

- the current failures are not solved by independently making seam skin support more local;
- local support changes can resolve some old failures while introducing more new failures;
- compatibility is a property of the composed topology + support field + rig under motion, not a monotonic function of support sparsity/locality;
- the next court should target joint compatibility/topology decisions, not another isolated skin-weight heuristic.

## Relationship to shared encoder

ATLAS/MIRA shared surface encoding remains a valid efficiency/representation experiment, but it is secondary to this requirement.

A shared encoder does not by itself guarantee compatible outputs.

Promotion of a shared encoder must therefore be evaluated through the same joint mechanical objective and Compiler G3 court.

## Binding rules

- `independent_teacher_perfection_implies_product_pass = false`
- `sequential_model_conditioning_implies_joint_compatibility = false`
- `shared_encoder_implies_joint_compatibility = false`
- `canonical_mechanical_substrate_identity_required = true`
- `joint_mechanical_training_objective_required_for_future_court = true`
- `G3_remains_fail_closed_authority = true`
- `topology_or_partition_change_requires_downstream_mechanical_evaluation = true`
