# RealSaS — Mechanically Optimal Control Basis V1

**Date:** 2026-10-04  
**Status:** `CANONICAL_RESEARCH_PRINCIPLE__NO_PRODUCT_PROMOTION`

## Core law

Rig cardinality is not a supervised target and is not a quality proxy.

```text
NOT: teacher has K controls -> RealSaS should have K controls
NOT: fewer controls -> better
NOT: more controls -> better

YES: grow where mechanics requires capacity;
     prune where mechanics cannot distinguish utility.
```

Teacher skeletons, deform flags, direct skin mass, third-party dense rigs and learned proposals are evidence only. None owns final control count.

For candidate control basis `S`, define a mechanical error vector

```text
E_mech(S) =
  [deformation_error,
   relative_articulation_error,
   contact_error,
   silhouette_or_volume_error,
   strain_or_shear_error,
   local_editability_error,
   motion_probe_instability,
   recovery_or_repair_burden]
```

and a complexity vector

```text
C(S) =
  [control_count,
   active_weight_support,
   runtime_cost,
   conditioning_cost,
   redundancy]
```

The desired basis is selected lexicographically:

```text
S* = argmin C(S)
     subject to E_mech(S) <= admitted mechanical proof thresholds.
```

Mechanical sufficiency outranks compactness. Complexity is minimized only among mechanically admissible solutions.

## Marginal utility

For control `c`:

```text
DeltaU(c | S) = E_mech(S) - E_mech(S union {c})
```

Pair/group synergy must also be measurable because twist/helper/corrective controls may have weak isolated utility but strong joint utility.

A zero-direct-skin-mass control is therefore **not** a deletion criterion. A teacher deform flag is **not** an admission criterion.

## Grow-to-need / prune-to-proof

Grow the basis only where a bounded failure signature proves missing capacity, including unresolved bend/twist decomposition, contact drift, silhouette/volume collapse, excessive strain/shear, motion-space collapse, non-local edit compensation or recurring Compiler repair burden.

Prune or merge a control only when all required probe families remain within proof thresholds and no required synergy is destroyed.

Stop when all required mechanical gates pass and every bounded removal/merge causes a proof regression or exceeds the frozen redundancy threshold.

## Capacity envelopes

Sweeps such as `K={8,12,16,24,32,48,64,...}` are diagnostic capacity envelopes, not target control counts. The final cardinality is emergent.

A subject may legitimately require fewer controls than its teacher, the same number, or more.

## Typed product-facing result

The intended Compiler-side abstraction is `MechanicalControlBasisIR`, containing:

- admitted control proposal provenance;
- parent/link evidence before final canonicalization;
- per-control marginal utility summary;
- synergy-group evidence where required;
- probe families and thresholds used for admission;
- residual mechanical error;
- prune/redundancy certificates;
- cardinality as an emergent result.

Compiler remains canonical authority for final legal tree, root, IDs and product admission.

## Knight interpretation boundary

The 2026-10-04 Knight court reporting `K0=20`, `K1=41`, and `0/21` nonzero internal articulation after common-mode removal is a narrow diagnostic only.

It does **not** prove that 20 controls are mechanically optimal and does not authorize deleting the extra 21 controls. Their full mechanical utility still requires the complete probe contract, including group synergy and non-skinning consequences.

Dense reference rigs likewise do not prove that high cardinality is intrinsically superior; they demonstrate that twist/helper/corrective capacity must not be ruled out by a low-cardinality prior.

## Binding flags

- `teacher_joint_count_is_target = false`
- `direct_skin_mass_zero_implies_delete = false`
- `fewest_controls_wins = false`
- `largest_rig_wins = false`
- `mechanical_sufficiency_first = true`
- `complexity_minimized_after_sufficiency = true`
- `cardinality_emergent = true`
- `compiler_final_authority = true`
