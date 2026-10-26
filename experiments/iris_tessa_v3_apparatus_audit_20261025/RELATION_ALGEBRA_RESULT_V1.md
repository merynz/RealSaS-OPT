# Relation Algebra Court — Result V1

**Date:** 2026-10-25  
**Preregistration:** `RELATION_ALGEBRA_PREREG_V1.md`  
**Workflow:** `iris-relation-algebra-v1-stage`  
**Run:** `40227715394`  
**Status:** `STRUCTURAL_FACT_ESTABLISHED__ALGEBRA_NOT_ENFORCED`

## Result

Static source audit confirms the current V3 pair representation is ordered:

`[a, b, a-b, a*b]`.

Therefore the current relation parameterization can represent directed quantities, but it does not enforce by construction:

- symmetry for symmetric pair quantities;
- complementarity for front/order under endpoint swap;
- symmetry for projected boundary relations.

The current output groups are:

Static:
- `same_drawable_affinity`
- `cut_probability`
- `attachment_affinity`

View-conditioned:
- `front_probability`
- `occlusion_probability`
- `projected_alpha_boundary_probability`
- `projected_appearance_boundary_probability`

The architecture can in principle learn some desired algebra from data, but the algebra is not guaranteed structurally.

## Interpretation

This is not evidence that a trained checkpoint necessarily violates symmetry or front complementarity. It establishes that current V3 asks training data/objectives to learn invariants that may be mathematically known from the causal truth definition.

Before causal relation training, every truth family must receive an explicit algebra class:

- `SYMMETRIC`
- `DIRECTED`
- `COMPLEMENTARY_OR_TIE_AWARE_ORDER`
- `RAY_GLOBAL`

Candidate family-specific parameterizations should then be compared against the current unconstrained generic head under matched capacity.

## Consequence

No current head is changed yet because the correct algebra depends on the final causal truth quantities. However, retaining one undifferentiated static sigmoid vector merely for checkpoint convenience is no longer an architecture assumption that can go untested.

## Authority

No Product, Truth139, FIT8 or ModelRelease authority changed.
