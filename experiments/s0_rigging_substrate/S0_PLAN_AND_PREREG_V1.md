# RealSaS S0 — Rigging Substrate Sufficiency Plan / Prereg V1

**Date:** 2026-08-22  
**Status:** `S0_A_AUTHORIZED__NO_PRODUCT_HEAD_LIST_YET`

## Scientific question

> What is the minimum observation-grounded geometric information that must be delivered from one neutral 8-view character observation so that downstream skeleton generation and skinning can recover a clean, editable, functionally valid rig?

This gate exists before product Geppetto/Arachne development so that their input contract is evidence-derived rather than guessed.

## Non-negotiable controls

- Product inference input remains `A×8` only.
- Pose B is forbidden as a required input or target-construction dependency.
- Hidden authored owner IDs/mechanical partitions are forbidden as IRIS inputs.
- Teacher rig/skeleton/weights may be used only as downstream supervision/evaluation truth for S0-B probes.
- The same family split, probe capacity, optimizer budget and downstream evaluator must be used across field-ablation arms.
- No field is called “required” from intuition or literature alone.
- No deterministic quantity becomes a learned head until a deterministic derivation baseline has been tested.

# S0-A — Information / derivability inventory

## Goal

Build a source-of-truth matrix with one row per candidate geometric quantity and answer:

1. Is it available in the 3D rigging references/corpus?
2. Is it observable/inferable from `A×8`?
3. Is it directly supervised by current RealSaS sidecars?
4. Can it be deterministically derived from another admitted field?
5. Which downstream query would use it?
6. What controlled experiment would prove necessity?

## Required candidate inventory

At minimum audit:

- common/object-frame position `P`;
- surface normal/orientation `N`;
- per-view visibility/support `V`;
- geometric uncertainty/risk `U`;
- source-view/pixel provenance;
- cross-view correspondence/persistence / optional `Z`;
- local adjacency / neighborhood graph;
- connected surface sheets/components;
- local curvature / differential geometry;
- occupancy / thickness evidence;
- silhouette/support boundary evidence;
- alternate/multimodal geometric hypotheses `H`;
- optional appearance/semantic surface features only as a last-resort downstream aid.

## S0-A output

`S0_A_INFORMATION_DERIVABILITY_MATRIX.json`

Every candidate must receive one provisional status:

- `LEARNED_PRIMITIVE_CANDIDATE`
- `DETERMINISTIC_DERIVED_CANDIDATE`
- `OPTIONAL_INTERNAL_LATENT`
- `RESERVE_ONLY`
- `NOT_OBSERVABLE_WITHOUT_UNCERTAINTY`
- `REJECTED_FOR_PRODUCT_INFERENCE`

S0-A does **not** promote the final head list.

# S0-B — Frozen downstream probe ablation

## Principle

The substrate is defined by downstream sufficiency, not geometric elegance alone.

Use fixed-capacity research probes, not product models, to prevent target definition from depending on a changing Geppetto/Arachne architecture.

## Probe 1 — GeppettoProbe

Input: candidate substrate arm.  
Output/evaluation target: clean skeleton joint/control locations + hierarchy/parents, using the existing canonical skeleton truth mapped to product controls.

Purpose: measure which surface fields materially help skeleton recovery.

## Probe 2 — ArachneProbe

Input: candidate substrate arm + **ground-truth product skeleton**.  
Output/evaluation target: dense/sparse skinning weights.

Purpose: isolate substrate sufficiency for skinning from Geppetto error.

## Probe 3 — JointProbe

Input: candidate substrate arm.  
Pipeline: `GeppettoProbe -> ArachneProbe -> compiler-side deformation checks`.

Purpose: measure compounded functional rig quality.

## Initial matched ablation ladder

The exact implementation must keep all non-substrate variables fixed.

```text
A0  P only
A1  P + N_derived_from_P
A2  P + N_learned/exact
A3  A2 + deterministic surface graph
A4  A3 + V/support + provenance
A5  A4 + explicit ambiguity/hypothesis representation where needed
A6  A5 + occupancy/thickness evidence if available
A7  A6 + appearance/semantic surface latent ONLY if geometry-only arms leave a reproducible gap
```

Additional arms may be added only by prereg amendment before their result is inspected.

`N_derived_from_P` versus explicit `N` is intentionally separated: normals do not become a required IRIS head merely because 3D auto-riggers consume normals.

Likewise surface graph/adjacency is first tested as a deterministic `SurfaceBuilder` product, not a neural head.

## Metrics

### GeppettoProbe

At minimum:
- matched joint/control localization normalized by object scale;
- parent/hierarchy accuracy after canonical product-control matching;
- missing/extra control rate;
- hard-tail family statistics;
- clean-editable structural validity.

### ArachneProbe

At minimum:
- weight distribution error;
- support overlap / dominant-control agreement;
- boundary error;
- deformation error under frozen standardized probes;
- hard-tail family statistics.

### JointProbe / Compiler

At minimum:
- deformation probe pass/failure signatures;
- silhouette preservation under motion;
- fold/tear/invalid deformation rate;
- rig editability/validity;
- abstention/repair rate where uncertainty is used.

All metrics report family mean, median, p90/p95 or equivalent hard-tail statistics; no mean-only promotion.

## Decision rule

A candidate field becomes `REQUIRED_SUBSTRATE_INFORMATION` only when a matched field addition or removal demonstrates a material, reproducible downstream or safety effect on family-disjoint evaluation.

A field becomes `REQUIRED_LEARNED_HEAD` only when it is required substrate information **and** deterministic derivation from already admitted evidence is insufficient.

A field remains deterministic when its derived form is downstream-noninferior to a directly supplied/learned counterpart.

Numerical non-inferiority margins must be frozen after the first scale-calibration pilot and before confirmatory S0-B evaluation. Pilot families used for margin calibration become open development and may not be reused as confirmatory qualification.

# S0-C — Contract freeze

S0-C emits:

- `RIGGING_SURFACE_CONTRACT_V1.json`
- exact primitive learned outputs required from IRIS;
- exact deterministic `SurfaceBuilder` outputs;
- uncertainty/ambiguity representation rules;
- required downstream geometry metrics;
- explicit rejected/reserve fields with evidence.

Only after S0-C may the provisional IRIS head list be called the final product substrate contract.

# Relationship to frozen G1

G1 is preserved exactly as frozen. It remains the smallest retained-core baseline for the current candidate factorization `P/N/V/U`.

S0 may later show that:

- some G1 outputs are derivable and need not remain learned heads;
- additional geometry information is required;
- the current factorization is already sufficient.

Any such change applies to post-G1 architecture gates and must not rewrite the G1 baseline.

# Execution order

1. `S0-A`: inventory current corpus/sidecar/reference rigging inputs and derivability.
2. Freeze the S0-A matrix and candidate arms.
3. Implement fixed GeppettoProbe/ArachneProbe interfaces.
4. Run a tiny open-development execution/parity pilot.
5. Freeze S0-B split, optimizer, evaluator and numerical decision margins.
6. Run family-disjoint S0-B ablations.
7. Freeze `RIGGING_SURFACE_CONTRACT_V1` in S0-C.
8. Resume/interpret G1 against the S0-C target and proceed through geometry gates.
