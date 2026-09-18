# B FULL Zero-Surface -> Dense Product Candidate -> Runtime-v4 BODY_UNDERLAY Lineage Diff V1

Date: 2026-09-18  
Status: CLOSED_AUDIT__DRAWABLE_REPRESENTATION_REGRESSION_LOCATED__PEPPER_ROOT_CAUSE_NOT_YET_PROVEN  
Execution authority for the next causal render: SELF_HOSTED_REALSAS_RUNNER_ONLY  
Audited Runtime-v4 branch parent: 2dbf386e2e989fc2b627dde752fca84b58237bc6

## Executive conclusion

The visually strongest column in the 2026-09-16 visibility causal experiment ("B FULL zero-surface") was **not** the representation later carried into Runtime-v4.

The lineage has three materially different stages:

1. **B FULL diagnostic reference** — one dense zero-surface domain, rendered with fixed rest-camera z-buffer, no selected-front face pruning.
2. **Dense selected/product candidate** — view-specific dynamically safe / visibility-selected subsets derived from the zero surface.
3. **Current Runtime-v4 BODY_UNDERLAY** — the product returned to the Sep14 P1Q directional mesh authority through the typed component assembly.

The source surface, skeleton, and skin authority lineages are the same across the dense and P1Q families. The major change is therefore **drawable geometry/topology + mesh-skin address + appearance/provenance policy**, not a replacement of the qualified skeleton or source skin model.

This audit does **not** claim that restoring B FULL is correct. B was diagnostic and partly used non-product appearance fill. The correct next candidate is a **source-authoritative, coverage-complete drawable body substrate**, not the old diagnostic B render verbatim.

## Why B looked good but was not accepted as the product

The corrected ABC experiment explicitly defined:

- A = sealed selected face set + sealed painter order.
- B = full original zero surface + fixed rest-camera depth z-buffer.
- C = B minus strong rigid-owner evidence with no BODY contradiction; invisible faces kept.

The same rasterizer was used for all arms, motion was unchanged, and weights were unchanged. Therefore the A/B/C comparison was primarily a drawable-face/visibility experiment, not a motion or skin change.

However, B's human-readable texture was not product appearance authority:
- total B/C face universe: 507,377 faces;
- source-coloured diagnostic faces: 372,278 (73.373%);
- nearest-surface-fill diagnostic faces: 135,099 (26.627%);
- the diagnostic report explicitly labels that texture as human-readable only, not product texture authority.

The corrected decision also explicitly says:
- B total rest is descriptive, not a gate;
- the diagnostic texture is not product texture authority;
- the experiment does not claim zero-surface component fusion is correct;
- it does not test 3D motion correctness;
- it does not test weight interpolation.

So B was a **valuable geometry/selection reference**, not a shippable RealSaS representation.

## Lineage comparison

| Dimension | B FULL diagnostic reference | Dense selected/product candidate | Current Runtime-v4 BODY_UNDERLAY | Audit meaning |
|---|---|---|---|---|
| Geometry family | Full original zero surface | Zero-surface compaction / view selection | P1Q directional meshes | Runtime-v4 did not preserve the B substrate family |
| Global/dense source topology | 257,505 dense zero-surface vertices; 507,377 faces in the ABC B/C universe | Same zero-surface source, compacted per view | 8 separate directional P1Q meshes | Representation changed from one dense surface domain to view-local product meshes |
| P1/P1Q topology used | false | false | true | Strong topology lineage break |
| Teacher/source mesh topology used | false | false | P1Q is its own current directional authority | Dense family was independently extracted |
| Source surface lineage | 65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da | same | same | Surface evidence did not change |
| Skeleton lineage | 69f05e4fdef65f2cd86fed66503911210e1dbc7212ad017d69bf3dcb7b0896a1 | same | same | Rig authority did not change |
| Source skin lineage | eb96b398282e4e25cd6df662e7a02e8ff1afe881818127190979bddd2f6c4006 | same | same | Skin authority did not change |
| Mesh-skin address | dense-zero-surface mesh skin | compacted zero-surface mesh skin | P1Q mesh-skin binding | Same source W, different drawable vertex addresses/bindings |
| Fresh model query | no | no | no | Difference is compiler representation, not new model inference |
| Historical FIT1 skin | no | no | no | No legacy skin regression |
| Weight-row mutation | not a model refit; deterministic dense binding | deterministic compacted binding | P1Q manifest says weight rows mutated = false | Source W preserved |
| Appearance in displayed B column | diagnostic texture: source colour + nearest-surface fill | source-qualified appearance artifacts exist for materialized selected mesh | P1Q qualified appearance IR + Runtime-v4 source provenance | B appearance cannot simply be promoted |
| Face visibility policy | no selected-front pruning in B | selected / dynamically safe per-view face set | current P1Q owner-view BODY asset | Visibility/substrate policy changed twice |
| Rigid ownership | B keeps full geometry | experiments later remove/partition strong rigid owners | rigid components are separate foreground attachments over BODY_UNDERLAY | Rigid ownership must not imply body substrate deletion |
| Product pass | no | no global product pass | no | None of these artifacts alone closes product quality |

### Count caveat

B is a single dense zero-surface domain, while current P1Q is eight directional meshes. Therefore raw face/vertex totals are **not an apples-to-apples complexity comparison**. They are included to establish that these are materially different representations, not revisions of the same mesh.

## Per-view geometry / coverage comparison

The dense manifest's candidate count describes the full zero-surface faces relevant to each view. "Dense selected" is the materialized zero-surface product candidate. P1Q is the current directional BODY geometry used by component assembly.

| View | Dense candidate faces | Dense selected faces | Dense selected vertices | Dense rest alpha recall | P1Q faces | P1Q vertices | P1Q rest alpha recall | P1Q uncovered components | Largest P1Q uncovered component px |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 55,862 | 13,437 | 17,995 | 98.0002% | 14,219 | 8,383 | 97.2143% | 509 | 5,425 |
| V1 | 85,872 | 26,417 | 31,367 | 98.0003% | 14,659 | 8,569 | 98.9480% | 599 | 1,552 |
| V2 | 128,492 | 39,487 | 49,766 | 98.0001% | 16,221 | 9,739 | 97.6755% | 732 | 5,457 |
| V3 | 154,361 | 49,732 | 58,228 | 98.0002% | 14,704 | 8,633 | 98.3377% | 530 | 3,341 |
| V4 | 168,101 | 47,401 | 59,675 | 98.0003% | 14,605 | 8,531 | 97.3267% | 517 | 5,449 |
| V5 | 159,154 | 50,263 | 59,006 | 98.0003% | 15,329 | 8,864 | 98.8632% | 516 | 1,905 |
| V6 | 107,816 | 36,029 | 44,566 | 98.0000% | 15,489 | 9,379 | 97.5021% | 691 | 5,457 |
| V7 | 88,448 | 28,898 | 34,446 | 98.0003% | 13,787 | 8,078 | 98.1964% | 492 | 3,239 |

Per-view totals:
- dense selected face instances: 291,664;
- dense selected vertex instances: 355,049;
- current P1Q face instances: 119,013;
- current P1Q vertex instances: 70,176.

Again, these totals are across view-local materializations and should not be interpreted as unique global topology counts.

## Why work continued after the attractive B render

There were at least four explicit reasons in the preserved experiment artifacts:

1. **B appearance was diagnostic, not admissible.** 26.627% of B/C faces used nearest-surface diagnostic fill rather than source-authoritative product texture.
2. **The first ABC v1 comparison was confounded.** It compared C BODY against B FULL and therefore counted intentionally removed rigid faces as holes.
3. **Corrected ABC v2 narrowed B to a same-view strong-rest-BODY reference.** It did not promote all hidden/full zero-surface faces to product truth.
4. **B did not prove the motion/skin path.** The decision guardrails explicitly say the experiment did not test 3D motion correctness or weight interpolation.

Therefore continuing after B was justified. The mistake was not "we ignored a finished solution"; the likely regression is that the useful lesson from B — **keep a continuous drawable substrate and separate it from visibility/occlusion ownership** — was not carried forward cleanly into the later product/runtime representation.

## Selection-pathology evidence from the same family

The corrected ABC diagnostics support a face-selection problem on the dense family:
- selected connected components ranged from 4,510 (V0) to 14,186 (V4);
- 44.3% to 68.4% of selected faces, depending on view, were not first-hit anywhere in the rest view;
- dynamic missing verified-BODY fraction increased from idle to run in all 8 views.

This establishes a causal warning about **selected-surface architecture**. It must not be mechanically transferred to P1Q as if the exact topology were identical, but it is directly relevant to any RealSaS design that turns visibility evidence into destructive face deletion.

## Rigid ownership finding

The ABC C-arm classification operated on 507,377 faces and removed 222,773 faces with:

`INTERIOR_RIGID_GT_0_AND_INTERIOR_BODY_EQ_0`

That experiment was diagnostic. It is **not** a valid general rule that a region owned by a movable rigid foreground component should be deleted from the deformable BODY substrate.

For runtime composition, the safer architectural invariant is:

`BODY_DRAWABLE_SUBSTRATE_EXISTS_INDEPENDENTLY_OF_CURRENT_RIGID_OCCLUSION`

Rigid components may occlude BODY during composition; ownership/visibility must not erase the revealable BODY substrate unless the Compiler has an explicit non-revealability proof.

## Current component assembly note

Current typed component assembly uses:
- BODY_UNDERLAY as a DEFORMABLE_COMPONENT with 22 active joints;
- BOOK, STAFF, HAT and CAPE as RIGID_BONE_ATTACHMENT foreground components.

The CAPE classification is a separate known product-design mismatch: a chest-rigid cape is acceptable only as a temporary demo simplification and is not the intended secondary-motion architecture.

## Most important preserved-vs-lost authority result

### Preserved
- observable/source surface lineage;
- qualified skeleton lineage;
- qualified source skin lineage;
- no new model query;
- no historical FIT1 skin fallback.

### Changed / lost
- full zero-surface drawable topology was not preserved into Runtime-v4;
- dense zero-surface mesh-skin addresses were replaced by P1Q drawable addresses;
- B's full-substrate visibility semantics were not preserved;
- B's diagnostic appearance could not be reused because it included non-authoritative nearest-surface fill;
- Runtime-v4 therefore has the correct current product authority, but not the continuous dense substrate that made B visually robust.

## Concrete next implementation decision

Do **not** restore B verbatim.

Define a new product-domain object, provisionally:

`QualifiedDrawableCoverageSubstrateV1`

Required properties:

1. derived from qualified source/surface evidence;
2. coverage-complete for all BODY regions that may be revealed by movable foreground attachments;
3. retains the current qualified skeleton and source-skin lineages;
4. deterministic skin transfer/binding onto the drawable substrate;
5. exact source/donor provenance per drawable face/corner;
6. no nearest-surface colour fill or other diagnostic appearance completion;
7. foreground rigid ownership affects composition/occlusion, not destructive BODY face deletion;
8. dynamic geometry qualification remains separate from visibility;
9. Compiler owns qualification; Runtime consumes a sealed substrate and does not infer missing surfaces.

## Next causal test after implementation

One self-hosted run, same:
- motion;
- skeleton;
- source skin;
- rasterizer;
- camera;
- foreground attachments.

Only BODY substrate changes:

A. current P1Q BODY_UNDERLAY  
B. source-authoritative coverage-complete substrate

Render both:
- BODY-only;
- BODY + rigid foreground;
- same exact sampled times.

Measure:
- source-authoritative reveal coverage under moving rigid regions;
- interior hole count / area;
- temporal hole birth/death;
- final visible-alpha spill;
- Compiler-vs-Runtime posed-vertex parity.

If B fixes the large reveal holes without source-authority violations, the drawable-substrate regression is causally confirmed. If peppering remains inside fully covered regions, continue to dynamic deformation / Runtime interpolation diagnostics.

## Audit disposition

Primary regression candidate:
`FULL_ZERO_SURFACE_CONTINUOUS_SUBSTRATE_NOT_CARRIED_FORWARD_TO_CURRENT_RUNTIME_BODY`

Confidence:
`HIGH_AS_LINEAGE_FACT__NOT_YET_CAUSAL_ROOT_CAUSE_PROOF`

Peppering root cause:
`OPEN`

Large under-rigid empty-region cause:
`STRONGLY_SUSPECT_DRAWABLE_SUBSTRATE / OWNERSHIP_SEPARATION`

Product pass claimed:
`false`
