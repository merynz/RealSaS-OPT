# Mage FIT1 real E2E integration audit — 2026-09-12

**Status:** `IN_PROGRESS__EXACT_S_G_W_PERSISTED__REAL_MWB2_X8_WIRED__VISUAL_COVERAGE_GAP_EXPOSED`

This is an integration audit, not a new scientific closure and not a `PRODUCT_PASS` claim.

## Goal

Run the already-closed Mage FIT1 rigging core through the current product stack without replacing, approximating or visually repairing any scientific artifact:

`real 8 images/cameras -> exact S/G/W -> MWB2 x8 -> appearance -> directional joint binding -> static Living Compile inspection -> motion/proof -> native runtime`

If a qualification-owned artifact is absent, the real path must stop or expose an explicit block. Preview-only reconstruction, hidden geometry filling and fabricated proof are forbidden.

## Exact mechanical authority now available

The missing exact V5 product skin serialization has been repaired with a forward-only replay of the historical FP32 V5 closure path. The final Drive transaction is sealed as:

`PASS__EXACT_PROMOTED_V5_S_G_W_AND_FULL_REAL_AUTHORITIES_PERSISTED`

Current exact Mage authorities:

- `RiggingSurfaceIR`: 950 nodes / 2813 local relations;
- `QualifiedSkeletonIR.v2`: 22 Compiler-qualified joints;
- `QualifiedSkinIR`: 950/950 rows;
- surface lineage: `67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb`;
- skeleton lineage: `738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306`;
- skin lineage: `ef28f75e0306dbbabc32e75b837412ede39b180248502f6e504994310d91edaf`;
- replay GSA row-L1 p95: `0.042377868412027`;
- replay articulated deformation ratio: `0.002780768321827054`;
- Compiler total skin correction L1: `2.8930073728035608e-05`;
- full frozen composite V5 checkpoint SHA-256: `820ca65296235803d4c952f6025247e3fa09e3211f5a945a88979968e1dcc629`.

The final evidence tree also contains the exact eight source PNGs and exact eight camera authorities. The teacher supervision bank is retained only under `EVALUATION_ONLY_TEACHER` and is not a product/predictor input.

No unseen-family or product-pass claim follows from FIT1 closure or from serialization replay.

## Integration defects closed

### LC-01 — Living Compile rig overlay projection

Closed on this branch. Living Compile consumes `DirectionalJointViewBindingSetIR` exact per-view pivots instead of support-surface centroids / nearest-node guesses. Missing qualified binding withholds the rig overlay.

### LC-02 — static inspection under blocked/absent product proof

Extended on this branch. Living Compile can now inspect IMAGE / MESH / RIG / WEIGHTS when product proof is either non-PASS **or absent**. Proof absence is reported as `UNAVAILABLE`; no synthetic ABSTAIN proof is created. Runtime clips/frames remain strictly unavailable until a current PASS proof exists.

### ART-01 — exact V5 `QualifiedSkinIR` persistence

Closed. Exact `SkinProposalIR`, Compiler `QualifiedSkinIR`, 950x22 proposal/qualified tensors, conditioning witness, K4x512 field tokens, exact S/G/W bundle and full composite checkpoint are persisted and hash-audited.

## First real downstream execution result

`run_mage_fit1_real_static_v1.py` is the first repository runner that consumes the exact Mage S/G/W authorities and the real 8-view observation set instead of the synthetic four-point fixture.

It performs:

1. strict Mage lineage/cardinality/raster-contract checks;
2. current MWB2 candidate + qualification independently for all eight directions;
3. exact convex mesh-skin transfer from the qualified V5 skin;
4. observation-only appearance binding;
5. `CanonicalPuppetGraphV3` assembly as an explicitly unqualified static candidate;
6. Compiler `DirectionalJointViewBindingSetIR` qualification;
7. proofless static inspection bundle emission with runtime disabled.

The runner explicitly records `product_pass_claimed=false`, `visual_completion_used=false`, `unknown_regions_remain_empty=true` and `full_silhouette_substrate=false`.

### Directional binding result

A direct audit over the exact Mage substrate gives affine rank 4 in all eight directions. Normalized P->raster fit residuals are effectively numerical zero (cardinal-view p95/span approximately `4.1e-16`, `1.43e-15`, `6.17e-16`, `8.95e-16` for S/E/N/W respectively). The qualified skeleton projection is therefore not the current visual blocker.

### MWB2 coverage result

The real MWB2 observed-safe mesh is intentionally conservative and **does not cover the full source silhouette**. Cardinal examples from the exact witness:

| view | observed surface nodes | qualified mesh vertices | qualified faces |
| --- | ---: | ---: | ---: |
| V0 / S | 343 | 276 | 273 |
| V2 / E | 309 | 222 | 220 |
| V4 / N | 343 | 250 | 225 |
| V6 / W | 296 | 221 | 226 |

The holes are not a rendering bug and are not currently repaired. MWB2 only admits target-view-observed surface support and safe relation-supported triangles; UNKNOWN/UNOBSERVED bridges remain forbidden. This is the first real productization blocker exposed by the exact chain.

## Newly identified missing subsystem — visual coverage completion

The repository already defines the authority types and qualification seam:

`VisualCompletionProposalIR -> qualify_visual_completion -> QualifiedVisualCompletionIR`

but this audit has not found a current production **completion proposal producer** capable of turning the uncovered directional regions into separately typed, support-bound proposals. Therefore the real chain must keep those regions empty for now.

The next scientific/product question is not whether to hide the gaps, but which completion class is justified by the evidence:

- deterministic cross-view transport when the missing target region has sufficient admitted support in another real view;
- learned visual completion only for residual regions that cannot be deterministically transported;
- permanent UNKNOWN/empty output when neither path is sufficiently supported.

Any such completion must remain outside S/G/W mechanical truth and must be independently qualified before it may enter a renderable component.

## Real E2E trigger contract

The eventual real trigger must fail closed unless all of the following hold:

- exactly 8 admitted source observations and 8 bound cameras;
- expected S/G/W lineages match the selected witness;
- no teacher mesh/skeleton/skin is consumed by product assembly;
- no mock, synthetic, demo-tessellation or reconstructed-weight artifact can satisfy a real-artifact requirement;
- MWB2 never bridges typed UNKNOWN/UNOBSERVED geometry;
- visual completion, if used, is separately typed, support-bound and qualified and never mutates S/G/W truth;
- per-view rig overlay uses the same qualified directional pivots as motion evaluation;
- missing deformation/motion evidence remains blocked, never synthetic PASS;
- native export remains forbidden until the exact current product has fresh PASS proof;
- every emitted artifact records source lineage/content hashes in one reproducible manifest.

## Current execution order

1. Exact S/G/W persistence — **DONE**.
2. Real MWB2 x8 + exact mesh-skin + observation appearance — **WIRED; coverage gap exposed**.
3. Exact directional joint/view binding — **WIRED; projection quality clean**.
4. Proofless Living Compile static inspection — **WIRED; runtime remains blocked**.
5. Design and qualify real visual-coverage completion without altering mechanical truth — **NEXT BLOCKER**.
6. Add qualification-owned Mage motion evidence and run product proof.
7. Export native runtime only after actual PASS.
8. Promote one fresh-process real E2E trigger after all previous gates close.

## Claim boundary

`MAGE_FIT1_RIGGING_CORE = CLOSED`

`MAGE_EXACT_S_G_W_PERSISTENCE = CLOSED`

`MAGE_REAL_STATIC_DIRECTIONAL_CHAIN = WIRED__VISUAL_COVERAGE_INCOMPLETE`

`MAGE_REAL_PRODUCT_E2E = OPEN`

`PRODUCT_PASS = NOT CLAIMED`
