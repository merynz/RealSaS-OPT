# Mage FIT1 real E2E integration audit — 2026-09-12

**Status:** `IN_PROGRESS__EXACT_S_G_W_PERSISTED__REAL_MWB2_X8_WIRED__MWB2_COVERAGE_FAILURE_EXPOSED`

This is an integration audit, not a new scientific closure and not a `PRODUCT_PASS` claim.

## Goal

Run the already-closed Mage FIT1 rigging core through the current product stack without replacing, approximating or visually repairing any scientific artifact:

`real 8 images/cameras -> exact S/G/W -> directional M/B -> appearance -> directional joint binding -> static Living Compile inspection -> motion/proof -> native runtime`

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

Closed on this branch. Living Compile can inspect IMAGE / MESH / RIG / WEIGHTS when product proof is either non-PASS **or absent**. Proof absence is reported as `UNAVAILABLE`; no synthetic ABSTAIN proof is created. Runtime clips/frames remain strictly unavailable until a current PASS proof exists.

### ART-01 — exact V5 `QualifiedSkinIR` persistence

Closed. Exact `SkinProposalIR`, Compiler `QualifiedSkinIR`, 950x22 proposal/qualified tensors, conditioning witness, K4x512 field tokens, exact S/G/W bundle and full composite checkpoint are persisted and hash-audited.

## First real downstream execution result

`run_mage_fit1_real_static_v1.py` is the first repository runner that consumes the exact Mage S/G/W authorities and the real 8-view observation set instead of the synthetic four-point fixture.

It performs:

1. strict Mage lineage/cardinality/raster-contract checks;
2. current relation-complex MWB2 candidate + qualification independently for all eight directions;
3. exact convex mesh-skin transfer from the qualified V5 skin;
4. observation-only appearance binding;
5. `CanonicalPuppetGraphV3` assembly as an explicitly unqualified static candidate;
6. Compiler `DirectionalJointViewBindingSetIR` qualification;
7. proofless static inspection bundle emission with runtime disabled.

The runner explicitly records `product_pass_claimed=false`, `visual_completion_used=false`, `unknown_regions_remain_empty=true` and `full_silhouette_substrate=false`.

### Directional binding result — clean

A direct audit over the exact Mage substrate gives affine rank 4 in all eight directions. Normalized P->raster fit residuals are effectively numerical zero (cardinal-view p95/span approximately `4.1e-16`, `1.43e-15`, `6.17e-16`, `8.95e-16` for S/E/N/W respectively). The qualified skeleton projection is therefore not the current visual blocker.

### Current MWB2 relation-complex result — real coverage failure

The current implementation is conservative: target-view-visible S nodes + safe local-relation cliques + deterministic non-overlap selection. It does not use source mesh topology or hidden completion.

Cardinal exact-witness geometry:

| view | observed S nodes | qualified mesh vertices | qualified faces | source-alpha coverage |
| --- | ---: | ---: | ---: | ---: |
| V0 / S | 343 | 276 | 273 | 20.05% |
| V2 / E | 309 | 222 | 220 | 22.43% |
| V4 / N | 343 | 250 | 225 | 18.18% |
| V6 / W | 296 | 221 | 226 | 23.76% |

Across all eight views the measured source-alpha coverage is approximately `18.18%..32.31%`, while rasterized mesh spill outside source alpha stays below roughly `0.52%`. Therefore the current producer is safe but far too sparse to be treated as a full render mesh.

This is not an appearance bug and not a skeleton bug. It is the still-open MWB2 behavioral seam already described by the hardening ledger: the existing observed-safe relation complex is a bounded baseline, not product mesh-quality closure.

A deterministic `DirectionalCoverageMeasurement.v1` diagnostic has been added. Its default prospective quality policy is:

- source-alpha coverage `>= 0.95`;
- mesh-outside-source fraction `<= 0.01`.

The measurement is diagnostic only; source alpha is not converted into geometry authority by that probe.

## Local-convex support ceiling — why one repair class cannot solve everything

The frozen mesh contract already types `LOCAL_CONVEX_INTERPOLATION`, and the historical contract explicitly authorizes inserted vertices only when their rest position is derived from mutually compatible admitted S support.

A direct geometric ceiling audit projects **all 950 admitted S nodes** to each target view and measures the fraction of source alpha lying inside their 2D convex hull. Cardinal ceilings are:

| view | source alpha inside all-S projected convex hull |
| --- | ---: |
| V0 / S | 75.45% |
| V2 / E | 72.51% |
| V4 / N | 75.43% |
| V6 / W | 72.53% |

Across all eight views the all-S convex-support ceiling is approximately `68.59%..75.45%`.

Therefore two different gaps must not be conflated:

1. **MWB2 discretization/topology gap:** large regions inside admissible S support are currently left uncovered by the relation-clique baseline. This is the immediate blocker and should be attacked with the already-typed MWB2 local-convex/constrained-triangulation gate.
2. **Residual visual-only gap:** roughly one quarter to one third of the source silhouette lies outside any local-convex projection of the current S substrate. No legal `SurfaceSupportBinding` can place a mechanical mesh vertex there without changing S authority or introducing a separately typed visual completion/shell mechanism.

The first gap must be closed before using visual completion as an explanation for sparse rest topology.

## Immediate next gate — MWB2 local-convex / constrained triangulation

The canonical mesh contract already froze this path before implementation:

`S -> view-local constrained triangulation -> LOCAL_CONVEX_INTERPOLATION support -> QualifiedEditableMeshIR -> deterministic W interpolation -> QualifiedMeshSkinIR`

The new real failure now supplies the demonstrated need. The next candidate must remain generic and must not use Mage-specific thresholds, source mesh topology, teacher mesh, negative/extrapolating support coefficients or UNKNOWN crossing.

Only after the legal local-convex support domain is substantially discretized and deformation/provenance gates pass should residual out-of-support visual regions move to `VisualCompletionProposalIR -> QualifiedVisualCompletionIR` work.

## Later visual-completion seam

The repository already defines the completion types and qualifier but no current production completion-proposal producer was found during this audit. That is a real later gap, but it is **not** allowed to hide MWB2 rest-coverage failure.

For residual regions that are mathematically outside admissible S support, future completion classes may include deterministic cross-view visual transport first, then separately qualified learned completion only where necessary, otherwise explicit UNKNOWN/empty output. Completion must remain outside S/G/W mechanical truth.

## Real E2E trigger contract

The eventual real trigger must fail closed unless all of the following hold:

- exactly 8 admitted source observations and 8 bound cameras;
- expected S/G/W lineages match the selected witness;
- no teacher mesh/skeleton/skin is consumed by product assembly;
- no mock, synthetic, demo-tessellation or reconstructed-weight artifact can satisfy a real-artifact requirement;
- MWB2 never bridges typed UNKNOWN/UNOBSERVED geometry;
- inserted mesh vertices use legal nonnegative simplex `SurfaceSupportBinding` only;
- visual completion, if used, is separately typed/support-bound/qualified and never mutates S/G/W truth;
- per-view rig overlay uses the same qualified directional pivots as motion evaluation;
- missing deformation/motion evidence remains blocked, never synthetic PASS;
- native export remains forbidden until the exact current product has fresh PASS proof;
- every emitted artifact records source lineage/content hashes in one reproducible manifest.

## Current execution order

1. Exact S/G/W persistence — **DONE**.
2. Real current-MWB2 x8 + exact mesh-skin + observation appearance — **WIRED; REAL COVERAGE FAILURE MEASURED**.
3. Exact directional joint/view binding — **WIRED; projection quality clean**.
4. Proofless Living Compile static inspection — **WIRED; runtime remains blocked**.
5. Implement/preregister generic MWB2 local-convex/constrained-triangulation candidate and run topology/coverage/deformation gates — **NEXT BLOCKER**.
6. Classify residual outside-S visual regions and only then open qualified visual completion if still required.
7. Add qualification-owned Mage motion evidence and run product proof.
8. Export native runtime only after actual PASS.
9. Promote one fresh-process real E2E trigger after all previous gates close.

## Claim boundary

`MAGE_FIT1_RIGGING_CORE = CLOSED`

`MAGE_EXACT_S_G_W_PERSISTENCE = CLOSED`

`MAGE_DIRECTIONAL_BINDING = CLEAN`

`MAGE_CURRENT_MWB2_RELATION_COMPLEX = REAL_FAIL__INSUFFICIENT_RENDER_COVERAGE`

`MAGE_REAL_PRODUCT_E2E = OPEN`

`PRODUCT_PASS = NOT CLAIMED`
