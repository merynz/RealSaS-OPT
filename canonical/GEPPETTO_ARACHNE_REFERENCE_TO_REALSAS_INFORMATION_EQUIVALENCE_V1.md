# RealSaS — Geppetto/Arachne Reference-to-Consumer Information Equivalence V1

**Date:** 2026-08-31  
**Status:** `R5_FIELD_MATRIX_COMPLETE__E1_LARGELY_SUPPORTED__E2_FULL_VS_PARTIAL_COVERAGE_REMAINS_EMPIRICAL_GATE`

## Authority boundary

This matrix implements:

- `GEPPETTO_ARACHNE_EXTERNAL_REFERENCE_CLEANROOM_AUDIT_PREREG_20260831.md`
- `GEPPETTO_ARACHNE_CONSUMER_SUBSTRATE_EQUIVALENCE_BOUNDARY_AMENDMENT_20260831.md`

Reference inputs are compared against the complete consumer-facing RealSaS stack, not raw IRIS output.

Geppetto boundary:

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`

Arachne boundary:

`B_A = ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`

The deterministic pre-consumer geometry class is planned to be renamed from `SurfaceBuilder` to `GeometricSubstrateAssembler`; this is a naming/scope clarification, not a new geometry authority.

## Status semantics

- `EXACT_EQUIVALENT`: same required information can be exposed directly or by exact deterministic packaging.
- `STRICTLY_STRONGER_REALSAS`: RealSaS boundary provides additional qualified authority relevant to the same function.
- `APPROXIMATE_EQUIVALENT`: analogous information is derivable but fidelity/coverage is not exact.
- `MISSING`: reference receives material information unavailable to shipping RealSaS boundary.
- `NOT_REQUIRED_AT_INFERENCE`: reference training/postprocess uses the quantity but the learned product-inference function does not require it.
- `UNKNOWN_FROM_PUBLIC_RELEASE`: public evidence is insufficient.

E1 is field availability/derivability. E2 is coverage/accessibility.

---

## A. RigAnything-class skeleton function -> Geppetto + Compiler

| Reference quantity/function | Reference use | RealSaS counterpart | E1 | E2 / caveat |
|---|---|---|---|---|
| 3D surface positions `P` | primary shape condition | `RiggingSurfaceIR.SurfaceNode.P` | `EXACT_EQUIVALENT` as field type | partial observable coverage remains open |
| surface normals `N` | point condition | deterministic qualified normal/local-frame operator over admitted geometry | `APPROXIMATE_EQUIVALENT` | exact reference normals are full-mesh; current product contract does not require learned N; normal fidelity is not assumed exact |
| exactly 1024 surface samples | reference point-token count | deterministic resampling/packing adapter | `EXACT_EQUIVALENT` as packaging when enough admitted support exists | sampling from partial surface differs from complete-mesh distribution |
| sampled-AABB center | inference normalization | deterministic from B_G sample | `EXACT_EQUIVALENT` | none once sample set fixed |
| max-absolute sampled scale | inference normalization | deterministic from B_G sample | `EXACT_EQUIVALENT` | none once sample set fixed |
| polygon faces/topology for skeleton network | not direct skeleton neural input | not needed | `NOT_REQUIRED_AT_INFERENCE` | no equivalence blocker for Geppetto |
| complete-mesh surface coverage | implicit through reference surface sampling | intentionally partial observation-grounded surface | `MISSING` relative to reference richness | **primary unresolved E2 blocker; necessity must be tested, not assumed** |
| global shape context | learned point-token self-attention | independent Geppetto geometry encoder over B_G | functional obligation, not raw field | R6 ceiling required |
| variable joint count / stop | autoregressive stop | Geppetto existence/count evidence + abstention/overflow policy | functional counterpart available by design | R6 required |
| continuous conditional joint location | diffusion next-joint decoder | Geppetto position distribution/uncertainty mechanism | functional obligation | exact mechanism not frozen; R6 required |
| explicit parent evidence | candidate-parent scoring | `SkeletonProposalEdge` scores + root scores | `EXACT_EQUIVALENT` in function | final tree owned by Compiler |
| final valid root/tree | sequential reference output | Compiler global graph optimization + canonicalization | `STRICTLY_STRONGER_REALSAS` authority boundary | quality depends on Geppetto evidence; R6 required |
| canonical joint identity | reference sequence identity | Compiler-minted canonical IDs | `STRICTLY_STRONGER_REALSAS` | proposal IDs deliberately noncanonical |
| released 64-joint ceiling | implementation capacity | C0 policy 160, overflow abstains/no truncation | `STRICTLY_STRONGER_REALSAS` policy capacity | learned 160-capacity candidate still must pass ceiling |
| BFS/sibling serialization | training sequence convention | no product requirement; proposal is set/graph evidence | `NOT_REQUIRED_AT_INFERENCE` | if AR candidate is chosen, ordering must not create arbitrary-truth authority |

### Geppetto E1 conclusion

No material **field-type** blocker is found except the intentionally absent complete-mesh hidden surface. P, normalization and relational graph evidence have direct RealSaS counterparts. Normal information can be deterministically derived but is not claimed exact; current independent RealSaS normal falsification found no required explicit normal channel under the frozen downstream information test.

### Geppetto E2 conclusion

The material unresolved question is:

> Does partial observation-grounded geometry retain enough global shape information for a reference-class skeleton proposal when the existing Compiler performs final graph optimization?

This is an empirical coverage/accessibility question and cannot be closed by code inspection.

---

## B. SkinTokens-class skin-field function -> Arachne + Compiler

| Reference quantity/function | Reference use | RealSaS counterpart | E1 | E2 / caveat |
|---|---|---|---|---|
| sampled surface `P` | geometry condition/query | `RiggingSurfaceIR.P` via deterministic adapter | `EXACT_EQUIVALENT` as field type | partial coverage remains open |
| sampled surface `N` | geometry condition/query | deterministic qualified normal/local frame | `APPROXIMATE_EQUIVALENT` | reference has full-mesh normals; exact fidelity not assumed |
| skeleton joint positions | skeleton/skin alignment | `QualifiedSkeletonIR` positions | `STRICTLY_STRONGER_REALSAS` authority | joint-position quality still matters and is tested downstream |
| skeleton parent structure | token alignment / bone identity | Compiler-qualified parents/root | `STRICTLY_STRONGER_REALSAS` authority | deterministic conditioning serialization required |
| stable bone/joint ordering for skin fields | sequence alignment | deterministic serialization of canonical joint IDs | `EXACT_EQUIVALENT` packaging | serialization is adapter, not authority |
| GT per-bone dense skin samples | codec training target construction | authoritative dense teacher weight field can construct active-region samples offline | `NOT_REQUIRED_AT_INFERENCE` | teacher-to-admitted-surface alignment must remain exact/frozen |
| full mesh faces for active-region sampling | training sampler support | teacher corpus geometry may be used only for offline target construction | `NOT_REQUIRED_AT_INFERENCE` | no product-input equivalence claim derives from teacher mesh |
| compact per-bone field representation | learned FSQ-CVAE | independent Arachne SkinFieldCodec candidate | functional obligation | codec ceiling must pass before predictor training |
| geometry-conditioned decode | learned decoder | independent Arachne field decoder queried on admitted surface | functional obligation | R6 codec and predictor ceilings required |
| unified skeleton+skin AR sequence | TokenRig architecture | intentionally split: Compiler-qualified skeleton -> Arachne | `NOT_REQUIRED_AT_INFERENCE` as authority topology | RealSaS must match skin function, not upstream topology |
| sampled->original mesh NN transfer | export helper | not required when prediction domain is `RiggingSurfaceIR` nodes; later runtime transfer is separate | `NOT_REQUIRED_AT_INFERENCE` for core skin field | export/runtime mapping separately qualified |
| complete-mesh surface coverage | reference geometry richness | intentionally partial observation-grounded surface | `MISSING` relative to reference richness | **primary unresolved E2 blocker** |
| legal simplex/reference constraints | partly model/output convention | Compiler `qualify_skin` binding/legal/simplex/bounded repair | `STRICTLY_STRONGER_REALSAS` authority | predictor must be within repair budget |
| deformation quality | downstream objective/reward evidence | exact RealSaS deformation/motion proof route | `STRICTLY_STRONGER_REALSAS` qualification route | must pass product-specific thresholds |

### Arachne E1 conclusion

The Arachne boundary is particularly well aligned with SkinTokens. Product inference needs geometry and a skeleton; RealSaS supplies both, and the skeleton is already Compiler-qualified. GT-skin-dependent dense samples are training-target machinery, not missing inference information.

### Arachne E2 conclusion

Again, the unresolved difference is full-mesh versus partial surface coverage. This may be less severe for skinning than for skeleton discovery because Arachne predicts only on **admitted** RealSaS surface nodes and is not required to invent weights for unseen/unadmitted geometry. Nevertheless, generalization and deformation quality on partial geometry require an oracle-substrate ceiling.

---

## C. Existing RealSaS deterministic authority must be credited, not duplicated

### GeometricSubstrateAssembler scope

Current `compiler/realsas_compiler_core/surface.py` already has a narrow observation-grounded function:

- analytic `P = O + dF`;
- fuse explicitly admitted persistence groups;
- preserve support/provenance/raster bindings/lineage;
- no teacher/source-rig identity;
- no normal authority by default.

`RiggingSurfaceIR.local_relations` may contain **local geometric relations only** when a separately qualified deterministic operator defines them. It is not a skeleton graph.

### Compiler skeleton scope

Current `qualify_skeleton`:

- verifies surface lineage/support;
- converts model proposal joints/edges to global graph candidates;
- invokes `optimize_canonical_graph_v18_98`;
- selects qualified root/parents;
- rejects failed graph optimization;
- mints new canonical joint IDs.

This is substantial historical synthesis/solver authority retained from the earlier model-free Compiler program.

### Compiler skin scope

Current `qualify_skin` owns:

- surface/skeleton lineage binding;
- legal reference validation;
- finite/nonnegative checks;
- optional bounded top-k sparsification;
- bounded simplex repair;
- fail-closed rejection outside correction budget.

### Binding non-overlap rule

`GeometricSubstrateAssembler` must not own rig root/parent/tree decisions.  
Geppetto must not duplicate final canonical graph optimization.  
Arachne must not own final simplex/legal-reference authority.  
Compiler must not be reduced to a passive checker merely because external references place more logic inside their neural model.

---

## R5 verdict

### Supported now

- RigAnything provides a valid external **skeleton solution-class reference** under complete 3D P+N surface evidence.
- SkinTokens provides the cleaner primary external **skin-field representation/solution-class reference**.
- At the RealSaS consumer boundaries, the required geometric **field classes** are largely present or deterministically derivable.
- The existing Compiler already provides substantial global graph and skin qualification functions that external models may internalize.

### Not yet proven

- shipping partial `RiggingSurfaceIR` is coverage-equivalent to a complete mesh surface;
- Geppetto can reach product tolerance on exact observation-limited B_G;
- a chosen Arachne SkinFieldCodec can reconstruct RealSaS teacher skin below product/deformation thresholds;
- predicted IRIS geometry is accurate enough for those final consumers.

### Single material equivalence blocker

`FULL_MESH_SURFACE_COVERAGE -> PARTIAL_OBSERVATION_GROUNDED_SURFACE`

This is **not automatically a fatal missing field**. It is the R6 causal question: if the observation-limited oracle substrate reaches the required consumer ceiling, complete hidden surface is proven unnecessary for the admitted RealSaS product task.

Scientific learned-model training authority remains unchanged.
