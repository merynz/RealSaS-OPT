# RealSaS — Geppetto/Arachne External Reference Clean-Room Audit Closure — 2026-08-31

**Status:** `R0_R5_CLOSED__R6_ORACLE_PROTOCOL_FROZEN__REFERENCE_SOLUTION_CLASSES_SUPPORTED__INPUT_EQUIVALENCE_AWAITS_ORACLE_CEILING`

## Closure statement

The code-level external-reference audit required by `GEPPETTO_ARACHNE_EXTERNAL_REFERENCE_CLEANROOM_AUDIT_PREREG_20260831.md` is complete through R5.

This closure does **not** claim that Geppetto/Arachne are already product-solved. It closes the upstream/code/information audit and reduces the remaining solvability question to explicit oracle-substrate experiments.

Authorities:

- `RIGANYTHING_CODE_LEVEL_REFERENCE_AUDIT_V1.md`
- `SKINTOKENS_CODE_LEVEL_REFERENCE_AUDIT_V1.md`
- `GEPPETTO_ARACHNE_REFERENCE_TO_REALSAS_INFORMATION_EQUIVALENCE_V1.md`
- `GEPPETTO_ARACHNE_CONSUMER_SUBSTRATE_EQUIVALENCE_BOUNDARY_AMENDMENT_20260831.md`

Frozen upstream snapshots:

- RigAnything `d03cdb21dd134fa81df6b0947522469db3f78bd2`
- SkinTokens `273b691d35989d71cd17ff2895fdc735097b92d1`

## 1. Clean-room / license result

### RigAnything

The Adobe Research License permits noncommercial research only and explicitly excludes commercial product development. Therefore RealSaS will not copy, translate, import, depend on, or productize the released RigAnything source. Its role is limited to a frozen research reference for observed functional behavior, information contracts and independently stated algorithmic requirements.

### SkinTokens

The frozen SkinTokens snapshot is MIT licensed. It may legally be used more permissively, but RealSaS still keeps a provenance boundary and does not make upstream architectural choices binding merely because they are available.

No audit result removes possible patent/other-IP review requirements; this is a source-license and scientific-provenance firewall, not legal advice.

## 2. Main scientific result

The earlier research premise is supported in a narrower and now well-defined form:

> Skeleton proposal and skin-field prediction have demonstrated external solution classes when sufficiently rich 3D surface evidence is available.

For RealSaS, the correct equivalence object is not raw IRIS output. It is the **consumer-facing deterministic substrate** after learned observation geometry has been assembled.

### Geppetto

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`

### Arachne

`B_A = ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`

At these boundaries, most material reference field types are directly available or deterministically derivable. The one unresolved material richness difference is:

`complete hidden/full mesh surface` versus `partial observation-grounded admitted surface`.

That difference is now an empirical R6 coverage/accessibility question, not a vague architecture concern.

## 3. Reference allocation decision

### Primary Geppetto reference: RigAnything skeleton function

What is retained as functional obligation:

- global surface-shape conditioning;
- template-free variable joint/control cardinality;
- continuous conditional joint location evidence;
- explicit relational parent/root evidence;
- endogenous count/stop/unsupported behavior;
- enough evidence to recover a valid canonical skeleton under downstream qualification.

What is **not** adopted as mandatory mechanism:

- exact BFS serialization;
- diffusion specifically;
- transformer width/layer count;
- released 64-joint limit;
- reference ownership of the final parent tree.

RealSaS deliberately splits the function:

`Geppetto learned evidence -> SkeletonProposalIR -> existing Compiler global graph authority -> QualifiedSkeletonIR`.

### Primary Arachne reference: SkinTokens skin-field representation

What is retained as functional obligation:

- geometry-conditioned per-joint influence fields;
- compact representation with a measured reconstruction ceiling;
- explicit skeleton-conditioned alignment;
- active/sparse influence regions receiving sufficient supervision;
- decode on all admitted surface nodes;
- deformation-sensitive qualification after static prediction.

What is **not** adopted as mandatory mechanism:

- FSQ specifically;
- exact latent/token count;
- Qwen or any specific LM;
- unified skeleton+skin autoregression;
- exact public-paper RL recipe.

RealSaS deliberately splits the function:

`QualifiedSkeletonIR + RiggingSurfaceIR -> Arachne field proposal -> SkinProposalIR -> existing Compiler skin authority -> QualifiedSkinIR -> deformation proof`.

### RigAnything skin path: secondary only

RigAnything's raw learned skin head is useful evidence, but its released final skin output materially uses mesh-neighbor smoothing, thresholding/sparsification and nearest-neighbor transfer. Because those depend on a complete mesh/topology route and blur learned-vs-postprocessed responsibility, it is not the primary clean Arachne reference.

## 4. RealSaS lower-stack ownership correction

Historical context matters: RealSaS first attempted generic rigging predominantly through a model-free Compiler. Generic inference was too difficult without learned evidence, but that work left substantial solver/qualification machinery that must be preserved.

### Planned deterministic class name

The old class/function family named `SurfaceBuilder` is henceforth the **planned nomenclature**:

`GeometricSubstrateAssembler`

The code rename itself is deferred to one atomic compatibility migration; historical artifact names/hashes are not rewritten.

### GeometricSubstrateAssembler scope

Allowed responsibilities:

- analytic `P = O + dF` from admitted observation evidence;
- persistence fusion/canonical observation grouping;
- support/provenance/raster bindings;
- geometry lineage;
- independently qualified local geometric operators such as normals/neighborhood/sheet evidence when causally justified.

Forbidden responsibilities:

- skeleton root/parent/tree authority;
- canonical joint IDs;
- mechanical rig graph optimization;
- skin simplex/legal-reference authority;
- hidden surface completion.

`RiggingSurfaceIR.local_relations` means **local geometric relations**, not skeleton edges.

### Existing Compiler skeleton authority

The current `qualify_skeleton` is not a passive schema checker. It translates proposal joints/edges into global graph candidates, invokes `optimize_canonical_graph_v18_98`, admits/rejects the graph, selects root/parents and mints new canonical joint IDs.

Therefore Geppetto is an evidence proposer, not a second canonical skeleton Compiler.

### Existing Compiler skin authority

The current `qualify_skin` owns surface/skeleton lineage binding, legal reference validation, finite/nonnegative checks, optional bounded sparsification and bounded simplex repair with fail-closed rejection.

Therefore Arachne is an influence-field proposer, not a second skin-law authority.

## 5. Upstream training visibility result

The audit separates functional solvability from exact recipe recovery.

### RigAnything

Inference/architecture/preprocessing are code-verifiable. The frozen public repository does not expose a complete authoritative published-run training pipeline. Exact unpublished training integration is `NOT_PUBLICLY_VERIFIABLE`.

### SkinTokens

The core codec/inference factorization is code-verifiable. In the frozen snapshot, wrapper-level `training_step` / loss integration is incomplete or `NotImplemented` in material released classes. Exact published-run training integration is therefore partly `NOT_PUBLICLY_VERIFIABLE` from code.

This means RealSaS cannot and need not reproduce an unknowable exact recipe. The proper standard is independent functional implementation + sealed oracle ceilings.

## 6. R6 — frozen oracle-substrate protocol

R6 is the remaining empirical gate before final learned apparatus seals. It must distinguish **reference richness**, **shipping-observable information**, and **consumer apparatus capacity**.

### Common arms

#### `U0_REFERENCE_FULL_SURFACE`

A separately labeled diagnostic upper bound using authoritative full-surface geometry transformed only into the independently specified consumer conditioning format.

Purpose: establish whether the candidate consumer apparatus can solve the task when given reference-class rich geometry.

This arm **cannot** prove RealSaS input equivalence.

#### `U1_OBSERVATION_ORACLE_SUBSTRATE`

Authoritative corpus geometry is rendered/restricted to exactly the observations available to shipping RealSaS, then passed through the same deterministic product route:

`exact observation evidence -> analytic reconstruction -> GeometricSubstrateAssembler -> B_G/B_A`.

No hidden back-side/full-mesh completion reaches the consumer.

Purpose: test whether shipping-observable information is sufficient before IRIS prediction error.

#### `U2_PREDICTED_IRIS_SUBSTRATE`

Only after U1 and the current IRIS qualification gates pass:

`predicted IRIS evidence -> same GeometricSubstrateAssembler -> same B_G/B_A`.

Purpose: measure the prediction/accessibility gap after information sufficiency is established.

### R6-G — Geppetto oracle ceiling

Sequence:

1. freeze independent `GeppettoConditioningAdapter` and candidate model before outputs;
2. one-family U0 overfit/ceiling;
3. one-family U1 observation-oracle ceiling;
4. sealed heterogeneous small-family U1 ceiling;
5. every prediction goes through the **existing Compiler skeleton qualifier/global graph optimizer**;
6. evaluate joint position, count/existence, root/edge evidence, qualified graph success and downstream deformation/mechanical consumer evidence where applicable;
7. only after U1 passes may predicted IRIS U2 be interpreted.

Primary causal interpretation:

- U0 fail -> Geppetto apparatus/representation is inadequate;
- U0 pass + U1 fail -> observation-coverage/substrate insufficiency is demonstrated;
- U1 pass -> complete hidden surface is **not necessary** for the admitted Geppetto task on that gate;
- U1 pass + U2 fail -> IRIS prediction/accessibility is the remaining bottleneck.

No full training from a failed small ceiling.

### R6-A0 — Arachne codec ceiling FIRST

Before any Arachne predictor:

1. take authoritative dense eligible skin truth aligned to the admitted surface/skeleton semantics;
2. encode/decode with the independently designed SkinFieldCodec candidate;
3. evaluate reconstructed weights through the existing Compiler skin qualifier;
4. evaluate static error **and deformation-sensitive proof**;
5. test one-family then heterogeneous small-family ceiling.

Primary interpretation:

- codec fail -> predictor training forbidden; representation itself is insufficient;
- codec pass -> predictor apparatus may be sealed.

### R6-A1 — Arachne predictor ceiling

After codec pass:

1. freeze `ArachneConditioningAdapter`, skeleton serialization and predictor before outputs;
2. compare U0 full-surface diagnostic and U1 observation-oracle substrate;
3. use Compiler-qualified skeleton inputs, not source-rig hidden identity;
4. emit `SkinProposalIR` only;
5. run existing Compiler skin qualification;
6. run deformation/motion proof;
7. predicted IRIS U2 only after U1 passes.

Causal interpretation mirrors Geppetto:

- U0 fail -> Arachne apparatus fail;
- U0 pass + U1 fail -> partial coverage/substrate insufficiency;
- U1 pass -> complete hidden surface unnecessary for admitted Arachne domain;
- U1 pass + U2 fail -> upstream geometry prediction/accessibility bottleneck.

## 7. Change-control / seal effect

R0-R5 are now closed. They may be reopened only by:

- a concrete upstream source contradiction;
- a newly released authoritative implementation that resolves a material `NOT_PUBLICLY_VERIFIABLE` field;
- a causal R6 failure showing a missing functional obligation or incorrect equivalence classification;
- an explicit product-contract revision.

They may not be reopened because a later architecture scores poorly and a more convenient interpretation is desired.

R6 protocols above are frozen at the causal-arm level. Exact learned architecture/loss/optimizer remain intentionally unsealed until:

- clean-C0 semantic membership closes;
- the current IRIS/corpus geometry gates needed for the chosen substrate close;
- final independent candidate designs are written before optimizer step 1.

## 8. Final audit verdict

### Supported

`REFERENCE_SOLUTION_CLASS_GEPPETTO = SUPPORTED`

`REFERENCE_SOLUTION_CLASS_ARACHNE = SUPPORTED`

`REFERENCE_FIELD_CLASS_EQUIVALENCE_AT_B_G_B_A = LARGELY_SUPPORTED`

`EXISTING_COMPILER_SOLVER_AUTHORITY = MUST_BE_PRESERVED`

`SKINTOKENS_AS_PRIMARY_ARACHNE_CODEC_REFERENCE = SUPPORTED`

`RIGANYTHING_SKIN_AS_PRIMARY_ARACHNE_REFERENCE = REJECTED`

### Still empirical / not yet claimed

`PARTIAL_OBSERVED_SURFACE_SUFFICIENCY = UNKNOWN_UNTIL_R6_U1`

`GEPPETTO_PRODUCT_SOLVED = FALSE_NOT_YET_TESTED`

`ARACHNE_PRODUCT_SOLVED = FALSE_NOT_YET_TESTED`

`PREDICTED_IRIS_CONSUMER_SUFFICIENCY = UNKNOWN`

### Current blocker before final learned seals

The external-reference audit itself is no longer open. The next scientific blocker is the **oracle observation-limited consumer ceiling**, after the prerequisite clean-corpus/substrate gates are closed.

No Geppetto/Arachne full training is authorized by this closure.
