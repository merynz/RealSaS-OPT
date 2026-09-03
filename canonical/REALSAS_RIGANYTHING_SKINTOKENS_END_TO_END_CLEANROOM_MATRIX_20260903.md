# RealSaS — RigAnything / SkinTokens End-to-End Cleanroom Architecture Matrix

**Date:** 2026-09-03  
**Status:** `CLEANROOM_CODE_MATRIX_COMPLETE__NO_ARCHITECTURE_REPAIR_AUTHORIZED__SHIPPING_CODEC_AND_OBSERVATION_COVERAGE_GATES_REQUIRED`  
**Record class:** side-branch hardening evidence; this does not supersede `CURRENT_STATE.md` on `main` until explicitly promoted.  

## Pinned code bases

- RealSaS: `merynz/RealSaS-OPT`, hardening branch `behavioral/geppetto-v2-integrity-v1-20260903`; comparison base `7f39a846ad05f91560836026e5ef6dfbc74dc731`.
- RigAnything: `Isabella98Liu/RigAnything@d03cdb21dd134fa81df6b0947522469db3f78bd2`.
- SkinTokens: `VAST-AI-Research/SkinTokens@273b691d35989d71cd17ff2895fdc735097b92d1`.

This report extends, rather than erases, `GEPPETTO_ARACHNE_REFERENCE_TO_REALSAS_INFORMATION_EQUIVALENCE_V1.md`.

## 1. Comparison law

No learner is compared against an external system's final product in isolation.

The comparison unit is the **complete capability boundary**:

```text
reference learned function + reference deterministic postprocess
                versus
RealSaS learned proposal + typed IR + deterministic Compiler/qualification/proof
```

Therefore:

```text
RigAnything skeleton capability
  ~= Geppetto + SkeletonProposalIR + Compiler graph optimization + canonicalization

reference skinning capability
  ~= Arachne + SkinFieldCodec + SkinProposalIR + Compiler skin qualification
     + mesh/weight binding + verified deformation/proof
```

RealSaS additionally owns an observation-to-mechanical-substrate stage absent from the 3D references:

```text
8 RGB views + exact cameras
 -> IRIS
 -> ObservationEvidenceIR
 -> persistence / GeometricSubstrateAssembler
 -> analytic P + qualified local geometry/N
 -> RiggingSurfaceIR S_hat
```

Binding non-goal: `3D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`.

## 2. Whole-system topology

### RigAnything

```text
closed mesh
 -> whole-surface sampling: 1024 P + face N
 -> point tokenizer / transformer
 -> autoregressive joint token
 -> diffusion joint XYZ
 -> parent scoring / self-parent STOP
 -> generated skeleton
 -> ALL mesh vertices + vertex N re-tokenized
 -> point-token x joint-token skinning_mlp
 -> raw W scores
 -> top-5 + threshold + renormalize
 -> mesh-neighbor smoothing
 -> projection to original GLB vertices
 -> rigged mesh export
```

### SkinTokens

```text
closed mesh
 -> sampled P + N
 -> mesh encoder / skeleton tokenizer
 -> autoregressive skeleton tokens
 -> deterministic detokenization / skeleton construction
 -> dedicated pretrained SkinVAE/FSQ skin representation
 -> skin tokens generated after skeleton tokens
 -> geometry-conditioned SkinVAE dense decode
 -> sampled W
 -> kNN/inverse-distance transfer to full mesh
 -> optional voxel/geodesic skin postprocess
 -> rigged asset
```

### RealSaS

```text
8 RGB views + exact cameras
 -> IRIS depth/support/uncertainty
 -> analytic reprojection + ObservationEvidenceIR
 -> persistence + support firewall
 -> analytic P = O + dF
 -> deterministic DTB-ND1 N/local geometry
 -> RiggingSurfaceIR S_hat
 -> Geppetto
 -> SkeletonProposalIR G*
 -> Compiler global graph optimizer
 -> QualifiedSkeletonIR G
 -> Arachne segment-aware conditioning
 -> per-joint codec latents
 -> SkinFieldCodec dense W
 -> SkinProposalIR W*
 -> Compiler.qualify_skin
 -> QualifiedSkinIR W
 -> qualified mesh-weight binding
 -> verified LBS + product proof
 -> exact-8 directional 2D/2.5D runtime product
```

## 3. Input-information provenance matrix

| Capability / information | RigAnything | SkinTokens | RealSaS equivalent | Verdict |
|---|---|---|---|---|
| Surface position `P` | Entire closed mesh surface; 1024 area/even samples for skeleton; all mesh vertices available for final skinning | Full mesh surface samples | `P=O+dF` from admitted multi-view evidence | `FIELD_EQUIVALENT`; coverage not yet equivalent |
| Surface normal `N` | Face normal at sampled skeleton point; vertex normal for full skinning domain | Sampled/vertex normals from mesh | Hash-bound deterministic DTB-ND1 local plane normal on admitted S | `APPROXIMATE_EQUIVALENT`; fidelity empirical |
| Surface coverage | Visibility-independent closed surface | Visibility-independent closed surface | Only observation-grounded supported surface from exact 8 views | `REFERENCE_INFORMATION_ADVANTAGE`; primary E2 gate |
| Mesh face connectivity into neural skeleton input | Not directly supplied as adjacency; used to define sampling surface/normals | Not direct transformer input | Not required | `NOT_REQUIRED_AS_NEURAL_FIELD` |
| Topology in deterministic geometry/postprocess | Mesh neighbors used heavily for RigAnything smoothing | Faces used for sampling; optional voxel/geodesic postprocess | Local observed relations + later direction-local mesh topology | `FUNCTIONAL_COUNTERPART`; not identical authority |
| Exact view/camera provenance | None needed; input already 3D | None needed; input already 3D | Explicit exact cameras, support views, raster bindings, lineage | `STRICTER_REALSAS_EPISTEMICS` |
| Hidden/backside surface | Available from closed mesh | Available from closed mesh | Available only if observed in one/more of 8 views; otherwise absent from mechanical truth | `INTENTIONAL_NON_GOAL`; necessity empirical |
| Uncertainty/support | Not explicit in mesh P+N input | Not explicit in mesh P+N input | IRIS support + uncertainty + persistence diagnostics | `STRONGER_REALSAS_EVIDENCE_TYPE` |

### Critical interpretation

The reference advantage is not simply "they have XYZ+N". It is:

```text
XYZ+N sampled from a complete visibility-independent surface authority.
```

RealSaS must not reconstruct that closed surface merely to imitate the reference. The correct requirement is functional sufficiency:

```text
f_rig(S_observation_oracle) ~= f_rig(S_full_surface)
```

If this holds, hidden full-mesh surface is unnecessary for the RealSaS product. If it fails, the missing *mechanical information class* must be identified; hidden 3D reconstruction is not automatically authorized.

## 4. Observation/substrate construction: function-by-function

| Function | RigAnything / SkinTokens | RealSaS implementation | Verdict |
|---|---|---|---|
| Define candidate surface | Closed triangle mesh faces | Production camera-only full-frame Q lattice; no alpha/mask/teacher pruning | `DIFFERENT_BY_DESIGN` |
| Sample evidence locations | Mesh-surface sampling | Camera rays / depth hypotheses | `OBSERVATION_EQUIVALENT_CLASS` |
| Estimate position | Exact mesh surface sample | Learned depth, then analytic exact-camera `P=O+dF` | `FUNCTIONAL_EQUIVALENT`; prediction error empirical |
| Multi-view consistency | Not needed; mesh is already common 3D | Analytic reprojection + persistence grouping | `REALSAS_ONLY_REQUIRED_FUNCTION` |
| Admit/reject surface evidence | Mesh validity/preprocess | support=True-only firewall + persistence consistency | `STRONGER_REALSAS_EPISTEMICS` |
| Produce normal | Mesh face/vertex normal | deterministic DTB-ND1 robust local plane | `APPROXIMATE_EQUIVALENT` |
| Local continuity | Implicit mesh geometry/connectivity | observed raster neighborhood + common support + robust world-distance local relations | `FUNCTIONAL_COUNTERPART`; coverage empirical |
| Preserve origin/provenance | Mostly asset identity | explicit observation IDs/raster/view/hash lineage | `STRICTER_REALSAS` |

### IRIS + Assembler verdict

**Architecturally present:** yes. The correct equivalent of reference mesh preprocessing is not a reconstructed mesh, but `RiggingSurfaceIR S_hat` carrying admitted `P`, qualified `N`, support, local relations, raster/view provenance and uncertainty.

**Not proven:** complete-surface information sufficiency. This remains a mandatory oracle-substrate ceiling, not a code-presence question.

## 5. Skeleton generation matrix

| Function | RigAnything | SkinTokens | RealSaS | Verdict |
|---|---|---|---|---|
| Surface encoder | 6D P+N -> 1024-d point tokens; 12-layer transformer context | Mesh encoder P+N; skeleton token stream | Geppetto 24D S features; local geometry attention + global transformer | `FUNCTIONAL_EQUIVALENT` |
| Global shape context | full-surface point self-attention | mesh-condition tokens | global transformer over admitted S | `EQUIVALENT_FUNCTION`; coverage is the difference |
| Joint cardinality | AR generation with max-joint ceiling and self-parent STOP | grammar/token sequence | dynamic resource-bounded Geppetto STOP; no canonical fixed product slot identity | `EQUIVALENT_FUNCTION` |
| Joint location | diffusion sample from next joint token | quantized/tokenized skeleton representation | 3-mode continuous loci + uncertainty | `EQUIVALENT_FUNCTION`, different parameterization |
| Parent evidence | learned parent candidate scores; sequence participates | skeleton token parent semantics + detokenization | all-pairs soft parent evidence + root evidence | `EQUIVALENT_EVIDENCE` |
| Final root/tree | largely determined by generation sequence / predicted parent | deterministic tokenizer/detokenizer grammar completes skeleton | global Compiler graph optimizer selects legal root/tree and mints canonical IDs | `STRONGER_REALSAS_AUTHORITY` |
| Teacher topology identity | reference training serialization | token order/parents | not product objective; final mechanical equivalence is authority | `INTENTIONAL_DIFFERENCE` |
| Support binding | implicit point context | mesh condition | each Geppetto joint carries surface support IDs; unsupported joints rejected by proof | `STRONGER_REALSAS_TRACEABILITY` |

### Skeleton verdict

Geppetto alone is not RigAnything-equivalent; **Geppetto + Compiler** is the matched boundary. That boundary is architecturally coherent and current generic behavioral tests are closed. The remaining upstream risk is S coverage/quality, not missing skeleton authority.

## 6. Skinning: how skeleton information is consumed

| Function | RigAnything | SkinTokens | RealSaS | Verdict |
|---|---|---|---|---|
| Geometry domain for W | Full mesh vertices + normals at inference | sampled mesh P+N; transfer to original mesh | admitted S nodes then qualified direction-local mesh bindings | `DIFFERENT_PRODUCT_DOMAIN` |
| Skeleton conditioning | generated joint token paired with every point token | skeleton tokens precede skin tokens; number/order/parents constrain sequence | Compiler-qualified G; Arachne joint features + explicit parent-segment geometry | `STRONG_FUNCTIONAL_EQUIVALENT` |
| Explicit point-joint relation | learned implicitly from point/joint token pair | geometry-conditioned VAE/mesh encoding; representation learned separately | explicit 10D contract: dXYZ, joint distance, parent-segment distance, t, length, normal-axis relation, parent flag | `STRONGER_EXPLICIT_MECHANICAL_CONDITIONING` |
| Skin representation | direct scalar skinning MLP | dedicated FSQ-CVAE / discrete skin tokens | per-joint latent field + shared SkinFieldCodec | `HYBRID`; capacity must be proven |
| Dedicated skin pretraining | no separate codec | yes; major architectural emphasis | yes, A0 Codec ceiling intended | `FUNCTIONAL_EQUIVALENT_INTENT` |
| Skin-aware training samples | regular/full geometry training paths | per-bone active-region dense skin sampling | teacher dense W can supervise admitted S; current synthetic ceiling does not reproduce SkinTokens' full training richness | `REFERENCE_TRAINING_ADVANTAGE / TEST_REQUIRED` |
| Dense decode | point-joint MLP over full vertices | high-capacity attention decoder, geometry-conditioned | Codec MLP over surface feature + joint feature + joint latent | `UNPROVEN_CAPACITY` |
| Predictor-to-codec interface | no codec bottleneck; direct point/joint tokens | LLM predicts pretrained discrete field tokens | Arachne compresses pairwise mechanics into per-joint latent, then Codec expands | `P0_INFORMATION_BOTTLENECK_RISK` |

### Arachne/Codec verdict

This is the primary architectural uncertainty discovered by the cleanroom audit.

Arachne itself consumes strong topology-aware mechanics. However, the final Codec decoder sees `surface_embed + joint_embed + per-joint latent`; the rich pairwise geometry is not directly present at the final decode boundary. This split may be sufficient, but hybridization is **not** evidence of sufficiency.

Current small synthetic panel cannot decide this because it instantiates a tiny `32 hidden / 8 latent / 2-layer` Codec, while shipping/default is `192 hidden / 64 latent / 3 encoder + 3 decoder layers`. Therefore the historical tiny sharp failure is a valid failure of that test model, **not a product architecture capacity failure**.

Status: `SHIPPING_CODEC_CAPACITY_AND_CROSS_BOUNDARY_INFORMATION_PRESERVATION_UNPROVEN`.

## 7. Deterministic skin postprocess / qualification

| Function | RigAnything | SkinTokens | RealSaS | Verdict |
|---|---|---|---|---|
| Sparsification | top-5 unconditionally | representation-dependent; later asset transfer | optional deterministic top-k only if discarded mass stays inside bounded repair budget | `SAFER_REALSAS_POLICY` |
| Thresholding | weights < 0.068 -> 0 | not equivalent in core decode | no silent semantic clipping | `STRICTER_REALSAS` |
| Normalize simplex | post-softmax renormalize | decoded field/asset normalization | Compiler finite/nonnegative/simplex validation + bounded repair | `STRONGER_REALSAS_AUTHORITY` |
| Mesh-neighbor smoothing | 10 iterations, neighbor factor .35 | optional topology/voxel/geodesic postprocess | no hidden smoothing in core; semantic predictor must be within repair budget | `DIFFERENT_POLICY` |
| Sampled -> full mesh transfer | full-vertex model path then GLB nearest projection | pinned commit uses k=8 inverse-distance interpolation from sampled W to full vertices | typed support/mesh binding; direction-local mesh must bind to admitted S | `FUNCTIONAL_COUNTERPART`; product domain differs |
| Illegal joint/surface refs | heuristic pipeline assumptions | asset/token invariants | fail-closed lineage/reference validation | `STRONGER_REALSAS` |

Compiler advantage is real but bounded: it can reject/normalize small legal deviations; it cannot transform a semantically wrong learned W into the correct W.

## 8. Deformation and product verification

| Capability | RigAnything | SkinTokens | RealSaS | Verdict |
|---|---|---|---|---|
| LBS-consumable final W | yes | yes | yes after QualifiedSkin + QualifiedMeshSkin | `EQUIVALENT_PRODUCT_FUNCTION` |
| Explicit deformation verification before product acceptance | not an equivalent fail-closed product authority in inspected release | training/reward/visual evaluation; optional postprocess | verified LBS RMS/p95 bound to exact product hash | `STRONGER_REALSAS` |
| Mechanical structure proof | implicit/benchmark | grammar/model | Compiler + proof domain | `STRONGER_REALSAS` |
| Stale lineage rejection | not comparable | not comparable | exact S/G/W/M/B hashes, stale proof rejection | `STRONGER_REALSAS` |
| Runtime release gate | export generated asset | export generated asset | runtime projection only from exact product + PASS proof bundle | `STRONGER_REALSAS` |
| Product representation | full 3D rigged mesh | full 3D rigged mesh | exact-eight directional editable 2D/2.5D puppet | `INTENTIONAL_NON_EQUIVALENCE` |

## 9. RigAnything vs SkinTokens — what can actually be concluded

### Code-supported specialization

**RigAnything is skeleton-generation-specialized:**
- entire model is organized around autoregressive template-free joint generation;
- next joint uses continuous diffusion XYZ;
- parent candidates and stop are generated alongside the sequence;
- skinning is comparatively direct: point token + generated joint token -> scalar MLP, followed by substantial deterministic mesh smoothing.

**SkinTokens is skin-representation-specialized:**
- skin field receives a dedicated high-capacity FSQ-CVAE;
- skin representation is learned independently before TokenRig predicts it;
- training uses skin-aware dense sampling around active bone regions;
- TokenRig predicts discrete skin tokens after skeleton tokens;
- frozen SkinVAE performs geometry-conditioned dense decode;
- optional voxel/geodesic postprocess provides a structural prior.

### Empirical head-to-head

`RIGANYTHING_RIGGING_BETTER_THAN_SKINTOKENS` = `UNKNOWN_FROM_INSPECTED_COMMON_BENCHMARK`.

`SKINTOKENS_SKINNING_BETTER_THAN_RIGANYTHING` = `PLAUSIBLE_AND_ARCHITECTURALLY_SUPPORTED_AS_SPECIALIZATION`, but not promoted to a direct empirical head-to-head claim without a common benchmark/table.

Do not use README self-reported comparisons against different baseline sets as a transitive RigAnything-vs-SkinTokens ranking.

## 10. RealSaS strength / risk matrix

| Area | Status | Reason |
|---|---|---|
| Observation provenance / truth firewall | `STRONG` | exact cameras, support, raster provenance, no shipping source mesh |
| 8-view -> mechanical P | `ARCHITECTURALLY_COMPLETE` | learned depth + analytic P + persistence |
| Local N/geometry | `ARCHITECTURALLY_COMPLETE` | qualified deterministic DTB-ND1 |
| Full-surface information equivalence | `UNPROVEN_P0` | reference has closed visibility-independent coverage |
| Skeleton proposal | `STRONG` | local/global geometry model, dynamic controls, uncertainty |
| Final skeleton authority | `STRONG` | Compiler global graph optimizer + canonicalization |
| Skeleton behavioral closure | `PASS` | generic synthetic final-G gate already closed |
| Arachne mechanical conditioning | `STRONG_ON_PAPER/CODE` | explicit segment/topology geometry |
| Skin field representation | `UNPROVEN_P0` | Arachne->per-joint latent->Codec split may bottleneck pairwise mechanics |
| Current tiny Codec smoke | `NON_AUTHORITATIVE_FOR_PRODUCT_CAPACITY` | test is 32/8/2; shipping is 192/64/3 |
| Skin legal/simplex authority | `STRONG` | fail-closed bounded Compiler repair |
| Deterministic semantic smoothing prior | `ABSENT_BY_POLICY` | safer than hidden heuristic repair, but predictor must meet stronger obligation |
| Mesh/weight lineage | `STRONG` | typed S/G/W/M/B binding |
| Deformation proof | `STRONG` | actual verified LBS metrics bound to product hash |
| Full product E2E | `UNPROVEN` | no real clean family end-to-end PASS yet |

## 11. Mandatory tests before architecture refreeze

No architecture patch is authorized merely because a reference does something differently. The cleanroom audit authorizes tests first.

### P0 — shipping SkinFieldCodec capacity smoke

Repeat the generic A0 representation ceiling with **actual shipping/default Codec dimensions and layers**. The previous tiny 32/8/2 smoke may remain as a cheap CI sentinel but may not decide product capacity.

Required output:
- exact teacher W reconstruction metrics;
- verified deformation RMS/p95;
- sustained checks;
- deterministic replay;
- no family-specific thresholds.

### P0 — information sufficiency ladder

For the same generic mechanical task, compare:

```text
S_full_surface_oracle
S_observation_oracle
S_predicted_IRIS
```

The first comparison answers whether closed hidden coverage is actually needed. The second isolates perception quality. Do not convert a failure automatically into a full-3D reconstruction requirement.

### P0/P1 — Arachne -> Codec information-preservation test

Test whether two mechanically distinct pairwise fields that require different W can collapse to indistinguishable/effectively insufficient per-joint latents for the Codec. Compare:

```text
oracle codec latent / teacher codec ceiling
Arachne predicted latent
final decoded W
actual LBS behavior
```

If shipping Codec itself passes but Arachne->Codec fails, the boundary is the problem. If shipping Codec fails, representation capacity owns the first blocker.

### P1 — full-chain skinning behavioral panel

```text
S + Qualified G
 -> shipping Arachne
 -> shipping Codec
 -> SkinProposalIR
 -> Compiler.qualify_skin
 -> Qualified W
 -> mesh binding
 -> verified LBS
```

PASS authority is behavior, not Arachne latent loss or raw W alone.

### P1 — deterministic-prior decision

Only if the learned shipping W repeatedly fails on a generic deformation class should a deterministic skin prior be considered. If introduced, it must be explicit, bounded, hash-bound, causal-tested and re-proven; no hidden RigAnything-style smoothing may silently become truth.

## 12. Architecture decision

Current architecture is **not falsified as a whole**.

The cleanroom audit finds:

1. IRIS + GeometricSubstrateAssembler is the correct observation-domain analog of the references' closed-mesh P+N preprocessing at the **information-function** level, while full-surface coverage remains an empirical gap.
2. Geppetto + Compiler is a credible and behaviorally closed counterpart to RigAnything-class skeleton generation.
3. Arachne's explicit segment-aware conditioning is strong, but the Arachne->per-joint-latent->Codec split remains the single material representation risk.
4. The current tiny Codec panel cannot adjudicate that risk because it does not instantiate the shipping Codec.
5. Compiler/proof authority is a genuine RealSaS strength, but it cannot compensate for semantically insufficient G/W evidence.

### Authorization

`ARCHITECTURE_REPAIR_FROM_CLEANROOM = NOT AUTHORIZED YET`

`SHIPPING_CODEC_SMOKE = AUTHORIZED_NEXT`

`FULL_SURFACE_VS_OBSERVATION_ORACLE_CEILING = REQUIRED_BEFORE_REAL_FAMILY_CLAIM`

`GLOBAL_REFREEZE = BLOCKED`

`FORMAL_FAMILY_SELECTION = BLOCKED`
