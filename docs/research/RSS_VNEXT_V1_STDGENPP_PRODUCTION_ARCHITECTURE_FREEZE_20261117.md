# RSS.vNext / BUSTED v1 — StdGEN++-led architecture freeze

**Decision date:** 2026-11-17. **Scope:** model, Compiler, runtime, Graphics, Mechanics, trainability audit. Training runs and dataset processing are explicitly OUT OF SCOPE.
**Authority:** engineering architecture proposal and executable candidate code, not a claim of a trained production ModelRelease or a PASS ProductRevision.
**Canonical reference:** He et al., *StdGEN++: A Comprehensive System for Semantic-Decomposed 3D Character Generation*, TPAMI 2026, https://arxiv.org/html/2601.07660 . Public reference implementation: the **older StdGEN/CVPR 2025**, https://github.com/hyz317/StdGEN . Do NOT assert that the 2025 repository implements the unpublished StdGEN++ branch/trainer.

## 1. Product v1, explicit boundaries

**Product:** two arbitrary-pose source RGBA images -> one immutable multi-stage neural release -> ONE persistent, editable, mechanically consistent 3D canonical puppet -> moving 2D RGBA projection. Retargetable clips and basic editing: canonical mesh, skeleton, skin, attachments/anchors and declared garment material policy. Source identity, unobserved texture status and inexact source cameras remain explicit. A source view NEVER equals a generated view.

**Supported focus:** human-like and common stylized/nonhuman articulated morphologies on which geometry/rig/skin can actually be learned; ordinary close-fitting clothes, robes/skirts/capes and where licensed source truth exists hair. Rigid armor and props use rigid or weighted attachment/skin, not automatic cloth.

**Not v1 required:** human-only facial expression/gaze tracking, video-based facial layer editing, learned artist-controlled drawable graph, tail/antenna physics for arbitrary exotica, a separate SAM3 model in product execution, a new TailorRig/cloth-generator expert, zero-evidence learned material physics, fully generative motion. No user-visible claim of arbitrary morphology or perfect hidden-surface source fidelity.

## 2. One neural release, four reconstruction responsibilities

- **Stage A:** two ORIGINAL images -> two generated canonical RGBA hypotheses (morphology-conditioned neutral pose, never assume human A-pose).
- **Stage B:** BOTH canonical hypotheses -> six aligned predicted RGB/alpha + normal images with virtual cameras. No generated camera is an observed source camera.
- **IRIS = one H(x):** same triplane spatial carrier + (a) continuous geometry signed solid candidate/UDF for open, (b) *3D surface strata* posterior incl. primary form, outer covering, secondary form, focal detail; (c) RGB/density/uncertainty appearance. Surface semantics are NOT automatic cloth, grips, collision, drawable/source truth or compositor order.
- **TESSA:** topology decisions over this very H(x), with component-safe face and vertex incidence; explicit thin/open surface handling. FlexiCubes is a differentiable surface prior/witness, not a second canonical mesh.
- **AXIS:** causal typed skeleton, parent relation, kinematic joints/control grammar and optional explicit anchors, consuming TESSA indexed faces and H(x) local material.
- **MIRA:** exactly the same mesh vertices and causal AXIS joint/skin-code indices; FSQ tokens -> hard decoded weights. No hidden cosmetic rebind. Validate pose consequences on source-qualified rigged examples.
- **Bundle:** Stage A + Stage B + unified reconstruction released as ONE SHA-pinned product model bundle. Stage-trained components are not independently selectable product models.

**Critical open issue:** StdGEN++'s *fullbody/facial* independent LoRA branches produce specialized field hypotheses. BUSTED v1 must preserve ONE H(x), ONE canonical mechanical surface identity. High-frequency spatial detail can use localized branch capacity only when merged into the SAME shared field and original mesh support before rig/skin admission; it cannot instantiate a second rigging or release authority. The original ++ facial branch is not mandatory for a garment-first v1. It remains a benchmark/reference rather than a falsely-ported network.

## 3. The StdGEN++ improvements to adopt, in source order

| Upstream StdGEN++ method | BUSTED implementation rule | Status at this freeze |
|---|---|---|
| Input unification / reference diffusion | Two original images; Stage A+B preserve original sources | Existing two-source generation code |
| Dual-branch S-LRM | Preserve fullbody scene H(x); only verified local-detail capacity under the same H | NOT equivalent to published ++ dual LoRA yet |
| Semantic NeRF surface rendering | Semantic-weighted alpha/transmittance for selected 3D roles | Existing shared-H staged S-LRM candidate |
| Semantic-equivalent SDF (Eq.3/12) | Reuse generic K-way signed solid extraction; OPEN/UDF via explicit separate patch route | Existing signed formulation; open route NOT closed |
| Coarse-to-fine (Eq.4/5) | Coarse negative occupancy with dilation -> sparse fine sample addresses; exact neighbor/cube-case consistency | NOT production sparse: current high-res implementation still evaluates dense full grids |
| 256 x 256 x 384 fine extraction | Sparse/tiled exact full-grid identities; independent parity test against pinned NVIDIA FlexiCubes; no compact-cube hack | NOT implemented. Current anisotropic axis cap 256; cannot claim ++ scale |
| Hole regularization (Eq.14) | Only on verified eligible closed/thin signed strata. NEVER close intentional openings/cloth hems | Truth mask and loss admissibility undecided until source audit |
| Inside-out refinement (Eq.15/16) | Inner static collision boundary while refining outer; same stable face identity for product | Existing geometric refinement module; no full ++ parity or safe canonical refinement acceptance |
| Backprojected appearance | Original pixels retain priority; unknown hidden texels from same H remain inferred; UV is canonical | Existing atlas; source 3D surface ownership and pose-conditioned visibility not qualified |
| Video diffusion texture decomposition | Explicit facial-layer editability not v1; retain typed provenance/role sublayers for future use | Not full ++ texture generator; intentionally outside first release |
| Rig/skin/deform | ++ does NOT supply learned AXIS/MIRA equivalent; BUSTED retains these | Present in our network; training quality/unseen inference open |
| Garment secondary motion | Same mesh canonical face selection -> AXIS/MIRA pins -> existing surface XPBD -> Graphics | SAM3-free proposal added; requires real animated validation |

**Paper facts:** ++ implementation uses fullbody and facial LoRA rank 128, 6 x 320 RGB S-LRM input, 256x256x384 fullbody and 180^3 facial extraction, progressively 512/768/1024 diffusion generation, and face-focused 3-frame video texture decomposition. These settings are *reference-specific*, not evidence that they fit BUSTED's mixed morphology/data/GPU budget. Published holistic improvement vs StdGEN is small (A-pose LPIPS .066 -> .064); the useful gains are geometry detail and decomposition, NOT proof of a learned cloth rig. Source: https://arxiv.org/html/2601.07660

## 2026-11-17 research amendment: TESSA supervised topology actions

**The architecture target is now more precise:** the v1 TESSA expert should learn a continuous *structural refinement/density metric and rest-surface correction* on the one IRIS seed. Topology split/collapse/flip operations can be deterministic and source-safe. **DO NOT introduce mandatory learned per-edge action-token CE or treat source triangulation/edit order as uniquely correct.** Actual StdGEN 2025 remesh code already interleaves normal/mask gradient optimizations and algorithmic split/collapse/flip; ExMesh 2026 and RADmesh 2026 support this split of responsibilities. Articulated simplification literature explicitly optimizes through representative deformed poses and source skin influences.

The source-qualified **posed fidelity** and **component/hem/UV/topology integrity** are required evaluation and conditional supervision, never data inferred from unknown customer camera or static cloth appearance. Only source-native rigged assets provide posed supervision. Source pose data is teacher-only: production AXIS/MIRA still consume full H(x) and EXACT final M*. Full comparison, code availability, licenses and phase/loss specification: [TESSA seeded retopology research decision](RSS_VNEXT_TESSA_SEEDED_RETOPO_RESEARCH_DECISION_20261117.md). The 2026 Learned Localized Mesh Refinement code URL currently has only a LICENSE file; it is a promising optional ablation, not a ready trainer. Existing structural `topology` CE trains an independent mesh, so is **incompatible with the v1 production head** and must be versioned/replaced.

## 3A. The decisive StdGEN++ / TESSA resolution — ONE mesh, no second decoder

**Frozen v1 structural rule:** **IRIS semantic field extraction is the only source of production surface vertex/face geometry; TESSA is a learned, mechanically-aware structural REFINER operating on that very extracted mesh**. The Compiler alone admits its final result as `CanonicalMeshIR`. Stop treating the existing independent TESSA autoregressive triangle sequence as an additional product geometry authority.

This is an intentional CHANGE from the current `models/rss_vnext/unified_character_model.py` architecture: today FlexiCubes witnesses feed TESSA as at most 256 vertex + 256 face tokens, while TESSA's token decoder constructs an entirely independent triangle mesh via `animation_ready_mesh.py`. That does NOT solve the original Knight dual-geometry failure. Merely expanding the token count or forcing correspondence after predicting two meshes also does NOT solve it.

The production intended graph is:

```text
ONE IRIS scene H(x)
   |
   +-> signed / open / semantic-equivalent fields and RGB
           |
           +-> StdGEN++ sparse high-resolution *semantic surface extraction*
                    (body, coverings, optional hair, hard details)
                    | EACH shell is face-indexed, source role is NOT mechanics
                    V
           ONE candidate indexed geometry graph M0
                    |
                    +-> TESSA(M0, H(surface), articulation prior)
                           topology refinements / repairs / structural
                           component edges, thin cloth boundaries, regions
                           optional face split/collapse/flip proposal
                           | EVERY topology edit records parent support
                           V
           ONE final animation-ready graph M*
                    |
                    +-> Compiler admits CanonicalMeshIR(M*)
                           +-> AXIS predicts rig ON M*
                           +-> MIRA predicts skin ON EXACT M*
                           +-> Mechanics LBS/XPBD ON EXACT M*
                           +-> Graphics projects UV/pixels ON EXACT M*
```

**Correctness invariants:**

1. M0 is extracted *once* from shared H, possibly as several disconnected semantic shells in one asset; nothing is welded across unrelated components. A "single mesh" means one canonical INDEXED ASSET and one mechanical vertex ID namespace, NOT a falsely fused closed watertight surface.
2. TESSA's high-resolution form is SEED-CONDITIONED: it can change M0's topology only through explicit operations with parent vertex/face support and validated geometric/semantic/mechanical quality. It MUST NOT autoregress a separate unrelated mesh and then project skin/UV/anchors onto it.
3. TESSA refinement, split/collapse/flip and optional inside-out collision-aware geometry optimization finish **BEFORE** canonical compiler admission, AXIS and MIRA. Any final vertex has a provable source parent lineage.
4. AXIS/MIRA query H at the FINAL post-TESSA vertex positions. They learn joints/weights over M*, never an earlier proxy mesh. MIRA skin is indexed by final vertex IDs; later graphical remeshing is not allowed.
5. One-to-four edge subdivision and identity refinement are implemented as a differentiable **address-preserving scaffold** in `models/rss_vnext/tessa_seeded_surface_v1.py`. Each new midpoint has an exact 0.5/0.5 parent-edge support. This is a minimal executable correctness court, **NOT** the adaptive high-resolution or learned TESSA topology upgrade. The production model is NOT yet wired to this path. Training on the existing independent topology decoder must NOT be called the final v1 architecture.
6. Adaptive refinement must prevent partial-edge T-junctions, component cross-welds, UV seam loss, inverted normals, self-intersections, rigid/cloth misclassification and cost explosion. Non-differentiable split/flip decisions need source-qualified discrete supervision or a separate differentiable/ranking surrogate; continuous positions must retain gradient through H(x). Solver/action quality is enforced by an independent Compiler court, not blindly generated labels.
7. To ship StdGEN++-class extraction, source-pinned FlexiCubes tile/neighbor ambiguity semantics and high-res sparse grid planner must be implemented and native GPU-qualified. The current bounded dense/chunked 100x100x150 product configuration is NOT a 256x256x384 implementation.

**What happens to existing TESSA causal grammar?** Keep it as historical/diagnostic training evidence until its learnable operators are explicitly mapped into the seeded graph structural program (component/face/boundary refinement). Do not silently train/release the old standalone token decoder as the product final mesh head. Reuse its structural memory, geometry-awareness, articulation bias and topology quality tests; replace its *final full-mesh generation contract*, not TESSA's expertise.

### Orthogonal morphology vs surface mechanics

StdGEN++ presentation strata are NOT physical types. A tank's exterior and a Knight cloak may both have high `outer_covering` probability. Model and Compiler distinguish via independent facts:

| Shape/surface | Geometry layering | Mechanical behavior in v1 |
| --- | --- | --- |
| Tank exterior / hard armor | solid outer shell, possibly separate component | rigid/weighted; NO cloth XPBD |
| Fitted garment | garment shell on body | MIRA LBS, not automatically simulated |
| Cape / skirt / loose clothing | independent thin/free sheet, free hem and explicit support boundary | LBS pinned root + optional XPBD free vertices |
| Hair (when available) | separated fiber/ribbon/surface group | skinned by default; optional carefully qualified surface XPBD |

No additional humanoid-only clothing ontology is needed to produce the geometry; a learned source-material coefficient is **not** inferred from 2D semantic roles. A bounded product-versioned Compiler policy may activate approved flexible-surface defaults when morphology, sheet topology, pin/skin and collision checks all pass; otherwise ABSTAIN or user edits. A separate human confirmation is not inherently required for every compile, but never activate XPBD solely because a face is `outer_covering`.

## 4. Exact one-surface ownership and deformation order

```text
Two user images
  -> pinned Stage A + B -> shared IRIS H(x) [geometry + 3D strata + appearance]
  -> coarse-to-fine semantic surface evidence, component-safe TESSA canonical topology
  -> ONE accepted CanonicalMeshIR [persistent vertices, faces, components, UV support]
  -> AXIS CanonicalRigIR + MIRA CanonicalSkinIR on SAME M
  -> clip/FK/IK -> primary LBS deformation(M, rig, skin)
  -> optional *explicit* garment surface XPBD(M, pin/cloth constraints)
  -> Graphics always consumes FINAL deformed vertex coordinates + canonical UV/texture
  -> native 2D projected RGBA; edit/save/reopen preserves all source-bound identities
```

No destructive remeshing AFTER canonical rig/skin identity is sealed. If a high-resolution surface is generated and needs additional topological detail, refine BEFORE canonical rig/skin admission or use fully checked stable parent-face barycentric refinement tied to the same canonical vertex/face support and deterministic weight lifting. Such a derivation cannot become an independently predicted cosmetic mesh, a second authoritative rig, or silent transfer. **No claim of that full refinement acceptance is yet made.**

**3D strata vs source components:** The body under cloth and the cloth may occupy distinct connected shell components; they may still be in ONE canonical mesh *artifact* and one common vertex-ID/skin domain. Do not require an open hem to be watertight. A globally signed field is admissible only for actual qualified closed solid components; UDF/open patches must not be forced to fill holes.

## 5. Garment motion: no SAM3 or learned TailorRig mandatory

Use the existing surface XPBD solver in `mechanics_xpbd_surface_v1.py`, not a new runtime. Semantic appearance and geometry are NOT material truth. Explicit authoring/approved default must determine whether a covering is flexible cloth versus rigid armor; physical profile is a reasonable starting policy, not learned ground truth.

New `mechanics_semantic_garment_v1.py` consumes the SAME model Graphics/role atlas, checks canonical hash and role posterior, chooses connected faces, validates explicit anchor joint(s) against MIRA skin weights and local boundary, pins any cut seams, refuses fully fixed patches, then delegates to canonical XPBD admission. Does not auto-activate or mint ProductRevision. The existing surface solver integrates at 120Hz only if the runtime supplies actual `dt_seconds`; snapshot does not simulate.

**CAPE visual acceptance:** On the SAME source animation and SAME canonical mesh, compare XPBD off/on: nonzero displaced free vertices, pinned vertices preserved, collision/contact residuals, per-frame final vertex coordinates and native RGBA pixel differences. If the old Knight browser shows frozen source artwork or an old CAA frame, that is not evidence of failed XPBD.

**Rigid attachment:** shape semantics do not prove grip. Existing attachment anchors and qualified socket/held face evidence stay; anything unsupported ABSTAINs rather than guessing contact.

## 6. Graphics and editing

Graphics remains sole final RGB/alpha/visibility/depth/compositing authority. UV atlas must belong to the same final canonical mesh and remain persistent. Source original RGB vs neural hidden completion vs unknown must stay distinguishable. Render the post-XPBD mesh, never the old LBS or sealed sprite instead. Independent learned editable 2D drawables/artist layout and full ++ video facial texture decomposition are not v1 blockers. An internal face/fragment address may exist to support occlusion/pick; eliminating learned artist-style Drawable does not mean removing basic depth and visibility.

**Coverage:** 100% *defined rendering of emitted surface samples* is an implementation target; 100% ground-truth hidden color from two views is impossible. Unknown hidden texels get explicit inferred status, not magical source fidelity.

## 7. Teacher/trainer work allowed in architecture stage

Do not change the 21k corpus, notebook runs, license screens or process new assets here. Model trainer may be inspected and contract mismatches repaired; do not activate loss families whose source evidence/mask/pose status is unknown.

- Stage A and B: independently source-rendered canonical and multi-view images / camera calibration, not student's own predictions as GT.
- IRIS S-LRM: phased RGB/alpha/semantics, then peeled inner-layer renders, then mesh depth/normal/deformation; only where corresponding source-native evidence exists.
- TESSA: source mesh topology and deformation consequences; topology CE alone doesn't prove animatable surfaces.
- AXIS/MIRA: source actual bind/rest skeleton, exact hard-FSQ token targets, source joint poses and LBS consequence; separate source subset qualification.
- XPBD: Runtime policy, NOT a learned cloth-material head in v1. No static semantic pixel becomes contact/physical supervision.
- Loss ownership: keep objective -> real output -> exact source target -> qualification mask -> trusted provenance -> shared-H gradient receivers -> training phase. Fixed weighted-sum optimization is not magically qualified; PCGrad/MGDA/CAGrad research doesn't select itself. See PR #577 inventory and current unified trainer audit.

## 8. Training admission — red gates, NOT claimed closed

1. Source 3D vs uncalibrated 2-image product contradiction: independent source mesh/camera support is possible for teacher data but **not generally available to customer inference**. Compiler must separate *source-training fidelity proof* from *product inference admission based on generated mesh geometric/pose/render evidence*; never invent measured original cameras. Existing strict source-correspondence branch cannot be silently treated as all-user input ready.
2. StdGEN++ high-resolution sparse FlexiCubes actually executes on pinned operator with exact full-grid cube and ambiguity semantics. O(25M) dense allocation is not equivalent.
3. Source signed solid vs open cloth surface policy is executable and non-lossy under canonicalization.
4. Generated semantic layers do not fragment a valid mechanical surface or create unsupported body intersections.
5. TESSA output topology identity -> AXIS rig -> MIRA hard skin -> posed LBS across original unknown test characters.
6. Surface XPBD animated garment and post-XPBD native Graphics actually change pixels without source appearance regression.
7. Production input DINOv2 patch 14 vs 8-pixel VAE/UNet pyramid is one ABI. Branch now defaults to 448-pixel common-grid product images; scientific quality and memory tradeoffs still require evaluation.
8. GPU native raster/FlexiCubes forward+backward and checkpoint/restart parity; CPU-only synthetic CI cannot qualify a large A100 run.
9. Edit/save/reopen and source-independent product proof accepted on unseen characters, not historical Knight composite.
10. A dataset truth-to-mask-to-loss coverage report required BEFORE the large training; none of this branch creates it.

## 9. PR disposition and non-goals

- **#378:** previously merged hosted CI/WSL deployment ops, NOT a rig proposal.
- **#577:** source-truth/loss audit, valuable as a *read-only contract*, no automatic training expansion.
- **#578:** opt-in draft RigProgram math; do NOT merge merely because it exists. Basic source pose chain and skin consequences may still be useful, but not a v1 learned advanced control promise.
- **#579:** SAM3 draft proposal; not a required dependency. Keep historical experiment, do not put SAM3 into v1 production.
- **Independent 2D Drawable partition:** outside v1 product surface. Existing face-addressed visibility/UV and mechanical attachment relations remain.

## 10. Observable code edits on this branch

- Shared DINO/UNet pixel ABI guard and production bundle default aligned to 448.
- A separate SAM3-free, explicit garment semantic-to-canonical-XPBD candidate compiler, without second geometry or physics authority.
- CPU regression test for garment failure cases, canonical identity and real surface XPBD stepping, inserted into unified product CI.

**Status is deliberately partial:** Seed-to-refiner topology contract and gradient regression exist, but the actual inference DAG still uses the independent TESSA face generator and MUST be rewired before v1 release/training.  this branch does NOT yet implement full published ++ dual-branch training, 256x256x384 sparse extraction, video-diffusion facial texture split, native GPU behavior court, or fully qualified user-input ProductRevision. Publishing or saying 'training-ready' now would conceal open P0 engineering. Use this as the frozen architecture TARGET and close executable gates before training.
