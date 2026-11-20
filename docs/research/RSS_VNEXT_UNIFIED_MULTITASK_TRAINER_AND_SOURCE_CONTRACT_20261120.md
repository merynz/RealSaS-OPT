# RSS.vNext / BUSTED — Unified trainer, shared-M* curriculum and evidence contract (2026-11-20)

**Status:** executable, source-qualified multi-task training *candidate* integrated with the active product-seeded trainer. **NOT:** a scientifically selected final optimizer, validated full-resolution training recipe, qualified corpus, or passing ProductRevision. The only executable product is the nine-role catalog in `platform/internal/capability/contracts/product_capabilities_v1.json`.

This record is for the model/training engineering workstream. Teacher production is a separate workstream; this does not mutate or authorize source data. It complements the [2026-11-20 Foundry handoff](BUSTED_RSSvNext_Veri_Supervision_Teslim_Sozlesmesi_2026-11-20.md) (the handoff PR may merge independently).

## One product; five training domains, not five product models

```text
TRAINING                          SAME FINAL PRODUCT BUNDLE
A. original pair -> canonical     canonical.pt
B. canonical pair -> six views    multiview.pt
C. S-LRM shared-H pretraining ─┐
D. MIRA embedded codec training├─> one reconstruction.pt (IRIS/TESSA/AXIS/MIRA)
E. seeded structural learning ─┘
                                ONE SHA-256-pinned ModelRelease bundle
```

There is **one reconstruction carrier H(x)** and exactly one canonical indexed **M***. Two arbitrary original RGBA observations condition the student's H(x) directly. Stage A and B output **generated** canonical/six-view hypotheses; virtual cameras are never measured source camera truth. Source-native mesh, rig, skin, motion, patch labels and 3D rendered targets may supervise reconstruction, but **must not enter product inference inputs**.

A/B have independent trained denoiser weights and optimizers. S-LRM, MIRA codec and seeded structural updates are distinct training domains on the **same eventual reconstruction weights** and use weights-only phase handoffs with a fresh optimizer. Train-time resume checkpoints are separately versioned and may be numerous. The released bundle contains three weight components, not one file per neural expert. See `realsas_product_neural_bundle_v1.py` and `training_phase_handoff_v1.py`.

### Reference mapping: what we did and did not borrow

- **StdGEN, CVPR 2025:** [paper](https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html); [released code](https://github.com/hyz317/StdGEN/blob/main/slrm/models/renderer/synthesizer_mesh.py). Geometry SDF/deformation, RGB and semantic probabilities share triplane features. The [ray marcher](https://github.com/hyz317/StdGEN/blob/main/slrm/models/renderer/utils/ray_marcher.py) uses semantic probabilities in alpha/transmittance, and [mesh extraction](https://github.com/hyz317/StdGEN/blob/main/slrm/models/lrm_mesh.py) conditions SDF layers on semantic evidence. **Source caveat:** the released repository announces inference code and checkpoints; the complete training launcher and loss schedule are not available as a fully reproduced upstream run. We implement source-qualified three-stage S-LRM; we cannot claim paper-exact training optimization.
- **TailorRig, November 2026:** [paper](https://arxiv.org/abs/2611.02902); [linked code location](https://github.com/Tencent/TailorRig). Chain-to-bone skin weighting, source-native skeleton identity and LBS-deformation supervision are useful **once our TESSA M* exists**. TailorRig's human template and artist labels are not customer inference inputs, nor is its source dataset automatically commercially eligible. The linked GitHub repository was not retrievable in this audit; **do not claim that source training code was reproduced**. BUSTED independently validates its native AXIS sequences, chain partitions, skin transport and posed-LBS teachers.
- **PAD-Net / MTI-Net:** [open implementation](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch) demonstrates multi-task cross-head feature distillation, deep supervision and multi-scale task interactions. **Not a warrant to add a second scene memory or extra learnable drawable.** Existing BUSTED semantic/graphics-to-structural bridges should be audited before copying adapters.
- **Recon:** [ICLR 2023](https://openreview.net/forum?id=ivwZO-HnzG_) motivates **per-owner/layer** conflict diagnosis, not blind global conflict removal. BUSTED keeps one H(x); automatic task-private backbone duplication is incompatible with current carrier ownership.
- **PCGrad / CAGrad / GradNorm:** [CAGrad NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html), [GradNorm ICML 2018](https://proceedings.mlr.press/v80/chen18a.html). No universal optimizer winner is demonstrated for BUSTED. The new opt-in candidate is **deterministically ordered, bounded shared-owner PCGrad-inspired projection**, **not** the paper-exact randomized PCGrad or K-task CAGrad/GradNorm.

## Three IRIS heads are three different supervision families

| Family | Independently admissible source truth | May affect shared H(x) | Cannot assert |
| --- | --- | --- | --- |
| **Geometry** | qualified source UDF, closed-solid SDF/sign, normals, native surface and topology witnesses | yes; TESSA seeded M* also receives gradient | open cloth is a signed solid; UV appearance implies correct surface |
| **Semantic/SLR** | original patch axis masks, actual source 3D surface role/query, source region co-membership | yes; may condition layered SDF/ray visibility | role means a physics material, authored constraint, cut drawable |
| **Graphics** | original opaque first-hit RGB + ray direction/visibility mask; independent 3D RGB/alpha/role/depth layers | yes, but through admitted appearance/ray losses | an observed first hit reveals hidden UV or pose-time visibility |
| **TESSA mesh** | native source face/vertex/normal/importance plus true boundaries/layer ownership on final M* | continuous rest correction; discrete incidence via its own qualified objectives | signed/unsigned source teacher creates a second cosmetic mesh |
| **AXIS/MIRA mechanics** | source-native rig program, exact joint order, native face-barycentric skin, chain partition; admitted native LBS posed surfaces | can return differentiable posed/strain loss to same M* and H(x) | artist controls or physical cloth are implied by a rig or semantic label |

**Important distinction:** a shared H(x) and differentiable continuous M* do **not** make discrete face selection, causal AXIS tokens or hard FSQ IDs differentiable. Cross-expert mechanical consequences reach continuous geometry/skin paths under qualified teacher alignment. Discrete decisions need their own source-qualified token/structure objectives. Source-native skin-only examples remain valid **without invented motion clips**; posed consequences require actual source poses.

### Product-seeded loss decomposition (implemented)

The active trainer constructs the aggregate objective from *admitted* sources only:

```text
G = seeded_tessa.total + lambda_component_udf * qualified_component_udf.total
S = product_semantic.total                   (only if source-valid)
A = lambda_visible_rgb * first_hit_rgb       (Graphics source-visible task)
R = lambda_joint_ray * source_3d_ray_rgb_alpha_semantic
    (cross-head Joint Render: Geometry density + Semantic + Graphics)
K = product_axis_mira.total + lambda_anchor * source_native_anchor
V = lambda_posed_mstar * native_Mstar_posed_RGB_alpha_roles_depth
C = lambda_native_clearance * native_body_face_garment_vertex_gap
L_total = G + S + A + R + K + V + C (unqualified teachers are UNKNOWN)
```

The seven-family aggregate split deliberately separates cross-head Joint Render, posed M* raster and native garment clearance from source first-hit RGB and uses **the exact product loss tensors**; none of the AXIS/MIRA or semantic sublosses is counted twice. `admitted_product_task_losses_v1` rejects a changed scalar decomposition; `PreparedProductGradientRouteV1.apply_after_backward` independently verifies that the full **selected shared-owner** gradient from `L_total.backward()` equals the sum of the family probes before modifying a single parameter.

This audit is stronger than simply noting that a module has a non-null `.grad`, but is still **not** a quality/gradient-benefit claim on real sources.

## Production optimizer contract and safe research alternative

Current default: **fixed weighted sum; one normal backward; shared and expert-private parameters follow ordinary optimizer ownership**. Research optimization is opt-in only via `ProductMultitaskGradientPolicyV1`:

- `mode="WEIGHTED_SUM"` is default; retains previous checkpoint objective-policy identity.
- `mode="BOUNDED_SHARED_PCGRAD"` performs exact full-vector per-task probes **only** on explicit shared-owner prefixes, default `iris.scene.`. Explicitly allowed expansions are `tessa_seeded_head.` and `surface_mechanics_context.`.
- A deterministic pair-projection candidate is blended with the original weighted sum and bounded by a norm ratio. If it loses first-order **total-loss** descent, it reverts to the true weighted sum. This is a local gradient diagnostic, **not proof of per-task or AdamW/post-step improvements**.
- Without two source-qualified families with nonzero shared gradients, no task conflict is invented and normal backward is left alone. Private AXIS/MIRA/Graphics/SLR parameters never have their gradients rewritten.
- Hard limits enforce maximum probed elements and estimated FP32 probe buffer bytes. Extra backward passes are expensive; **never silently fall back under OOM** or enable per-step full routing on an unqualified A100 corpus run.
- Model shape, active objective coefficients, trusted source registries, stage/curriculum and gradient policy are tied to checkpoint/resume identity. A new phase receives weights only, **not** old optimizer moments/sampler position.
- Measure task-pair cosine/conflict frequency, negative transfer, loss deltas, gradient memory/latency and held-out M* geometry/pose/RGB/alpha. Compare against same-seed fixed weighted-sum, bounded router, and separately validated GradNorm/FAMO/CAGrad where appropriate; **promote nothing by synthetic CI alone**.

Source: `models/rss_vnext/product_multitask_gradient_control_v1.py`, called from `RSSvNextStructuralTrainer.train_step()` before and after the normal backward. The research-only existing two-task routing module remains separate and unchanged.

## Seeded structural curriculum: explicit subphases (implemented policy)

The policy checks **objective eligibility**, not merely that someone assigned a string stage name. It does not create source teacher labels or change freeze sets automatically:

| Subphase | Allowed/required objective families | Intended handoff |
| --- | --- | --- |
| `MSTAR_GEOMETRY_BOOTSTRAP` | source-true Geometry/TESSA; no Semantic/Graphics/AXIS objective | preserve initial H(x), stabilize final M* addresses |
| `IRIS_GEOMETRY_SLR_GRAPHICS` | geometry + real SLR + source RGB/rays; no AXIS | learn three heads without bogus 3D labels |
| `AXIS_MIRA_MSTAR_BINDING` | source AXIS program plus **source skin**, pose consequence not yet required | frozen pretrained embedded codec; exact M*/joint transport |
| `SOURCE_POSED_CONSEQUENCE` | source rig+skin+LBS posed consequence | test deformability and cross-expert feedback |
| `FULL_SHARED_H_FINETUNE` | source-qualified SLR, Graphics, AXIS and posed consequence together | measured cross-owner update and free-running validation |

`COMPATIBILITY_UNSCHEDULED` preserves existing seeded-trainer behavior and checkpoint identities for older tests/experiments; it is **not** a final production curriculum. No version is chosen as scientific winner until held-out trials. Parent domains A/B/S-LRM/MIRA_CODEC have their existing distinct source render/optimizer contracts; the above subphases apply **only to product-seeded reconstruction**.

**Do not infer a strictly linear inference execution graph from this curriculum.** Training may freeze/unfreeze and exchange differentiable mechanical/visual feedback through the same carrier and same canonical mesh. It must never create an alternate canonical mesh, second surface truth, or an independent learned runtime-skin authority.

## Machine-readable Foundry handoff

`RSSvNextStructuralTrainer.source_supervision_requirements_v1()` emits a checkpoint-policy-bound table of requirements enabled by actual coefficients:

- always: provenance, rights, original two RGBA bytes, pinned generator hypotheses, exact seeded M* source normal/face/barycentric evidence;
- as enabled: component UDF for open surfaces; true original patch axes and source 3D semantic roles/co-region; original visible RGB first-hit and qualified source 3D RGB/alpha ray stack;
- as enabled: exact AXIS native hierarchy/joint order, source-native M* face-barycentric skin, source joint-to-chain map and vertex validity, source-authored anchor/constraint intervention;
- for **posed** losses only: source-native pose transforms, matched M* posed vertices, qualified pose mask, true strain/fold targets;
- optionally: native source **posed M* RGB/alpha/role/depth camera render** with immutable raster-source binding, and source-certified outward body face + native garment-vertex barycentric pairs and valid masks for garment/body gap; neither is a source for inferred XPBD material.

Every teacher needs asset lineage and licensed-truth provenance; `UNKNOWN` is masked, **never interpreted as a negative or physical class**. The inventory is *conditional*, not a claim that every licensed asset has all teacher families.

## Important COMPILER / visible puppet consequences that remain open

These are **real product acceptance gaps**, not to be papered over by adding loss weights:

1. **Native M* Graphics Stage3 training:** the **optional source-qualified posed M* raster objective is now connected to the seeded trainer** (PR #632), using the same AXIS/MIRA-deformed indexed M* and Compiler-compatible canonical vertex logits. It remains teacher-forced under a native source pose; GPU execution, unseen independent motion, alpha-compositing beyond opaque first hit and post-XPBD native C++ parity are **not yet qualified**.
2. **Garment not stuck to body:** a **source-native signed gap supervision candidate now exists** (PR #633), but only for a certified outward body shell, independently indexed body-face/garment-vertex pairs and valid native source poses on the same M*. It is not a whole-body collision detector or source-backed XPBD material. Unsupported assets remain UNKNOWN; runtime still needs authored/default XPBD policy or ABSTAIN.
3. **Cape movement:** canonical Mechanics kinematics first, then declared XPBD Physics, then Graphics post-Physics native 2D. Need real source-driven anchors, cloth boundaries and measured collision/contact/pose-time pixel witnesses when claiming correctness. A synthetic cape preview and an available solver do not prove this for unknown characters.
4. **Free-running two-original input:** teacher-forced M* / AXIS / MIRA training is not the same as predicted-geometry+predicted-rig+predicted-skin quality. Evaluate rollout on unseen morphology and source-independent artist motion. Training generated six-view views are conditioning hypotheses, not GT.
5. **ProductProof:** source rights, native renderer parity, independent source appearance/visibility and edited .rss close/reopen remain independently gated. A compile stage returning PASS is not an admissible ProductRevision.

The five remaining acceptance gates are included as explicit non-authority diagnostics in the code-generated handoff; their exact status changes with activated loss policies. They require separate engineering / measured evidence before training can be declared final. Do not make them compulsory *learned* targets without source truth or a compiler contract.

## Acceptance test plan (not yet empirical validation)

1. Synthetic source-qualified seeded geometry+Graphics smoke: exact loss partition, actual IRIS shared gradient, private head unchanged; no ProductPASS.
2. False/missing target and disabled source family rejection; no source-pseudo-label upgrade; exact backward mismatch aborts before optimizer step.
3. Frozen checkpoint routing/curriculum/registry identity; mismatch refuses resume.
4. Same-seed matched-compute comparison of baseline and candidate on licensed multi-morphology train/val/unseen; source-native geometry, masked layer, pose and UV truth coverage.
5. A100 CUDA PyTorch3D/FlexiCubes actual backward+resume; source layer/raster and final M* vertex/face correspondence; memory/step time.
6. Real skirt/cape, wings, accessories: held-out source-native rest and posed M*, cloth–body gap, self-contact residuals, post-pose RGB/alpha and edited .rss native C++ replay.
7. Select routing and stage schedule only if held-out metrics improve or at least jointly meet engineering thresholds; persist the exact selected stage policy and training-source matrix.

**Code state:** candidate functionality and synthetic regression tests in the review PR. No real corpus was consumed, no GPU training job was launched, no default production optimization was silently changed and no ProductPASS was minted.
