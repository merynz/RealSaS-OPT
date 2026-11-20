# RSS.vNext — pre-E2E trainer, task interactions and gradient closure audit (20 November 2026)

**Purpose:** engineering audit before the already-selected single-subject E2E. **NO E2E training is authorized by this document.** This is not an assertion that all tasks or model quality are ready. Every source obligation below must be resolved against the real pinned selected subject or explicitly classified `UNAVAILABLE`, with optional loss disabled. We cannot eliminate the need for an actual GPU/real-source acceptance run by code review.

## 1. Verified model execution and data directions

The target product remains **two user images -> trained canonical diffusion -> trained six-view RGB/normal diffusion -> one IRIS H(x) -> one indexed TESSA M* -> AXIS rig -> MIRA skin -> Mechanics (LBS then declared XPBD) -> native 2D Graphics -> Compiler/Studio**.

**Never reverse source authority:** Stage A/B generated images are conditioning hypotheses, not captured observations or source 3D truth. The structural student receives the original two observations plus generated six views, but the native Mesh/Rig/Skin/pose/render teachers enter losses only. The generated view camera is virtual; do not treat it as measured original camera. Source-native labels MUST address the same final M* after candidate mesh refinement, through independently verified native face+barycentric transport.

**What is sequential:** Stage A -> Stage B at inference, followed by the shared reconstruction. Within a structural **optimizer step**, IRIS/SLR/Graphics are shared H outputs; TESSA uses predicted IRIS surface, AXIS and MIRA consume the SAME mesh and face context, and qualified posed/raster/intervention losses send gradients back through reachable continuous M*, skin and H(x) paths. A gradient cannot pass through a hard discrete face choice, hard FSQ token, or argmax by assertion: those require qualified own discrete objectives/surrogates.

**What is NOT sequential:** The product is not four independent generators that produce disconnected mesh/rig/skin outputs. TESSA's old independent token mesh, teacher-forced AXIS sequence, six-view render GT, and XPBD-only material/contacts are never product inference authorities.

## 2. Five macro training domains and allowed handoff

| Domain | Entry / learned parameters | Truth | Required handoff |
| --- | --- | --- | --- |
| A. Canonical diffusion | Pretrained SDXL + **new source-pair canonical pose adaptation** | 2 arbitrary-pose source RGBA -> source-native canonical targets | Pinned trained canonical A weights |
| B. Multiview diffusion | Pretrained SDXL MV-Adapter + **new paired-reference / RGB+normal adaptation** | Independent six RGB+six normal renders and cameras | Pinned trained B weights; A->B generated conditioning hypotheses |
| C. S-LRM | One IRIS/H(x), layered RGB/alpha/semantic, source pose projection only if qualified | 3D rendered RGB/alpha/roles, face normals/depth where admissible | Reconstruction model weights **only** to next compatible phase |
| D. Embedded MIRA codec | Codec encoder, FSQ/hard decoder in the same eventual reconstruction graph | Native skin, source face/vertex/joint address; posed deformation only when native poses exist | Frozen exact codec state + SHA in AXIS supervised sequence |
| E. Seeded unified structural | Same IRIS H(x), seeded TESSA M*, AXIS sequence, MIRA deform/Graphics | All *activated* source-qualified loss families | Final reconstructed weights + A/B generated checkpoint pins and ModelRelease court |

Training C and D may be scheduled in either order only if the prior model state is carried through explicitly, the optimizer is new, and no previously learned weights are lost. The current handoff code explicitly allows `SLRM -> STRUCTURAL`, `MIRA_CODEC -> STRUCTURAL`, `STRUCTURAL -> STRUCTURAL`. It **does not currently enforce a full C->D->E dependency graph**; no launch recipe is qualified for simultaneous old C and D outputs. This is an **OPEN P0 orchestration gap**.

E subphase eligibility policy exists: `MSTAR_GEOMETRY_BOOTSTRAP`, `IRIS_GEOMETRY_SLR_GRAPHICS`, `AXIS_MIRA_MSTAR_BINDING`, `SOURCE_POSED_CONSEQUENCE`, `FULL_SHARED_H_FINETUNE`. However `ProductSeededCurriculumV1` validates objective combinations, **not an actual ordered, time-based scheduler, freeze/unfreeze configuration, or empirical duration**. `COMPATIBILITY_UNSCHEDULED` must not become a final research conclusion.

## 3. Exact task gradients and optimizer audit

- `RSSvNextStructuralTrainer.train_step` computes an actual product-seeded objective, may probe per-task shared gradients, performs a real full loss `backward`, checks ownership, clips and steps the optimizer. This is *implemented*, not merely loss names in documentation.
- `product_multitask_gradient_control_v1.py` splits the full scalar objective into **nine actual loss groups**: Geometry, Semantic, visible Graphics, joint ray render, Mechanics, posed M* render, source-native layer clearance, one-control intervention and OPEN-pin support. Any enabled group absent in the current source is `UNKNOWN`, not negative.
- Gradient owner checks prove non-null gradients on named modules for a synthetic/source-qualified batch, not absence of negative transfer nor a scientifically chosen optimization policy.
- Old `training_gradient_routing_research_v1.py` implements **two-task** PCGrad/MGDA/CAGrad-ball ablations. Do not call it official full-K CAGrad or active product training.
- **This PR adds a K-task CAGrad route** `BOUNDED_SHARED_CAGRAD_K`, solving the official 2021 small-dimensional simplex dual with SciPy SLSQP from true full shared-owner gradient probes. It preserves task-private gradients and the current weighted-sum objective, uses a bounded candidate and checks first-order TOTAL-loss descent and full backward gradient replay. This is an opt-in **research candidate, not a selected winning optimizer**.
- `WEIGHTED_SUM` remains the product default. `BOUNDED_SHARED_PCGRAD` and `BOUNDED_SHARED_CAGRAD_K` must undergo matched **same-source, same-initial-checkpoint, same-compute** comparisons. MGDA (already implemented two-task research) is a required comparator where implementable. Evaluate memory, step time, cosine/conflict frequency, geometry M*/dynamic strain, source RGB/alpha, SLR layer semantics, AXIS/MIRA joint/pose/skin and held-out quality. CAGrad's first-order guarantee is **not** a guarantee for AdamW/post-step or unseen product behavior.

Official references: [CAGrad paper](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html), [official CAGrad](https://github.com/Cranial-XIX/CAGrad), [official MGDA](https://github.com/chrisshencs/MGDA). Licensed upstream research ideas do NOT authorize pulling other checkpoint or dataset weights.

## 4. Hard technical red gates for the selected subject E2E

| Gate | Check on the actual repository/runtime | Status on 20 Nov |
| --- | --- | --- |
| A/B true pretrained task release | Actual trained Stage A paired + Stage B paired RGB/normal, official SDXL/MV-Adapter adapter retained; architecture-versioned inference trained checkpoint, strong visual source review | **OPEN**, separate Stage A/B PR only pins official upstream RGB I2MV |
| Pretrained/legal provenance | Actual weight bytes SHA + immutable revisions + reviewed SDXL OpenRAIL++ and MV-Adapter Apache-2.0 obligations, safe code dependency chain | **OPEN**, no full checkpoint bytes verified on worker |
| Teacher truth coverage | Selected asset native mesh/face+bary provenance, authentic RGB/alpha/SLR/skin/pose evidence and loss-active masks; abort rather than fake | **OPEN**, data-team-dependent |
| C/D/E phase lifecycle | Frozen codec, independent optimizer reset, S-LRM transferred weights, curriculum stage ordering, launcher and resume identity | **PARTIAL**, handoff primitives exist; orchestrator missing |
| Gradient correctness | All enabled family losses into shared H and expected private head, no weak zeros, finite updates, loss partition, measured conflicts | **PARTIAL**, synthetic validations + opt-in gradient routers |
| Gradient strategy selection | Compare true weighted sum, routed PCGrad, opt-in K-task CAGrad and MGDA; select via independent source and held-out metrics | **OPEN**, choose only after controlled GPU experiments, never via preference |
| TESSA M* production | Real adaptive OPEN/CLOSED geometry + persistent face/vertex lineage under same M*, no independent mesh, no accidental component weld | **PARTIAL**, correctness candidates without source/unseen acceptance |
| AXIS/MIRA/constraint gradient closure | Frozen compatible codec; face M* teacher address; actual branch/chain/skin and source pose consequences; no invented control | **PARTIAL** |
| Physics and Graphics coupling | Primary LBS -> declared XPBD -> final native RGBA, verified per-frame skin/cape/gap/visibility and alpha/transparency | **PARTIAL**, no arbitrary-material learned claim |
| GPU operation | 448/896 DINO14/SD VAE8 common ABI or a versioned compatible high-resolution adapter; full-resolution native PyTorch3D/FlexiCubes CUDA, memory, AMP, restart | **OPEN**, real GPU evidence not present |
| Compiler/Studio | Original two-image teacher-free actual causal puppet; canonical .rss; edit .rsedit round-trip; ProductProof evidence independent of edit approval | **PARTIAL**, editor pin persistence #637, no production neural E2E |
| Scientific admission | Exact source/asset-family split, held-out morphology/pose, source-vs-generated binding and product acceptance thresholds frozen beforehand | **OPEN** |

Single-asset E2E is NOT being started now. A future E2E can show overfit and integration but not generalization, final optimizer quality or product release. The above engineering gates must be reviewed and logged before invoking the chosen subject.

## 5. Immediate engineering priorities

1. Finish Stage A/B (new ABI + actual task training and genuine pinned inference). No weak scratch baseline.
2. Finish C/D/E training orchestration and phase-scheduler lifecycle; choose explicit freeze/gradient ownership policies and checkpoint identity.
3. Finish source-scoped exact real teaching tensor mask/quality coverage and selected-subject inputs without running a learner.
4. Run source-qualified gradient-court/GPU infrastructure admission **separate from E2E**, then match CAGrad vs weighted-sum to select optimizer.
5. Last pre-E2E design review: pin weights, qualified dataset, frozen acceptance thresholds, frozen configs and rollback.

The fact that these remain OPEN is not a reason to run an E2E prematurely.
