# Single-view IRIS foundation decision — 2026-11-03

Status: **single-view frontend deferred by user; current multiview corpus/materialization first; empirical backbone selection pending; no training authorization**.
Machine-readable decision: [IRIS_SINGLE_VIEW_FOUNDATION_DECISION_V1.json](../../canonical/IRIS_SINGLE_VIEW_FOUNDATION_DECISION_V1.json).

## User sequencing decision — 2026-11-03

The accepted future product candidate is **one front-facing reference → pose/camera
normalization → eight jointly predicted, camera-conditioned views → IRIS/TESSA**.
The user explicitly deferred its implementation until the current corpus and
multiview model/truth work are complete. Do not turn this future single-view
milestone into a new gate on current multiview corpus closure/materialization.
This does not authorize current training: existing family/head, sign/probe,
tangent-frame, loader, split and materialization checks still apply.

A larger DINO encoder does not guarantee missing-side generation or overcome
unproven data sufficiency. Assess a compatible pretrained multiview/3D generative
prior and domain coverage separately before resuming this future work. The eight
source views remain truth targets; predicted views never become source truth.
The design below is retained for that later milestone, not an instruction to
change the running notebook or current research conditioning.

## Product objective and scope

The user explicitly confirms **one real reference image → full 3D prediction**.
The existing eight calibrated views are a research/supervision apparatus, not proof
of this product capability. Other real views may remain training/evaluation targets;
they must not leak into the student's inputs. Unknown camera, pose, alpha and scale
require a declared inference interface; source 3D normalization is not an oracle
available for a real illustration. Hidden geometry/appearance remains prediction.

The 2026-08-31 Small-only apparatus default and its S-ceiling prerequisite are
superseded for this new objective. Preserve historical evidence and exact replays.
Large-capacity foundation evaluation is authorized now. Neither a larger encoder
nor a source merge establishes single-view completion or authorizes an optimizer.

## Audited implementation

The actual V3.2.3 trainer overlay SHA256 is
`ce21c3ff518319106de2f6c97f10e1961398c7e7faf68e3fe4f52f6bcb0fb35a`,
matching its Drive RUN_BINDING. It calls `load_exact_dinov2_s_v2`, freezes all encoder
parameters and builds a feature cache. The V2 authority pins S/14, 384 channels,
518 input and a 37×37 grid. The V3.2.3 fusion has 576 channels: final384,
layer2→64, alpha→64, RGB588→64.

`models/iris/vnext/inference_input_v1.py` and
`models/iris/v2/foundation_adapter_v2.py` enforce eight views; inference also binds
the Small authority and 13-dimensional calibrated camera features. Therefore this
is more than a weight filename change. A versioned single-view frontend, compatible
projection/scene conditioning, cache identity and checkpoint migration are required.
Do not relax the historical validators to make new tensors pass.

“Frozen” describes parameter updates, not model capacity or completeness. The
88,283,115-byte Small checkpoint is a complete small pretrained encoder. A Drive
copy is optional persistent caching; runtime still needs verified local weights.
Retain existing weights while selecting the successor; avoid redundant Drive copies.

## Research decision

| Candidate | Visual width / patch | Role |
|---|---|---|
| DINOv3-L/16 | 1024 / 16 | Preferred new research candidate |
| DINOv2-L/14 | 1024 / 14 | Required lower-migration reference |
| SigLIP2-L/16-512 | 1024 / 16 | Language-aligned alternative |
| DINOv2-g/14 | 1536 / 14 | Optional higher-cost comparator |
| Existing DINOv2-S/14 | 384 / 14 | Historical replay / explicit ablation |

Official model cards/configurations define these shapes. DINOv3-L and DINOv2-L
are around 300M visual parameters; DINOv2-g is around1.1B. Dense-feature quality and
token resolution matter alongside capacity. DINOv3 uses its own license and weight
access process; do not label it Apache2.0. No candidate weights or license acceptance
have been obtained in this change. Official TRELLIS.2-4B's published pipeline uses
`facebook/dinov3-vitl16-pretrain-lvd1689m` for image conditioning. This supports
testing that encoder in image-to-3D, not importing TRELLIS quality into IRIS.

The SigLIP2 alternative adds language-aligned pretraining; it is not a demonstrated
winner for our geometric/appearance targets. Preserve native RGB/local detail paths.
Do not compress 1024 channels through the old384 projection and call it equivalent.

## What STDGen++ actually contributes

The paper uses a ViT encoder and image-to-triplane reconstruction. Its two LoRA
branches specialize full-body and face reconstruction; implementation details place
rank128 adapters in attention Q/K/V. A single reference first produces six RGB/normal
views. It initializes reconstruction from pretrained InstantNeRF.
Thus LoRA is adaptation, not a substitute image encoder. The paper does not pin
an exact DINO variant; the older StdGEN public inference repository is not verified
++ training code. Body/clothes/hair targets do not automatically apply to this corpus.

A frozen image encoder and trainable reconstruction decoder are compatible.
Reconstructor LoRA can be evaluated separately if a relevant pretrained decoder is
actually adopted. Encoder LoRA changes features and invalidates frozen caches;
pin its target layers, rank, adapter weights and parent checkpoint. Copying rank128
or adding LoRA to a randomly initialized decoder is not equivalent transfer learning.
A pretrained2D encoder alone does not supply a pretrained3D completion prior.

## Required evidence before future single-view training readiness

Use identical lineage-safe splits and qualified head masks. Compare Large arms
with exactly one real observed view; evaluate held-out views as targets only.
Treat repeated copies of the input as one observation, never eight independent
views. An input-leakage check must vary hidden targets/cameras without changing
student conditioning. Generated views/normals have predicted provenance, not truth.

Separate visible/occluded geometry and appearance, thin structures and failure
cases, and measure peak VRAM, latency and cache bytes. Match token/pixel budgets
or report their difference explicitly; a higher score at higher resolution is
not isolated evidence of backbone capacity. Start with frozen encoders, then
evaluate reconstruction adaptation as a separate controlled intervention.

Record exact source/weight SHA256, size, preprocessing, output levels, patch grid,
register handling, projection, observed views, camera interface, split and all
adaptation identities. Rebuild encoder-dependent features and changed model
consumers; raw source truth remains independently reusable.

Corpus family closure, sign/probe and tangent-frame repairs, masked370 loader,
real lineage split, materialization validation and final model/truth checkpoint
remain open. This decision does not change the running sealed7a61 truth worker
or claim that a Large model has loaded/trained. Next model implementation is the
versioned Large/single-view frontend and its consumer/cache integration, followed
by actual preflight/evaluation under the Platform research contract.

## Primary sources

- [STDGen++ paper](https://arxiv.org/html/2601.07660v1)
- [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- [DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
- [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786) and [official model](https://huggingface.co/google/siglip2-large-patch16-512)
- [TRELLIS.2 official pipeline](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/pipeline.json)
