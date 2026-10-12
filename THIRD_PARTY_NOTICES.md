# Third-Party Notices

RealSaS is proprietary. See `LICENSE`.

This file records **direct incorporated/upstream dependencies** that are part of the current RealSaS implementation boundary. Research papers, comparison targets and clean-room mechanism references are not dependencies merely because they are discussed in reports.

## DINOv2 — direct upstream model dependency

Current IRIS foundation authority explicitly binds:

- upstream repository: `facebookresearch/dinov2`;
- source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
- constructor: `dinov2_vits14`;
- frozen weight identity: SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`;
- RealSaS authority source: `models/iris/v2/dinov2_foundation_v2.py`.

The upstream DINOv2 source at that revision is distributed under the Apache License 2.0. RealSaS does not relicense that upstream work under the proprietary RealSaS license. Any redistribution or use of upstream DINOv2 material remains subject to its applicable upstream license and notices.

The RealSaS repository binds DINO source/weight identities for reproducibility; source availability or a recorded hash does not itself grant rights beyond the applicable upstream license.

## Frozen semantic evidence research dependencies — research branch only

The semantic-evidence research branch executes frozen upstream models to produce **non-authoritative semantic mask evidence**. They do not replace, fine-tune or mutate the canonical IRIS geometry model, M, rig, skin or motion.

### Grounding DINO — V1 negative-witness provider

- model repository: `IDEA-Research/grounding-dino-tiny`;
- pinned Hugging Face revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`;
- `model.safetensors` SHA-256: `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`;
- declared upstream license: Apache License 2.0;
- historical role: text-first box proposal evidence in provider V1.

Provider V1 is retained as a negative witness after the Knight run showed cross-label whole-character mask collapse. It is not the preferred provider path.

### SAM 2.1 Hiera Small — research segmentation producer

- model repository: `facebook/sam2.1-hiera-small`;
- pinned Hugging Face revision: `ee5bba1d82bb8749febdf90f45e84b687142ba03`;
- `model.safetensors` SHA-256: `0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60`;
- declared upstream model/checkpoint license: Apache License 2.0;
- V1 role: box-prompted modal segmentation;
- V2/V3 role: label-independent segment-everything candidate masks;
- V4 role: box+point prompted pixel segmentation after multiview semantic localization.

### CLIPSeg RD64 refined — V2 semantic field producer

- model repository: `CIDAS/clipseg-rd64-refined`;
- runtime-resolved immutable Hugging Face commit is recorded in each provider audit;
- `model.safetensors` SHA-256 required by V2: `d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55`;
- declared upstream model-card license: Apache License 2.0;
- historical role: frozen text-conditioned dense semantic fields used to classify label-independent SAM2 candidate masks.

### PartCLIPSeg — V3 part-specialized semantic field producer

- upstream repository: `kaist-cvml/part-clipseg`;
- pinned source revision: `8abceefff960920019096982688e3e8209ed39de`;
- upstream repository code license: MIT;
- official zero-shot Pascal-Part-116 checkpoint source: Google Drive file id `1ytZpfutk__67GLwB-4jMClehJlnAzYFS` as linked by the upstream README;
- the checkpoint SHA-256 is computed and recorded by each V3 run before evidence is consumed; the checkpoint is not redistributed by RealSaS;
- role: frozen part-specialized dense semantic evidence used to name label-independent SAM2 local masks.

The official PartCLIPSeg repository ships a legacy Transformers fork. RealSaS V3 executes that code only in an isolated research worker environment and passes dense probabilities back to the current SAM2 provider through a typed `.npz` boundary. It is not imported into canonical IRIS or the main runtime environment.

### Qwen3-VL 4B Instruct — V4 multiview semantic oracle

- model repository: `Qwen/Qwen3-VL-4B-Instruct`;
- declared upstream model-card license: Apache License 2.0;
- each V4 run resolves and records the immutable Hugging Face revision actually consumed;
- role: frozen, non-authoritative multiview semantic localization and post-segmentation mask adjudication;
- it does not create canonical geometry and does not directly authorize runtime behavior.

V4 intentionally uses the multimodal model for contextual reasoning rather than pixel authority. It proposes cross-view semantic instances and box/point prompts, SAM2 performs the pixel segmentation, deterministic mask-shape gates run next, and the multimodal model finally judges the resulting overlay across the eight views. Only evidence surviving both courts may reach canonical-M fusion.

These dependencies are **not promoted product authority** by their presence on a research branch. Promotion requires an explicit dependency review, retained notices, exact reproducibility evidence, and successful RealSaS qualification. External model outputs remain proposals/evidence; Compiler qualification remains authoritative.

## Software/runtime dependencies

RealSaS also uses ordinary third-party software libraries such as PyTorch, torchvision, NumPy, SciPy, Pillow, pytest and repository-development tooling. They remain subject to their own licenses. Current developer/CI pins live under `requirements/` and do not replace those upstream licenses.

The semantic-evidence research runner additionally uses Hugging Face Transformers, Hugging Face Hub, `accelerate`, `gdown` and `uv`. The V3 PartCLIPSeg worker uses a separate compatibility environment described by `requirements/partclipseg-worker.txt`; that environment does not rewrite current mainline or SAM2 dependency pins.

## External research references are not dependencies by default

A third-party project/model name appearing in a comparison, clean-room audit, bibliography, historical report or scientific-lineage document does **not** imply that its code, weights or model are incorporated into RealSaS.

RealSaS model/package/class/architecture names should describe RealSaS-owned responsibilities or mechanisms rather than third-party project branding. Direct dependencies may retain their real upstream identity where technically and legally necessary; DINO/DINOv2 is the current explicit model-level example.