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

`tools/research/run_frozen_semantic_evidence_v1.py` on the semantic-evidence research branch executes two frozen upstream models to produce **non-authoritative semantic mask evidence**. They do not replace, fine-tune or mutate the canonical IRIS geometry model, M, rig, skin or motion.

### Grounding DINO — research evidence producer

- model repository: `IDEA-Research/grounding-dino-tiny`;
- pinned Hugging Face revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`;
- `model.safetensors` SHA-256: `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`;
- declared upstream license: Apache License 2.0;
- role: open-vocabulary box/class proposal evidence only.

### SAM 2.1 Hiera Small — research evidence producer

- model repository: `facebook/sam2.1-hiera-small`;
- pinned Hugging Face revision: `ee5bba1d82bb8749febdf90f45e84b687142ba03`;
- `model.safetensors` SHA-256: `0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60`;
- declared upstream model/checkpoint license: Apache License 2.0;
- role: box-prompted modal segmentation evidence only.

These dependencies are **not promoted product authority** by their presence on a research branch. Promotion requires an explicit dependency review, retained notices, exact reproducibility evidence, and successful RealSaS qualification. External model outputs remain proposals/evidence; Compiler qualification remains authoritative.

## Software/runtime dependencies

RealSaS also uses ordinary third-party software libraries such as PyTorch, torchvision, NumPy, SciPy, Pillow, pytest and repository-development tooling. They remain subject to their own licenses. Current developer/CI pins live under `requirements/` and do not replace those upstream licenses.

The semantic-evidence research runner additionally uses Hugging Face Transformers. The research environment is pinned separately and does not rewrite current mainline dependency pins.

## External research references are not dependencies by default

A third-party project/model name appearing in a comparison, clean-room audit, bibliography, historical report or scientific-lineage document does **not** imply that its code, weights or model are incorporated into RealSaS.

RealSaS model/package/class/architecture names should describe RealSaS-owned responsibilities or mechanisms rather than third-party project branding. Direct dependencies may retain their real upstream identity where technically and legally necessary.