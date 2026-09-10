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

## Software/runtime dependencies

RealSaS also uses ordinary third-party software libraries such as PyTorch, torchvision, NumPy, SciPy, Pillow, pytest and repository-development tooling. They remain subject to their own licenses. Current developer/CI pins live under `requirements/` and do not replace those upstream licenses.

## External research references are not dependencies by default

A third-party project/model name appearing in a comparison, clean-room audit, bibliography, historical report or scientific-lineage document does **not** imply that its code, weights or model are incorporated into RealSaS.

RealSaS model/package/class/architecture names should describe RealSaS-owned responsibilities or mechanisms rather than third-party project branding. Direct dependencies may retain their real upstream identity where technically and legally necessary; DINO/DINOv2 is the current explicit model-level example.
