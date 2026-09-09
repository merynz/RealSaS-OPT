# RealSaS Dependency and IP Policy

## Ownership baseline

Original RealSaS materials are proprietary and all rights are reserved. See `LICENSE`.

A research reference is not a dependency. Reading a paper, comparing an architecture, or implementing a mechanism from first principles does not authorize external project branding as the identity of a RealSaS-owned model.

## Direct model dependencies

A third-party model is a **direct dependency** only when RealSaS actually loads, executes, embeds, links to, or otherwise requires that upstream model/code/weights as part of the current implementation.

Current explicit model-level dependency:

- **DINOv2-S**, exact upstream source/weight authority defined in `models/iris/v2/dinov2_foundation_v2.py` and attributed in `THIRD_PARTY_NOTICES.md`.

If another direct model/code dependency is introduced later, it must be added to `THIRD_PARTY_NOTICES.md` together with its exact source/version/license authority before promotion.

## Clean-room research references

External project names may remain in:

- comparison reports;
- code-level reference audits;
- bibliographies;
- historical scientific lineage;
- preregistrations/results where the exact comparison provenance matters.

They should not be adopted into current RealSaS-owned:

- package/directory names under promoted `models/`;
- class/function/identifier names that define a RealSaS model identity;
- `RealSaS.*` architecture IDs;
- user-facing product/model names.

`tests/repository/test_model_branding_boundary_v1.py` enforces the currently known external-reference boundary. DINO/DINOv2 is intentionally exempt because it is a real direct dependency.

## Software libraries

PyTorch, torchvision, NumPy, SciPy, Pillow, pytest and developer tooling are ordinary software dependencies, not RealSaS model architecture dependencies. They retain their own upstream licenses. Current development/CI pins are in `requirements/`.

## Scientific reproducibility

Root dependency pins govern current repository development/CI only. A sealed experiment's environment is evidence and must not be silently rewritten by a later dependency update. Replaying a sealed experiment under a newer environment requires an explicit compatibility/replay record when exact environment identity matters.
