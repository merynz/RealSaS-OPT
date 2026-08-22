# RealSaS Master Corpus — Source & Legal Audit V1

Status: `CANONICAL_SOURCE_REGISTRY_V1`
Date: 2026-08-22

A public/downloadable dataset is not automatically training-authorized. `TECHNICAL_PASS ∧ LEGAL_PASS ∧ PROVENANCE_RESOLVED` is required.

| Source | Truth value | Legal state | V1 action |
|---|---|---|---|
| Objaverse / Objaverse-XL linked originals | original mesh, armature, hierarchy, vertex weights, animations/materials | per-object | Primary L0 authority for assets linked from Articulation/RigXL. |
| Articulation-XL2.0 | mesh, normals, joints, bones, root, point cloud, sparse skin | conditional per object | Full main + diverse sources, object license resolved. |
| RigXL / UniRig | mesh, joints, parents, names, matrix_local, dense skin | conditional per object | Full pool after object license gate; VRoid REVIEW. |
| MakeHuman / MPFB core | parametric humanoid mesh/rig/weights | CC0 core | Legal-gold; explicit source/version provenance. |
| Blender Studio open rigs | production rigs/weights/control/correctives | per-asset CC-BY | Curated only with explicit compatible page/license and attribution. |
| Quaternius | stylized rigs/weights/animations | CC0 packs | Legal-gold. |
| KayKit | stylized rigs/weights/animations | CC0 | Auto legal-gold. |
| Khronos glTF samples | skin/IBMs/hierarchy/animation conformance | per-model | QA only. |
| Anymate processed | 230k-scale rig/skin supervision | unresolved annotation-data license | QUARANTINE; originals preferred. |
| HumanRig | rigging truth | CC-BY-NC | REJECT commercial training. |
| RigNet / TARig | rigging truth | ModelsResource source-rights chain | REJECT commercial training. |
| Mixamo | rigs/animations | explicit AI-training restriction | REJECT. |
| RigAnything research source | research assets/checkpoints | noncommercial/research | REJECT product training. |
| OmniRig/ARMO | potentially large | release/source-rights unresolved | QUARANTINE. |
| SkinTokens composite | useful reference | composite lineage unresolved | REFERENCE only. |
| Objaverse-SK/TMS/TextuRig | derived annotations | redundant to source authority for current master | REFERENCE only. |
| MB-Lab default outputs | generated human assets | incompatible with clean commercial policy | REJECT. |
| SMPL-family restricted data | human-body truth | custom/research restrictions | QUARANTINE absent separate commercial rights. |

The default V1 does not blindly render the entire animated Objaverse universe. It renders the legally admitted expert-annotated Articulation/RigXL pool and resolves/materializes their linked original assets. Unlinked Objaverse armature expansion is an explicit future/optional pass.

CC-BY attribution is retained per admitted source record. Unknown/custom/share-alike cases never auto-promote to PASS.
