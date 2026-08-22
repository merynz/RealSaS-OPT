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

## V1 automatic acquisition closure (2026-08-22)

The Colab builder now has executable automatic adapters for all clean V1 source classes that do not require a paid/private entitlement:

- **Articulation-XL2.0:** pinned Hugging Face revision; both diverse-pose and main preprocessed train/test NPZs are discovered from repository metadata rather than hard-coded guesses. Large object-array archives are opened only when an explicit free-RAM safety gate passes; otherwise the source remains pending rather than silently omitted.
- **MakeHuman / MPFB:** pinned MPFB source is used as a deterministic CC0 generator. The builder produces low-discrepancy morphology samples and attaches multiple bundled rig variants, preserving generator parameters and exact generated `.blend` bytes as L0 authority.
- **Blender Studio:** a curated set of public character pages is scraped only when the page explicitly states CC-BY and exposes a public `.blend`/`.zip` link. Page URL, attribution, page-content hash and acquired file hash are retained. A page that requires credentials or no longer exposes the asset is a blocker, not a silent replacement.
- **Quaternius:** the builder walks the OpenGameArt collection associated with the original `quaternius` uploader and accepts only pages that state CC0 and contain rig/animation semantics. Downloaded source ZIP hashes and author page provenance are retained.
- **KayKit:** pinned/open CC0 Git source path remains automatic.
- **Khronos glTF Sample Assets:** selected explicit-license models remain QA-only.

Paid/source-only editions are not required for V1. Free legally clean releases are used when they contain the required rig/skinning truth. No third-party mirror is treated as legal authority merely because it is easier to download.
