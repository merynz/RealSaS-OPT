# RealSaS Unified Master Corpus Contract V1

Status: `CANONICAL_DATA_ARCHITECTURE_FROZEN__BUILD_AUTHORIZED__MODEL_TRAINING_NOT_AUTHORIZED`
Date: 2026-08-22

## Decision
RealSaS maintains one information-preserving Master Corpus for IRIS, Geppetto and Arachne. Compiler remains deterministic.

Admission is fail-closed:

`TECHNICAL_PASS ∧ LEGAL_PASS ∧ SOURCE_PROVENANCE_RESOLVED`.

Downloadability, a repository software license, or academic use is not enough. CC0/public-domain and clear CC-BY are auto-admissible; CC-BY-SA/custom/unclear terms are quarantined; NonCommercial, NoDerivatives, research-only, explicit AI/ML bans and unresolved rights are rejected from commercial training.

## Layers
- L0 SOURCE AUTHORITY: exact upstream record/file, source hash, provider identity, object-level license evidence, attribution, source-native fields.
- L1 MASTER TEACHER: canonical geometry plus all available rig/hierarchy/skinning truth, source-native passthrough, canonical render authority.
- IRIS export: raster + geometry/correspondence only. Joint/bone/parent/skin/weight/mechanics fields are physically absent.
- Geppetto export: geometry + skeleton/hierarchy; skin/weight fields physically absent.
- Arachne export: geometry + skeleton/hierarchy + exact skinning truth.

Unknown source-field loss is a build failure. Derived normalized fields never replace L0.

## Canonical identity and splits
One physical original asset has one `canonical_asset_id` across original Objaverse, Articulation and RigXL variants. Cross-source variants are linked, not duplicated. Splits are canonical-asset-disjoint. Historical RealSaS Articulation3456 is an anchor, not a ceiling; its exact 384 validation identities remain VALIDATION. Historical RigXL closed/test authority is preserved where resolved. Master possession does not authorize a sealed/test record for model selection.

## Sources
Tier A scale: legally resolved linked Objaverse/Objaverse-XL originals, full Articulation-XL2.0, full RigXL/UniRig.
Tier B legal/artist gold: MakeHuman/MPFB core, compatible Blender Studio CC-BY rigs, Quaternius CC0, KayKit CC0.
Tier C QA: compatible Khronos glTF skinned samples.
Quarantined/rejected sources remain in the registry with reason and are never silently imported.

## Truth preservation
When available L1 retains source-frame and canonical vertices, faces, normals, UV/material provenance, connected components, point clouds, bone/joint positions, roots, parents, bone edges, names, deform/control inventory, source rest/bind/local transforms, skinning weights/sparse triplets, shape keys, constraints/action inventory and any otherwise-unmapped source-native information through the exact L0 blob.

A transform field is not exposed as a canonical teacher until its coordinate convention is audited. Source-frame rest transforms remain source authority.

## 1024×8 observation contract
Each canonical asset is rendered in one neutral/rest state at native 1024×1024 for 8 cyclic yaw slots V0…V7 (0,45,…315 degrees), level orthographic orbit about canonical global Z, screen-up +Z. Semantic S/SE/E/… labels are stored only if facing is actually resolved; otherwise semantics remain UNKNOWN.

The exact image convention is TOP_LEFT origin, image y DOWN, NDC y UP; nvdiffrast output is vertically flipped once before all RGB/primitive/barycentric pixel authority is generated.

Geometry-safe NPR may alter RGB but not camera or geometry coverage. V1 core styles are clean cel and inner-ink cel. No diffusion/generative stylization is permitted in exact-GT core.

## Dense correspondence authority
For each covered native pixel, retain compact reproducible triangle/primitive ID plus float barycentric coordinates and image pixel index. Together with immutable canonical mesh and camera this deterministically reconstructs canonical XYZ, normals, depth and exact same-surface cross-view identity. 512 derivatives are deterministic controls; 128 is historical only.

## Technical gates
Geometry: finite vertices/normals, valid triangle indices, nondegenerate usable mesh, invertible canonicalization.
Rig: finite meaningful deform rig, valid root/parents, acyclic hierarchy.
Skin: finite nonnegative weights, row alignment, audited row sums, no dropped influence columns.
Render: all 8 views complete, nonempty coverage, primitive/barycentric authority aligned to top-left images, style variants share coverage, hashes complete.

Capability-specific admission is allowed: geometry-only may feed IRIS; rig-no-skin may feed Geppetto; full geometry+rig+skin may feed Arachne.

## Execution
The canonical Colab builder is resumable. It legal-prefilters before expensive downloads/rendering, preserves L0 before normalization, checkpoints every processed candidate to Drive, physically packs each consumer allow-list, runs forbidden-field firewalls and validates all manifests against bundled JSON Schemas. It does not start any optimizer.
