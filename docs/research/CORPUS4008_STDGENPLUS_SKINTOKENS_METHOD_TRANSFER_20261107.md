# Corpus4008: StdGEN++ / SkinTokens method-transfer audit

**2026-11-07 — Data engineering research, not architecture adoption or training authority.**

RealSaS/BUSTED is compiling an editable persistent puppet from source evidence, not merely generating a separated mesh or predicting a skeleton. The two papers provide useful subproblem techniques, not substitutes for our jointly source-bound semantics + mechanics + visibility + graphics + edit-consequence contract.

## Reviewed references (source, not inference)

- StdGEN / StdGEN++: [CVPR 2025 StdGEN](https://openaccess.thecvf.com/content/CVPR2025/papers/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.pdf) and [2026 TPAMI StdGEN++](https://pubmed.ncbi.nlm.nih.gov/42776890/), code [hyz317/StdGEN](https://github.com/hyz317/StdGEN).
- SkinTokens: [SkinTokens, ACM TOG 2026](https://doi.org/10.1145/3859874), [research method](https://zjp-shadow.github.io/works/SkinTokens/), [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), [model checkpoints](https://huggingface.co/VAST-AI/SkinTokens).
- We inspected repository modules, not just abstracts: `StdGEN/slrm/utils/camera_util.py`, `slrm/utils/mesh_util.py`, `SkinTokens/src/data/sampler.py`, `src/data/dataset.py`, `src/rig_package/parser/bpy.py`, and `src/model/tokenrig.py`.
- Source code licenses, training data asset rights, model artifact rights and rendered-derivative rights are separate legal gates. Method descriptions may inspire our independent implementations; no wholesale code copy.

## What we transfer from StdGEN++

StdGEN's public Anime3D++ preparation filtered approximately 14,000 gathered 3D anime candidates to 10,811 accepted models. It used standardized humanoid A-pose and multi-view RGB, depth, normals and semantic renderings, including visual combinations such as full character and body/cloth separation. StdGEN++ extends source decomposition and semantic layer fidelity. These are **dataset process reference points**, not acceptable universal labels for our animals, wings, tails, quadrupeds, spiders, side-authored props or stylized 2D output.

| Research practice | RealSaS target evidence | Extra court before PASS |
|---|---|---|
| Source filtering and canonical rest | Identity/rights/import/camera per asset | Source SHA, native pose parity and orientation review |
| RGB/depth/normal/semantic synchronized views | Shared source-frame 8-view packet for IRIS Geometry/Semantic/Graphics | Camera intrinsics/extrinsics/face-ID ray support, no view-specific source renormalization |
| Multi-component source rendering | Source-face semantic region support, visibility graph, layer reveal | Native source isolation intervention; no naming heuristic pretending semantic truth |
| Body/cloth/head detail separation | Functional/control/attachment/occlusion evidence | Generic component necessity, not a humanoid-only clothing ontology |
| Texture-layer handling | Native material truth plus derived 2D game-art target | Source appearance vs stylized output must stay separately typed |
| Cropped/canonical input imagery | Valid foreground, framing, raster consistency | Anatomical camera and identity cannot come from guessed source up |

**Do not copy:** VRoid material-name-to-label shortcuts, forced A-poses for nonhumanoids, synthetic occluded-surface completion as original source truth, or a source-normalized 3D RGB image as direct target for our stylized 2D Graphics head. In addition, the cited StdGEN repository includes modules carrying separate third-party license notices; use only principles unless source use is cleared.

## What we transfer from SkinTokens

SkinTokens' *FSQ-CVAE* compresses sparse skinning into discrete tokens; TokenRig jointly predicts skeleton parameters and skin tokens. Its training includes importance sampling of deforming areas and an RL stage with Volumetric Joint Coverage, Bone–Mesh Containment, Skinning Coverage/Sparsity and Deformation Smoothness rewards.

We should evaluate these against AXIS/MIRA, but **our supervision must retain the exact source weight/influence identity and mechanical control consequences independently of learned compression**.

| Research practice | AXIS/MIRA application | Independent source court |
|---|---|---|
| Bone-parent graph + skin jointly conditioned | Shared rig/skin teacher binding | Stable source bone IDs, bind matrices, world transforms, source vertex/face mapping |
| Sparse skin tokenization | Optional auxiliary latent representation or compression ablation | Exact original sparse weights preserved; reconstruction error and unaffected regions measured |
| Importance sampling near deformation | Increase sampling density at joints/contact/tail bases and narrow limbs | Preserve unbiased coverage audit across full mesh and zero-influence regions |
| Joint coverage / bone-mesh containment | Loss/validation diagnostics for mechanical plausibility | Nonhumanoid exceptions, props, external controllers, malformed open meshes |
| Skin coverage/sparsity and smoothness | Secondary MIRA quality diagnostics | Must not prefer false skin smoothing over exact source-attested weights |
| Unified rig autoregression | Potential AXIS→MIRA conditional architecture study | Rig hierarchy is not the native animator IK/FK graph; no controller teacher from mere bone names |
| In-the-wild generalization | Stratified morphology/rig complexity holdout | No duplicate rigs/clips/assets crossing train/validation/unseen |

### New data we must collect beyond both references

1. **Controller graph**: source controller IDs, transforms, IK targets/poles, FK/IK switches, constraint order, driver dependencies and negative unaffected controls.
2. **Causal intervention**: control perturbation → evaluated bone transforms → skin/deform geometry → attachment positions → rendered rays/pixels; pose/camera changes as separate interventions.
3. **Semantic authority**: independent source-face component support vs designed functional affordances vs minimal alternative representation policy; UNKNOWN never becomes a supervised negative.
4. **Graphics authority**: source-native RGBA/visibility (evidence) vs Corpus370-style, source-bound **2D stylized** target (presentation).
5. **Persistent identity**: the same canonical source identity and stable source vertex/bone/face addresses across all edits, time samples, views and render styles.

### Pre-materialization experiments and ablations

- **Baseline MIRA**: original sparse source skin weights conditioned on a proven rig. No tokenizer.
- **SkinTokens-inspired MIRA**: optional discrete FSQ-derived skin token prediction, while decoding back to source vertex indices and independently measuring L1, sparsity, per-bone influence recall, LBS deformed-vertex Chamfer, and held-out poses.
- **AXIS/MIRA coupling**: shared joint/skin source truth teacher vs sequential rig-then-skin; compare controller response only on native control rigs which pass interventions.
- **StdGEN-inspired semantic**: candidate source-face region maps from source groupings and synchronized views vs human-reviewed generic component teacher; compare morphologies, not only humanoids.
- **Graphics ablation**: source material render vs approved stylized 2D game-art render; compare with Core370 approved reference-art domain. Do not allow the source-render baseline to contaminate the target as if it were accepted.

**Data admission remains head-specific**, not one global PASS flag. 1,000 distinct FIT source lineages per model head is a coverage objective, with separate stricter full-IK and 2D-art counts. Missing IK does not reject an otherwise legally sourced and independently validated Geometry / Semantic / skin candidate.

## Current implementation

- `canonical/REALSAS_CORPUS4008_PREMATERIALIZATION_ENGINEERING_STANDARD_V1.json`: versioned reversible normalization and target standardization policy.
- `tools/corpus/blender_corpus4008_source_bounds_v1.py`: SHA-pinned importer source-world bounds without anatomical or source-rest authority.
- `tools/corpus/build_corpus4008_normalization_plan_v1.py`: pure reversible IRIS and TESSA mapping with roundtrip tests; not a full source mesh/rig deformation parity court.
- `tools/corpus/validate_corpus4008_prematerialization_v1.py`: head-specific immutable evidence precheck. Intentionally cannot authorize materialization without independently trusted admission.

**Unfinished gates are not PASS**: source native unit/up/front, per-face original mesh address, bone/skin/bind-rest parity, action controller interventions, 2D art human review and professional source rights. All still require actual producer receipts and separate reviewers.
