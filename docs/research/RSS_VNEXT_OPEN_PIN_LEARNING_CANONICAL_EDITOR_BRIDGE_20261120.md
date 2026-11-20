# RSS.vNext — Learned OPEN pin support and canonical editor handoff (2026-11-20)

**Scope:** source-qualified, default-off training objective with review-only inference. Not release quality or automatic executable attachment.

## Research gap

StdGEN (CVPR 2025) reconstructs layered body/clothing/hair surfaces, but semantic OPEN geometry alone cannot determine physical attachment. TailorRig (arXiv 2611.02902v2) learns auxiliary humanoid bone chains and hierarchical chain-to-bone weights, not independently measured two-image source cape pins. MagicArticulate (CVPR 2025) broadens auto-rig morphology but likewise is not a 3D source pin oracle. UIST 2021 Automated Accessory Rigs distinguishes attachments and overlap in layered artwork, motivating separate physical and appearance authorities.

BUSTED previously predicted a HELD versus HOLDER face in one H(x) / TESSA M* / AXIS program but had no trainable vertex-level pin output. The native source-qualified OPEN XPBD compiler had real pins only from separately verified source proof, and the semantic garment policy used explicitly approved heuristics.

## Model head — source pin support, not material

OpenPinVertexHeadV1 in models/rss_vnext/product_open_pin_vertex_head_v1.py uses the same exact final indexed M*, raw H/mechanics features, normals/rest coordinates, topological OPEN boundary and causal AXIS socket/anchor hidden state. It predicts one per-anchor vertex-logit row [A,V], allowing multiple pins without an extra mesh. OPEN edges are calculated by component-safe incidence; touching geometrically coincident body/cape surfaces never weld. More than two incident faces per edge fail closed.

Only source-native pin_targets [A,V] plus known_mask [A,V] may supervise a masked BCE. Unknown is not a fabricated negative; every labeled anchor has an actual source pin positive on the OPEN boundary. A trusted independent launch receipt binds dataset/source rights, native OPEN hem and pin witness SHA, held-out host-only source intervention SHA, source-to-final-M* vertex transport, two original RGBA/camera bytes, generator pin, AXIS sequence, final M* address and target/mask hashes. No student-nearest or generated-view teacher is admitted.

The training stage is PRODUCT_SEEDED_IRIS_TESSA_V1 with source-qualified directed anchor-face supervision present. The pin loss is optional, default OFF, a distinct ninth gradient task named OPEN_PIN_SUPPORT, frozen into objective/checkpoint policy and source registry identity. Sparse assets without pin truth continue Geometry/SLR/Graphics/AXIS/MIRA normally.

## Exact model-to-editor candidate

models/rss_vnext/product_neural_open_pin_editor_review_v1.py joins face and pin neural outputs only if AXIS event and source M* topology digests agree. It checks the entire oriented final-M* face order, RSSV indexed candidate vertex identities, rest XYZ, and canonical Mesh/Rig/Skin lineage checks. It refuses same-component HELD/HOLDER, cross-component or overpinned sheets and ambiguous or unsupported host chains. Proposed host chain is a final MIRA skin-weight hypothesis, not a guessed shoulder or a source truth.

The returned record has canonical vertex IDs, face identities and suggested host chain with review-only authority. source_pin_qualified=false, compiler_xpbd_activation_qualified=false and inferred_physical_parameters=false. It never invokes the native source pin admission court or activates declared XPBD. Editor approval and actual executable runtime constraints remain separate. The holder's canonical MIRA skin can suggest a host chain, but the HELD cloth is **not required to have LBS weights on that host chain**: a pinned, independently rigged/simulated cape is physically valid.

Teacher-free inference does **not** run the new pin head by default. It is only callable by explicit `emit_unqualified_open_pin_research_candidate=True` alongside `generate_controls=True`, and then its status is explicitly `UNQUALIFIED_RESEARCH_ONLY_NOT_PRODUCT_RELEASE`. An initialized but untrained head must never silently become a product/editor recommendation. Releasing this feature requires native pin evidence, held-out calibration and an independent model-release gate; enabling the diagnostic flag does not qualify the weights.

## Product gates not closed by code alone

1. Licensed, independently replayable OPEN cape/scarf/robe pin and host-only intervention corpus with source-to-M* face/barycentric transport.
2. Held-out cape pin precision/recall, incorrect-host rate, false pins on touching components and AXIS/MIRA intervention skin leakage.
3. Ablate face head, pin head and intervention contrast separately with fixed compute and source qualification.
4. Canonical M* save/edit/reopen, guided repair and explicit dynamic-policy approval, full XPBD contact and **post-physics** native Graphics pixel/alpha parity.
5. A100 throughput/memory qualification and product proof on unseen wing/cape bodies; tiny synthetic regressions are only ABI checks.

References: https://stdgen.github.io/ ; https://arxiv.org/abs/2611.02902 ; https://openaccess.thecvf.com/content/CVPR2025/html/Song_MagicArticulate_Make_Your_3D_Models_Articulation-Ready_CVPR_2025_paper.html ; https://shape.stanford.edu/research/autoRigAccessories/autorigging_accessories.pdf .

**No TailorRig proprietary training data, weights, material constants, or external GPU proof are claimed.**
