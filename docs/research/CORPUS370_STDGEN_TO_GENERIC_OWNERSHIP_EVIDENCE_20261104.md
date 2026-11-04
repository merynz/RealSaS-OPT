# Corpus370: StdGEN teacher semantics -> RealSaS generic ownership evidence

**Status (2026-11-04): IMPLEMENTED DIAGNOSTIC CONTRACT ONLY. NOT training READY.**
**Owner:** Corpus370 IRIS/TESSA truth authority on draft PR #327; scientific gate Issue #330.

## External verification, not speculative analogy

The public CVPR 2025 StdGEN paper trains a semantic-aware triplane reconstruction network with 2D rendered semantic, RGB and mask supervision. Its *multi-layer* training selectively masks semantic sets, re-renders interiors, and uses layered depth/normal supervision before explicit semantic mesh extraction.

- CVPR 2025 paper: https://openaccess.thecvf.com/content/CVPR2025/papers/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.pdf
- Exact public teacher renderer: https://github.com/hyz317/StdGEN/blob/main/blender/blender_lrm_script.py
- Public StdGEN++ paper (January 2026 preprint; September 2026 TPAMI publication): https://arxiv.org/abs/2601.07660

**Inspect the actual StdGEN Blender producer**, not just the paper's model diagrams: it maps VRM materials/objects to hair, cloth, face and body using literal names including `Hair`, `Face`, `CLOTH`, `Body`, `SKIN`, `EYE`. It raises errors on unknown materials; paints the groups with fixed distinct RGB colors; renders **complete / no-hair / no-hair-and-no-cloth** conditions; repeats semantic palettes and emits camera, alpha, depth, normal. Its Anime3D++ corpus is curated at 10,811 VRoid anime bodies after filtering ~14k inputs. These are *teacher-data conveniences*, **not a generic part ontology**.

The StdGEN++ publication adds higher-resolution semantic extraction, a dual full-body/face branch and editable facial texture layers. **Do not infer unpublished production internals from its abstract.** It still uses semantic categories such as body, clothing, hair; these categories do not solve wings, tails, quadrupeds, accessories or source-identity persistence by themselves.

## Decision: transfer measurement instruments, not material names

What RealSaS may reuse conceptually:
- canonical aligned camera+pose and immutable source import
- complete render and controlled removal/standalone/recolor counterfactuals
- layered reveal after removing nearer contributor
- per-locus RGB(A), alpha, surface normals, complete ray stacks, depth and TESSA address
- multiple causal source-locus conditions and independent mathematical replay

What RealSaS **must not** copy into its labels:
- body/hair/cloth semantic classes
- material names, Blender object IDs, rig/bone names as student inputs
- color equality or palette index as fragment identity
- source-face IDs as a proof of drawable, skeleton or mechanical owner
- evidence that two source samples affect one pixel as evidence they are a single component

**Non-identifiability:** a face can always be selected and recolored in a teacher scene; therefore editability of arbitrary source triangles alone cannot establish the puppet's correct drawable partition. A *separately proven stable source-locus grouping and consumer consequence* is needed. The correct answer may be UNKNOWN/ABSTAIN.

## Operational identity layers (separately courted)

1. **Source locus**: exact `(source SHA, mesh/triangle teacher address, barycentric, camera/view/time)`. This is an address, not a learned semantic label.
2. **Rendered contribution**: FULL / NO_OP / REMOVE(A) / REMOVE(B) / REMOVE(A,B) / recolor controls under unchanged geometry, camera, pose and alpha. Consequence is measured in premultiplied RGBA. Complete/tie-free ray stack is mandatory; occluded B can be **revealed after removing A**.
3. **Drawable ownership candidate**: a *stable equivalence relation on source loci across view, controlled color changes and time*, conditional on verified source grouping and independent renderer/consumer tests. Mutual visibility or identical RGB is insufficient. Require positive AND negative pairs plus explicit NOT_COMPARABLE/ABSTAIN.
4. **Mechanical ownership**: independently verified authored controls, skin/rig deformation, displacement under A->B/B->A, sham and common-root baselines; same motion does not imply same drawable.
5. **Attachment/cut/draw order**: geometric adjacency, contact and signed rendering interactions independently checked across cross-view/temporal observations; two same-colored samples are not automatically one graphics fragment.
6. **Canonical target**: only after source-relative proofs, tie to IRIS shared-Z / geometry carrier and TESSA scene topology with exact address and family-specific masks.
7. **Student observability and lineage**: raw source artwork and image evidence enter student; privileged teacher names/IDs never do. Original and recolored variants remain one lineage cluster with a leakage-safe split.

## Executable, deliberately conservative contract

`tools/corpus/generic_source_locus_counterfactual_v1.py` independently checks:
- source/camera/rest SHA, two barycentric face loci, 8-view camera pixel and complete non-tied rays
- all seven controlled arms: full, noop, removal A, removal B, removal AB, recolor A, recolor A noop
- *source parity* no-op and alpha-preservation on recolor
- reciprocal conditional effects (B may be revealed only when A removed)
- explicit UNKNOWN for same-drawable, cut, attachment, mechanics, graphics owner; no teacher labels or training authority

`tests/corpus/test_generic_source_locus_counterfactual_v1.py` includes positive *causal visibility* fixture and no-op/alpha/semantic/tie/duplicate/fake-authority negative tests. **These synthetic tests are only a schema/math gate**; they provide *zero* measured Corpus370 ownership rows.

Existing `tools/truth_foundry/blender_corpus370_source_locus_palette_v1.py` emits two-face recolor, FULL/STANDALONE/REMOVAL and already enforces per-ray source parity, with original source art unchanged. It **does not emit independent REMOVE(B)/REMOVE(A,B)** and therefore cannot yet satisfy this stronger contract. Next production step is to add those real Blender render arms, replay with the independent court, run a preregistered stratified pilot, and inspect measured positive AND negative efficacy. Do not fabricate the missing arms.

## Release criteria before materialization notebook

- Actual real Blender producer + external independent court for source-locus pixel effects on fixed held-out subjects.
- Qualified negative pairs for distinct source-drawable and motion effects; explicit N/A for untestable pairs.
- Stable source/rest/camera/canonical identity across views and frames; geometry and alpha replay.
- Cross-family **intersection**, not marginal geometry counts: `IRIS appearance × geometry × graphics/ownership × relations × response × TESSA mechanical teacher`, with missing-head reasons and family masks.
- Leakage-proof global lineage split, stratified unseen, consumer-oracle consequence tests and no privileged teacher fields in V3.2 model inputs.
- Materialization manifest may include only qualified and masked targets, never forged PASS; 18 V5 observation-fail subjects remain excluded until repaired.

**Decision:** Prioritize the source-locus counterfactual/ownership uncertainty as P0, retain all other geometry/Response/TESSA courts as separate mandatory gates, and keep PR #327 draft until real proof closes. Neither StdGEN's semantic map nor this diagnostic code is ownership-learning evidence by itself.
