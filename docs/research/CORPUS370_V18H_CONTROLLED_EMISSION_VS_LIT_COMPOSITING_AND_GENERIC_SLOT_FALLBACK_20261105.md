# Corpus370 V18h — source-controlled emissive compositing falsifier and generic modular fallback

Date: 2026-11-05. Status: **REAL ORIGINAL SOURCE RUN 3/3 PASS; INDEPENDENT COURTS AND PRIVATE REMOTE SHA READBACK VERIFIED; NO DRAWABLE OR TRAINING AUTHORITY**.  
Repo: merynz/RealSaS-OPT, draft PR #343, branch `corpus370-v18b-reactive-stain-source-edit-court-v1`.  
Actions: https://github.com/merynz/RealSaS-OPT/actions/runs/42120479356 (2 original source jobs, independent redteam before Blender).

## Why this research remains warranted

The "compiler, not generator" product requires generic, source-authoritative, persistent and independently editable 3D-mechanics/2D-presentation puppet structures. Mechanical source-face reaction similarity produced non-semantic candidate groups in V18a. Physical source edit visibility was fixed and independently proven in V18d/e, and all 416 actual physical source edits in V18f were independent-courted. V18g demonstrated that **independently rendered G and its complement R do not always recompose to source under two simple premultiplied OVER orders**. For CAT all thresholds yielded between 147 and 261 per-pixel-oracle failures on six probed candidate groups; for TRICERATOPS threshold 0.045/0.10 yielded 119/197, and 0.18 yielded perfect but *vacuous* fit at K=1.

This does **not** show the problem is insoluble or that any candidate must be artist-incorrect. Independent lighting/shadows/reflections, subpixel coverage, multiple ray hits interleaved between groups and deeper graphics authority can defeat naive 2-layer OVER.

## V18h fixed experiment: isolate shading from ordering

Original-source full native parent V14; exact independent V18a source group graph; original third pose (pose 2); four calibrated source views (0,2,4,6); 192px/32 Cycles samples; physical removal not Transparent BSDF. Original Cat and Triceratops only. Frozen target query ordinal 0 of the 6 precommitted V11 heldout source faces. Frozen thresholds 0.045/0.10/0.18; **no post-hoc selection**.

**40 genuine Blender renders/source = 4 views × 10 arms:** one physically empty control plus for each threshold `FULL / ISOLATE / REMOVE` with **two fixed emission colors** chosen by the candidate group membership, retaining exact original 3D geometry, pose and source camera. All materials deliberately become the test's two colors and emission-only, without original artist appearance or inferred layer identities; source V18f already proved source geometry/material transport for original lit source, and V18h cannot take over that appearance authority.

The separate V18h independent court reopens the 40 RGBA frames, checks exact parent V18f independent court SHA and V18g independent compositor SHA, source, group face counts, frozen target, empty alpha, numeric values, and original render file SHA. It then contrasts the same per-target/group/view `per_pixel_order_oracle_failed_pixels` against V18g lit original source.

This is a targeted *causal intervention on material/illumination*, not an image-quality contest. A zero emissive error could implicate lighting/appearance coupling while leaving other source regimes unresolved. Nonzero errors under emission suggest raster-sampling coverage, partial multi-hit depth interleaving, shadowing still retained by other scene paths, or representation issues; disaggregate with V17 original ordered source rays. **No V18h result, whether PASS or FAIL, proves original artist layer labels.**

Do not equate `0.18 Triceratops K=1` perfect OVER score to editability; such case has empty complement.

## Original V18h causal outcome — independently confirmed

Actual source run `42120479356-1` completed **3/3 job SUCCESS** (redteam + Cat + Triceratops). Independent court reopened each full 40-frame original Blender V18h payload and the exact original V18f/V18g courts. Private SHA-256 receipts: CAT `305a518f238de91903128fb13fcccac5f2179fa944e9281a6a3cbc0f7b756ee2`, TRICERATOPS `6e493f531bc6e2a914b621e15bb8aceb8e3de62f75515a24261ad3a1656d73ff`.

The measured source **one-target-per-threshold** comparison is:

| Source | Frozen threshold | Parent original lit failure pixels | New emission-only failure pixels | Emissive discriminating overlap pixels | Group size |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAT | .045 | 0 | 0 | 20 | 2 faces |
| CAT | .10 | 0 | 0 | 20 | 2 faces |
| CAT | .18 | 0 | 0 | 20 | 2 faces |
| TRICERATOPS | .045 | 17 | **43** | 75 | 1 face |
| TRICERATOPS | .10 | 197 | **457** | 2,785 | 1,298 faces |
| TRICERATOPS | .18 | 0 | 0 | **0** | 1,334 faces (whole source) |

At .18 Triceratops remains K=1 and is an **uninformative perfect reconstruction**. The controlled emissive Triceratops test worsened its failure count, **disfavouring shading-only explanations for that one fixed source group**. Cat's corresponding original lit reference already had zero error and thus provides no evidence that emission improved Cat; earlier Cat failures from V18g were over **other** groups.

A remaining hypothesis is **sample-filtering / antialias coverage** (filtered alpha of isolated source subsets is not equal to the alpha required by joint source compositing), alongside within-pixel mixed ray order. This is not proved by V18h. V18i's exact alpha inclusion-exclusion test was therefore preregistered to distinguish order-invariant alpha mismatches from RGB/order-only errors on the same sealed original source bytes. It does not re-render or mint artist labels: `docs/research/CORPUS370_V18I_ALPHA_COVERAGE_VS_DRAW_ORDER_FALSIFICATION_20261105.md`.

## Scientific selection of modular StdGEN-inspired alternative (not merely renaming labels)

Papers and published implementations:
- He et al., *StdGEN*, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html , source https://github.com/hyz317/StdGEN
- He et al., *StdGEN++*, IEEE TPAMI advance online 2026 / arXiv: https://arxiv.org/abs/2601.07660
- *Slot Attention*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html
- *Mutex Watershed*, ECCV 2018; *Lifted Multicut*, ICCV 2015, as separate candidate grouping solvers **only after signed edge truth**.

StdGEN S-LRM jointly decodes geometry/appearance and *semantic* fields, and uses differential multi-layer surface extraction and part-wise refinement. StdGEN++ extends dual-branch semantics and appearance layer factorization. Important useful transferable design: **explicit multi-layer surfaces with independent rendering/editing**, not its predefined human/anime body/clothes/hair ontology. These systems were developed for a different input distribution and use generated fills; their semantic predictions/completions must not become exact source-truth.

Renaming semantic labels to `component_0`, `component_1` is **not** morphology-generalization. The teacher/student contract must make part count variable, identities permutation-invariant, support multiple components moved by one control or one component moved by multiple controls, permit overlap/order relations, and enforce source-local validity/UNKNOWN under occlusion. No fixed universal artist part vocabulary or source names.

A candidate **generic modular Graphics head** could predict *anonymous component affinities / occupancy support + appearance/alpha / ordered relation evidence / confidence/abstention* in shared scene memory. Its predictions can be generated by a surface extractor and independently checked by edited rendered source consequences. Matching loss would be permutation-invariant, mask-aware and possibly set-valued when many decompositions are operationally equivalent. This is an ARCHITECTURE HYPOTHESIS, not a change to IRIS heads or model-training admission.

### Decision gate, in order

1. V18h real renderer + independent court must close. **If lit-vs-emissive gap disappears**, investigate layer-aware shadow/illumination and CAA; do not penalize V18a geometry simply for shadow coupling.
2. If emissive still fails despite exact physical face counts, integrate V17 original all-hit rays and distinguish pixelwise interleaving, mixed visibility and antialias/coverage from true segmentation. Test view-dependent draw order rather than inventing single global order.
3. Introduce anonymous candidate part-surface representations and non-trivial edit-addressability tasks **on already sealed source evidence**. Compare native mechanical-only reactive groups versus semantically inspired modular/head baseline only with identical heldout original-source, morphology and render budgets.
4. Derive ATTRACT/REPEL/UNKNOWN from independent operational edit consequences; then compare Mutex Watershed/Lifted Multicut. Neither graph optimizer nor StdGEN provides truth merely by naming a group.
5. Final model-family admission only after genuine exact appearance/draw-order/layer editing authority, canonical consumer binding and lineage-safe holdout; evaluate student IRIS separately.

No new source artist labels; `graphics_owner_authorized=false`, all IRIS/TESSA/training/materialization authorization remains false. Existing original source and prior experiments immutable.
