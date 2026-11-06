# Corpus370 V18p — Why V18a partitions fail: source topology floor vs collapsed multi-axis response

Date: 2026-11-06. **Original source SHA-locked 3/3 PASS; research and diagnostic only.** Draft PR #343. Main and all IRIS/TESSA/materialization/training authorities unchanged, FALSE.

## Motivation and valid scientific target

V18o showed a frozen Cat source with V18a threshold .10 has 288 candidate parts, mostly small; Triceratops 22 candidates but one contains 1298/1334 faces (97.3%). The two-patch source compositors V18m/n reproduced appearance with oracle-free source depth, but this is a rendering mechanism and does not demonstrate a usable editable drawable decomposition.

Discussion of mechanically distinguishable perturbations raised an important distinction: `same response` is NOT `same drawable`; `different response` is NOT automatically `different drawable`. Structural prior from multi-view geometry and appearance may improve proposal candidate ranking, but source identity remains UNKNOWN without independent operational edits. Old native SAM/SAM2 mask failures cannot be solved by declaration.

Before adopting an external 3D part prior or graph partition optimizer, measure whether current V18a failure comes from **(A) source face topology never connecting candidate islands**, or **(B) mechanically distinct faces within one topological component being merged due to response aggregation**.

The V18a original source V14/V16 fit controls include multiple independently selected native control-axis interventions and source poses 0+1, normalized signed 3D displacement, magnitude, rigid-residual channels plus nonlinear joint-response features. **V18a was already multi-axis**; its final connected-component rule takes the RMS across *all* response channels and merges a shared-vertex-edge face pair if distance <= fixed threshold. This averaging can hide an axis-specific difference. V18p measures that mechanism; it does not invent a new artist part truth.

## Immutable original-source experiment (no new Blender render)

Source: exact original Cat and Triceratops private V18a `PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz`, frozen original SHA-court and source V18o independent partition court, replay of actual original face adjacency, per-axis normalized channel groups, original frozen thresholds `.045/.10/.18`. Note that a source topological island is **a connected component of triangle faces sharing the same source vertex-index edge**. Different vertex IDs, UV/chart seams or object boundaries can create independent islands even where the visible material looks continuous. **An island is NOT an artist drawable.**

Actions: original source independent run https://github.com/merynz/RealSaS-OPT/actions/runs/42201517989 — **3/3 PASS**, mathematical adversarial redteam PASS, Cat and Triceratops full original source independent courts PASS with SHA-verified private Drive readback. Private receipt SHAs:
- CAT `503e1b3879acfa15ab03741ae4ed0afd56270707d7fbff55b490c908b8659502`.
- TRICERATOPS `e5eb1d9078630792015816c1d8f728818d31b116bf3f4e36e30812a63906ca5b`.

Code:
- `tools/corpus/court_corpus370_v18p_response_topology_identifiability.py`
- `tests/corpus/test_corpus370_v18p_response_topology_identifiability.py`
- `.github/workflows/corpus370-v18p-source-response-vs-topology-court.yml`

No new candidate groups are minted. This experiment rereads exact original V18a labels and reproduces them from feature+face-edge arrays; cross-checks 100% against V18o source counts. No unverified artist same/different relation is asserted.

## Actual source geometry and response outcomes

| Original source | Original faces | Source edge-connected topology islands K_topo | Source shared-vertex-edge adjacencies | Independently probed control axes | Original normalized signed channels |
|---|---:|---:|---:|---:|---:|
| CAT | 684 | **260** | 424 | **8** | 168 |
| TRICERATOPS | 1,334 | **1** | 2,001 | **12** | 248 |

### V18a partition components vs source topology floor

| Source | Frozen threshold | V18a candidate K | K_topo | Additional components created by source response cuts | Original shared edges rejected by global RMS |
|---|---:|---:|---:|---:|---:|
| CAT | .045 | 320 | 260 | 60 | 60/424 |
| CAT | .10 | 288 | 260 | **28** | **28/424** |
| CAT | .18 | 280 | 260 | 20 | 20/424 |
| TRICERATOPS | .045 | 469 | 1 | 468 | 862/2,001 |
| TRICERATOPS | .10 | 22 | 1 | **21** | **117/2,001** |
| TRICERATOPS | .18 | 1 | 1 | 0 | 3/2,001 |

**Cat root cause constrained:** At frozen .10, *90.3% of V18a candidate count is an irreducible topology floor under a strictly local edge-only connectivity model*. The 8 independently measured control axes cannot merge separate source vertex-index islands no matter how low their response distance. Relaxing the existing threshold can at best approach K=260. V18p finds **3,335 pairs** of different topology islands whose *mean normalized response* is within .10, but this is neither proof they should be merged nor reliable artist drawable identity. It demonstrates the large, unresolved cross-island relation space. Source action: permit **typed lifted (non-neighbor) candidate hypotheses** from appearance, source chart/UV/material/visibility and controlled local edits, not blind all-to-all mechanical merges.

**Triceratops root cause constrained:** With one original mesh adjacency island, the problem is not disjoint source topology. At frozen .10, response-averaged local edges cause 21 extra components, but 97.3% of original source triangles are retained in a single connected part. Source action: keep **axis-local response evidence and orthogonal discriminative cuts** distinct from common-motion evidence. Avoid reducing every response channel into one RMS score before generating signed graphic hypotheses. However, axis-specific motion differences may just reflect bending across one real drawable and do not alone authorize any cannot-link.

### Multi-axis response information loss

At original frozen .10:
- CAT: **59/424** shared edges have response distance above .10 on **at least one individual native control axis**; only **28/424** are rejected by pooled RMS. 42 original adjacent-face edges are above threshold on exactly one axis. Eight axes, 168 response channels were already recorded.
- TRICERATOPS: **844/2,001** shared edges exceed .10 on **at least one** of 12 original native axes; only **117/2,001** fail the pooled RMS over all 248 signed/nonlinear channels. **365** source edges are above threshold on exactly one axis.
- Per-axis edge threshold is **not** a calibrated independent `different drawable` test. The 20-channel axis-specific RMS is compared with the same *numerical* .10 threshold for this informational ablation; pooled RMS also includes nonlinear joint-response channels. Their magnitudes and group scales are not independently calibrated, so `844 - 117` is a measure of a strong pooling contrast, NOT a count of true lost drawable boundaries.

A naive OR rule (separate if any axis says different) might turn Triceratops into thousands of artificial fragments, Cat remains bounded below by 260 topology islands, and neither is proven good. This is specifically why **the user's suggestion of parallel cuts** needs an evidence-preserving, multi-channel Graph IR plus **appearance and independent edit consequences**. A raw response-only OR cut is not the solution.

## External priors: role and licensing boundary

- [PartField (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PartField_Learning_3D_Feature_Fields_for_Part_Segmentation_and_Beyond_ICCV_2025_paper.html): continuous class-agnostic 3D part feature field and *hierarchical* clustering; attractive **nonlocal proposal** baseline on original source 3D meshes. The public pretrained README notes the downloadable checkpoint is Objaverse-trained, while the PartNet-enriched checkpoint was not released due to licensing. External weights and model/data licenses require review before training use.
- [PartSAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d6be51e667e0b263e89a23294b57f8cf-Abstract-Conference.html): native 3D part supervision/promptable model, unlike direct SAM mask-lifting; unvalidated on RealSaS original sources and not guaranteed to yield editable drawables.
- [GeoSAM2 (CVPR 2026)](https://github.com/VAST-AI-Research/GeoSAM2): 12-view normal/position-aware 2D SAM2 propagation to source mesh faces. Code released Apache-2.0 and inference checkpoint exists, **training/fine-tuning code is not released**. It still requires a seed 2D mask or click and cannot autonomously recover editor intent; SAM family previously underperformed on our task.

These are three different **optional proposal providers**. None is an accepted source of artist drawable truth or a prerequisite for the core compiler. No model checkpoint was downloaded or trained in V18p.

## Next experiment contract: V18q (bounded)

Before writing a novel neural segmentation model, specify separate evidence and unknowns:

1. `SourceFaceObservationIR` (private original teacher only): stable source face handle, UV/material carrier addresses, source camera sample/ray visible support, poselocal response axis IDs and valid masks, not artist owner.
2. `GraphAffinityEvidenceIR`: near-neighbor geometry edge, lifted cross-island geometry/image continuity, axis-specific response evidence, independent render occlusion and raw material boundary. Each affinity carries typed provenance, confidence, allowed epistemic interpretation, and **no SAME_DRAWABLE labels**.
3. `DrawableHypothesisIR`: uncertain candidate splits/merges and hierarchical source surface support; candidates can span non-neighbor source vertex islands. No hard-coded anatomy name or fixed K. Multiple hypotheses can be retained until edit consequences are run.
4. `IndependentEditCourtIR`: original physical remove/isolate/move operations in *heldout authored source pose/cameras*, source alpha/UV fidelity, overedit and underedit collateral, persistence across views; explicit UNKNOWN. Do not tune candidates on the edit outcome.
5. Baseline compare original V18a, graph-local response evidence preserving axis identity and optional class-agnostic PartField/PartSAM proposals under **the same source court**. A nonlocal merge must be justified with independently measured graphics evidence, not response-only equivalence.

**Stop criterion:** If neither source original physical edits nor a fit-valid appearance identity constraint can distinguish a pair, it stays UNKNOWN. Do not invent source artist grouping to solve a fundamentally nonidentifiable choice. No IRIS/TESSA training or product graphics promotion until actual editor-like independent editability is proven on multiple morphologies and heldout conditions.

PR #343 draft; original source files remain private in Drive; no public original source screenshots or meshes committed.
