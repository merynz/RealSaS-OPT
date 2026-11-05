# Corpus370 V18l — exact Blender Cycles sample-indexed three-arm source canary

Date: 2026-11-06, Türkiye. **REAL ORIGINAL SOURCE 3/3 INDEPENDENT COURT PASS; AUTHOR EDITABLE LAYER TRUTH UNKNOWN**. PR #343 draft; main not merged.

## Why this is the necessary next experiment

Original V18k 4×4 geometric rays independently demonstrated correlated subpixel surface coverage / mixed nearest original source faces, but a manually chosen 4×4 ray grid does NOT match Blender's 32spp Cycles film sampler. On frozen Triceratops threshold .10 locked cameras, raw common geometric joint-first estimate locally improved 15/16 previously failing Blender postfilter pixels, yet was worse on 131/149 preselected pixels and rescued only 3/16 into .025 tolerance; 32 vs 16 full-domain source failure pixels. This **disqualifies** the V18k geometric 4×4 estimate as production renderer, and motivates the source-matched sample-index canary instead of more arbitrary segmentation thresholds.

## V18l fixed protocol (before source render)

Same immutable original source Cat/Triceratops, same V14 original 3D posed teacher, heldout native pose2, four original source calibrated cameras `0,2,4,6`, same exact V18a source-face partition **threshold 0.10** and V11 target ordinal zero, no new labels. Exactly SHA-pinned original V18a/V18f/V18g and V18h independent source parent courts.

Use original Blender 4.5.14 Cycles binary, SHA-256 verified from official release. Cycles setup: `samples=32`, `seed=0`, `use_adaptive_sampling=False`, `use_sample_subset=True`, `sample_subset_length=1`, `sample_offset ∈ [0,4,8,12,16,20,24,28]`, `pixel_filter_type='BOX'`, `filter_width=0.01`, no denoising per inherited renderer. Transparent film, fixed opaque **red G / blue R emission** materials only; no artist shader/texture/UV truth obtained.

For every original camera × every fixed Cycles subset offset, physically render on the *same source original posed geometry* (not opaque masks or 2D geometry simulation):
1. `FULL` original 3D source soup (group + remainder);
2. `ISOLATE` only source faces in frozen V18a component proposal;
3. `REMOVE` physically remove group, retain complement.
Source face removal operators were independently proven in V18e/f; this canary creates no new artist-defined edit labels.

8 offsets × 3 physical arms × 4 source cameras = **96 original Blender render calls per source**. Each individual render is one selected Cycles sample; 8 indexed sample results are not the full 32spp film and the narrow BOX filter differs from the historical V18h. No claim that 8 subset outputs reproduce the historical 32spp full source picture.

## Independent court and falsification

Independent source court reopens raw 4×8×3×192×192×4 original source RGBA and original V18h court SHA. Recompute source straight-to-premultiplied RGBA. Per single Cycles sample index, evaluate both `G OVER R` and `R OVER G`; count image pixels for which neither reconstructs the original `FULL` sample within fixed .025 tolerance. Also measure order-invariant alpha identity `alpha_FULL = alpha_G + alpha_R - alpha_G alpha_R`.

**Identical `sample_offset` and seed alone do not prove matching pixel rays after changing the geometry.** V18l *tests*, rather than assumes, that per-sample source FULL and either physical separate-layer order agree. If sample-index alignment breaks (or an unexpected filter/alpha policy contributes), the discrepancy must remain a **measured failure**, never papered over by masks, min/max clamps or artist part labels.

Compare two operations on the *same set of eight Cycles samples*:
- **Pre-filter sample compositing:** `mean_i(G_i OVER R_i)` or `mean_i(R_i OVER G_i)` (fixed order); also report per-index pixelwise best-of-two oracle as a **non-product diagnostic**;
- **Post-filter layer compositing:** `mean_i(G_i) OVER mean_i(R_i)` and reverse;
- Reference: `mean_i(FULL_i)` **from these identical source-subset conditions**.

The source FULL target is real Cycles under this exact frozen subset + BOX policy, **not the 32spp original artist appearance target**. Report every 192×192 pixel for the four cameras; don't only report pixels where a proposed method succeeds, and don't count K1 whole-character candidates as editable success. Compare the locked cameras 4/6 separately from exploratory cameras 0/2. Preserve raw array SHA and independent court SHA readback in private Drive.

**Non-vacuity gate, fixed before source results:** At least one originally rendered source FULL pixel within each nontrivial source must exhibit a max inter-subset premultiplied RGBA difference above `.025` (8 distinct source subset indices). If none does, the sample-index test is *vacuously consistent*: it may only have replayed the same pixel-center visibility, and cannot establish anti-alias/common-sampling effects. The source court must count this explicitly for original Cat and Triceratops; keep any absence a fail-closed scientific result rather than assigning negative graphics relations.

Possible outcomes:
- Sample-best and fixed pre-filter both reproduce the original source subset while postfilter fails: **controlled proof of raster operation order under one common Cycles sampling domain**, not arbitrary art layer ownership.
- Per-index best fails: source pixel sample alignment / shader sampling / face deletion mechanism must be diagnosed before claiming a fixed common-domain reference.
- Fixed pre-filter fails but per-sample best passes: within-pixel source depth order varies, so a static two-layer total order is insufficient; requires pixel/sample-depth-aware partial order or K-hit.
- Full sample references fail against original lit/32spp: this is **expected distribution mismatch**; V18l explicitly does not test source artist appearance.

Any result retains `graphics_owner_authorized=false`, `iris_training_authorized=false`, `tessa_training_authorized=false`, `training_authorized=false`, `materialization_authorized=false`. A valid graphics layer compiler must eventually preserve original material/CAA, establish edit/reveal consequences, handle confidence/occlusion and produce permutation-invariant variable-cardinality anonymous components without exporting teacher source names.

### Implementation
- `tools/truth_foundry/blender_corpus370_v18l_cycles_common_sample_source.py`
- `tools/corpus/court_corpus370_v18l_cycles_common_sample_source.py`
- `tests/corpus/test_corpus370_v18l_cycles_common_sample_source.py`
- `tools/corpus/hosted_corpus370_v18l_cycles_common_sample_source.sh`
- `.github/workflows/corpus370-v18l-original-cycles-common-sample.yml`
- Real run: https://github.com/merynz/RealSaS-OPT/actions/runs/42131579613

Blender 4.5 public RNA/Film references:
- https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/sampling.html
- https://docs.blender.org/manual/en/4.5/render/cycles/render_settings/film.html
- https://github.com/blender/blender/blob/main/intern/cycles/blender/addon/properties.py

## Observed exact original source Cycles result — 3/3 PASS

Run `42131579613-1`: https://github.com/merynz/RealSaS-OPT/actions/runs/42131579613 . Real Blender 4.5.14 native API smoke PASS; mathematical/adversarial 9/9 PASS; original Cat and Triceratops worker independent source courts each PASS. Both source archives were SHA-readback sealed in private Drive. Source courts:
- CAT: `cce24e65850308ed0c27bcb63f33ec060ae7242fa57b27f4c82dc075e8be9a17`.
- TRICERATOPS: `3245c11e49bc3e39b3209eda2ff771b39ec0269b7f66c45949476d8cb557333d`.

Each of the four original calibrated source views has 8 Cycles subset indices × 192×192 image pixels = **294,912 per-index pixel instances**, or **1,179,648 per source**. **0 same-sample alpha/necessary two-order reconstruction failures in both characters**, and no raw 1spp intermediate alpha values under fixed BOX + emission. The successful comparison is against the actual source `FULL_i` rendered at the **same subset index**, not the V18h historical 32spp Blackman-Harris film nor the source's original painted appearance.

| Original source | Camera | Actual 8-subset postfiltered G/R best-order failures | Actual 8-subset sample-first best fixed-order failures | Source FULL pixels visibly different between subset indices (source-private separate raw reanalysis) |
| --- | ---: | ---: | ---: | ---: |
| CAT | 0 | 0 | 0 | 427 |
| CAT | 2 | 0 | 0 | 509 |
| CAT | 4 | 0 | 0 | 427 |
| CAT | 6 | 0 | 0 | 533 |
| TRICERATOPS | 0 | 7 | 0 | 200 |
| TRICERATOPS | 2 | 64 | 7 | 542 |
| TRICERATOPS | 4 | 77 | 70 | 290 |
| TRICERATOPS | 6 | 68 | 8 | 545 |
| **TRICERATOPS total** | **four cameras** | **216** | **85** | **1,577 source pixel-view occurrences** |

On Triceratops the sample-first **fixed-order** method is closer to its *actual same-index source 8-sample mean* on **139 original image pixels**, strictly worse on 0. It corrects (to tolerance 0.025) **131 pixels** that previously failed postfilter OVER; 85 pixels remain unresolved by either fixed layer order at the source-sample aggregation stage. **Every single source Cycles sample FULL is nevertheless explainable by choosing the correct local G/R compositing order**, so the unresolved 85 are specifically a *fixed order across multiple sample rays* limitation, not arbitrary missing artist body segmentation proof.

The important controlled causal claim is **not** that V18l made a product renderer. Within exact source-matched eight-sample Cycles data, non-commuting per-sample color/alpha visibility combination and postfiltered image combination explain observed compositing discrepancies; a single global layer order remains insufficient at some pixels. The per-index best-of-two oracle is **not an implemented depth-driven scheduling policy** and carries privileged source outcomes. Unseen source morphology, real source textures, CAA, motion and heldout independently editable graphics are still unqualified.

A *separate* nonvacuity source court on these same exact original rendered bytes is required to reproduce sample diversity counts and paired pixel class improvements and regressions, with an explicit fail-closed anti-vacuity gate. Code: `tools/corpus/court_corpus370_v18l_source_sample_diversity.py`; workflow: `.github/workflows/corpus370-v18l-source-index-nonvacuity-court.yml`. Exact original-run SHA outputs are inputs; do not count this as independent new source collection.

## Revised next decision

1. Source-matched sample compositing is mechanically and numerically plausible and now independently measured in original Blender source data. Preserve `source face → sample ray hit → local visibility → coverage → premultiplied RGBA` as the **Graphics Evidence IR continuation**, with guaranteed source-validity masks.
2. **Do not infer global source drawable part identity from these tests.** Graphics ownership should be determined from operational editable consequences; face membership K=1 or huge K near-whole-character remains unqualified.
3. Explore a `visibility and sample-local ordering` renderer / partial-order program that can predict the correct local order from original geometry, not oracle picking whichever image color matches FULL. Apply to **original appearance authority / textures/UV and heldout poses** before compile qualification.
4. On equally controlled original source faces, compare an anonymous variable-K surface-proposal compiler (geometry/mechanics-inspired, optional StdGEN/Bunraku-like modular heads as baseline) using original heldout edit fidelity; never assign hair/body or animal semantic fixed target labels.
5. No model/training/materialization gate advanced; PR #343 remains draft, main unchanged.


