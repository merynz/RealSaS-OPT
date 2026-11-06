# Corpus370 V18i — Alpha filtering vs draw-order explanations for V18h failures

Date: 2026-11-05. PR #343, branch `corpus370-v18b-reactive-stain-source-edit-court-v1`.  
**Status: V18i original-source independent court 3/3 PASS, SHA readback verified; NO GRAPHICS OR TRAINING AUTHORITY.**

## Proven evidence — not conjecture

- V18a mechanical reactivity only; original V18a K (Cat 320/288/280, Triceratops 469/22/1 at 0.045/0.10/0.18) is NOT artist drawable truth.
- V18f original native source actual physical triangle edit group responses 416 original Blender frames, 3/3 jobs PASS; all original edit-render geometry and source SHA receipts pinned.
- V18g original lit-source alpha-compositor diagnostic 3/3 jobs PASS: Cat 147/261/261 per-pixel-best two-OVER-order failures over **six distinct fixed groups** per threshold; Triceratops 119/197/0 over *six/one/one* groups; K1 at 0.18 is vacuous, never editability success.
- V18h exact controlled original emission-only, two-color group/remainder, still same **frozen V18a source partitions**, heldout authored pose2, original four cameras, 192px/32 Cycles samples, real physical face exclusion, target 0 only. Actual run [42120479356](https://github.com/merynz/RealSaS-OPT/actions/runs/42120479356) **3/3 PASS**: original Cat court receipt SHA `305a518f238de91903128fb13fcccac5f2179fa944e9281a6a3cbc0f7b756ee2`; Triceratops receipt SHA `6e493f531bc6e2a914b621e15bb8aceb8e3de62f75515a24261ad3a1656d73ff`. These two private source courts independently re-opened actual Blender 40-frame source emission renders each, checked original V18f/V18g independent parents and the original face counts.
- V18h per-frozen **one source target group only**: Cat emitted pixelwise-oracle two-order failures 0/0/0, and corresponding original lit failures 0/0/0; this DOES NOT establish that emission rescued any other Cat group. Triceratops original lit failures 17/197/0 and emitted failures **43/457/0**, while order-discriminating overlap pixel instances 75/2785/0 at thresholds 0.045/0.10/0.18. K1 at 0.18 once again vacuous. Emission deliberately changes artist color/material/illumination, thus no loss of original appearance can be measured with it.

Therefore, **material lighting/shadow coupling alone cannot explain the Triceratops discrepancy**. Removing shading did not remove the two-layer compositing contradiction for the frozen group. Candidate remaining mechanisms include (i) anti-alias filtered alpha of separated components compared with the true joint source, (ii) per-pixel mixing of differently ordered subsample rays, (iii) interleaved depth contributions within a single ray, (iv) residual controlled-render inconsistencies. The current experiment does NOT yet distinguish them.

## Mathematical diagnostic: order-invariant alpha

For original full-source RGBA alpha `a_F` and physically isolated group/remainder alpha `a_G,a_R`, **both** `G OVER R` and `R OVER G` produce the same predicted alpha:

`a_over = a_G + a_R - a_G*a_R`.

Define per-pixel `alpha_error = abs(a_F - a_over)`. If `alpha_error>0.025`, **neither** choice of global nor pixelwise front/back layer ordering can fix that pixel under the two independently filtered alpha images. This is a necessary failure of that specific *postfiltered two-layer OVER* representation, **not** an artist SAME/DIF or source visibility truth.

A counterexample that explains why 2D compositing can fail even when the underlying source is perfectly valid: two **opaque but spatially disjoint subpixel components** each cover half the pixel. Joint rendering gives `a_F=1`. Separate filtered masks give `a_G=a_R=0.5`. Naive postfilter OVER gives `0.5+0.5−0.25=0.75`, although no sample ray is missing. Reverse draw order cannot help. This is a **constructed mathematical falsifier**, not a claim that Triceratops is exactly this case.

V18i thus measures for **all four original views × all three frozen thresholds × one immutable V11 target**:
- `order_invariant_alpha_failure_pixel_instances`: unavoidable alpha mismatch under either order;
- `RGB_or_order_failure_without_alpha_mismatch_pixel_instances`: per-pixel-oracle RGBA failure *not explained* by alpha, where depth/order/appearance requires investigation;
- `inclusion_exclusion_union_bounds_warning_pixels` and group/complement visible support, but treats these as **diagnostics**, not universal constraints on transparent/material-rich source;
- exact recomputation of every `per_pixel_two_order_oracle_failed_pixels` from the original V18h independent sealed court. No renderer rerun.

## Scientific constraints

- Frozen parent [V18h run 42120479356](https://github.com/merynz/RealSaS-OPT/actions/runs/42120479356) and exact original independent courts SHA readback. No changed V18a threshold, chosen group or character.
- V18i code: `tools/corpus/court_corpus370_v18i_emissive_alpha_coverage_diagnostics.py`; synthetic/redteam test including the mathematical half/half opaque coverage counterexample: `tests/corpus/test_corpus370_v18i_emissive_alpha_coverage_diagnostics.py`; workflow: `.github/workflows/corpus370-v18i-source-alpha-coverage-vs-layer-order.yml`.
- Actual V18i final run: https://github.com/merynz/RealSaS-OPT/actions/runs/42123720590 — **3/3 PASS** (contract, Cat, Triceratops). First attempt `42122645313` had Cat source PASS but Triceratops transport stuck in Ubuntu apt installation. Replaced only transport installation by the official rclone v1.75.2 linux-amd64 ZIP with **static SHA-256** `349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702`, retaining exactly the original V18h SHA parents and mathematical court. First run cancelled by GitHub branch workflow concurrency after corrected run began; no scientific acceptance was based on this operational restart.

Final independent V18i private output SHA-256 readback:
- Cat: `763c4bfebe78cc04dc37cd48aa1258b2b2ddffd0b222a158fc8fedf7d0d575d0`
- Triceratops: `36b1bb856c24fa204dbb1cd3cef0a0cbbdfb2cc7fd5b8b31f59cd22286883bd2`

### Actual source outcome: breakdown of the 2-layer pixelwise-oracle error

Same immutable V18a group containing V11 target ordinal 0; exact four original camera views summed. Counts are repeated *view/pixel instances*, **not artist region counts**.

| Source | V18a threshold | Full original groups K | This probe's faces | Emissive pixelwise-oracle failures | Alpha mismatch alone guarantees failure | Residual RGB/depth/coverage failure with alpha match |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CAT | .045 | 320 | 2 | 0 | 0 | 0 |
| CAT | .10 | 288 | 2 | 0 | 0 | 0 |
| CAT | .18 | 280 | 2 | 0 | 0 | 0 |
| TRICERATOPS | .045 | 469 | 1 | 43 | **17** | **26** |
| TRICERATOPS | .10 | 22 | 1298 | 457 | **172** | **285** |
| TRICERATOPS | .18 | 1 | 1334 | 0 | 0 | 0 |

For Triceratops .10, **172/457 (37.6%)** oracle failure pixels violate the order-invariant alpha identity `aF == aG+aR-aG*aR`; no choice of which image to draw on top can repair these under the two-layer postfiltered representation. The other **285/457 (62.4%)** have alpha identity within tolerance, but neither G-over-R nor R-over-G reconstructs premultiplied RGB under the original straight alpha assumption. This latter set does NOT independently prove reversed ordering, nor rule out alpha/raster transport subtleties: it is only a residual category.

The original V18h grouping at .18 makes the complement physically empty, so **0 error is vacuous** and must NEVER be promoted as a successful drawable decomposition. Cat's original group was already zero-error in the lit source; the Cat result is not evidence that emissive materials fixed its other groups.

**Scientific implications:** Both alpha/coverage representation and visibility/order/appearance treatment require investigation. No single `REPEL` or `ATTRACT` source-face edge may be inferred from these pixel counts alone. The next decisive test is **source-posed fixed 4x4 subpixel geometric ray attribution** of known alpha-failing, RGB-only failing, and passing pixels using existing V17b tools, with per-ray depth tie/incomplete states explicitly UNKNOWN. The original `failed/pass` image stratum is retrospective/outcome-based and thus serves only explanatory diagnosis; it cannot be presented as a blind heldout truth benchmark.
- Source RGBA/teacher-only identities retained in private Drive only; exact source emission/private files receive hash-checked readback. Explanations are relative to an assumed straight-RGB→premultiplied conversion used by V18g/V18h, not source artist RGB truth.
- No original artist drawable labels, negative `REPEL` relation, model-training or materialization admission. ALL `graphics_owner_authorized`, `iris_training_authorized`, `tessa_training_authorized`, `training_authorized`, `materialization_authorized` are **false**.

## Next decisive experiment, conditional on V18i results

**If alpha mismatch explains most order-oracle failures:** V17b already provides source-consistent 4×4 subpixel orthographic ray infrastructure. Probe a predeclared, balanced set of V18i alpha-failing, RGB-only failing and passing pixels on the exact source **posed** mesh, classify source-group/other source-face coverage, unknown ray/tie and opposing per-ray nearest hit order. This is a retrospective explanatory test (pixel selection is outcome-based), NOT a heldout truth benchmark. Require independent source-ray recomputation and no artist identity leakage. An eventual alpha-aware renderer must compute compositing before raster filtering or preserve per-ray/subsample K-hit evidence, not naively composite separately filtered source parts.

**If RGB-only remains dominant:** investigate view-dependent ordered ray stacks and group/rest interleaving before assuming a semantic slot representation or marking different groups as incompatible.

Any StdGEN-inspired multi-layer anonymous slots remain **conditional representation research**, not a solution delivered by renaming their fixed human semantic labels. V18a/Mutex Watershed/Lifted Multicut may propose groups only after **legitimate operational** signed edge evidence.

Product north star: source-faithful 3D mechanics + independently editable 2D presentation in a generic puppet. Preserve distinct Graphics, Mechanics, Geometry and Relations truth families and their UNKNOWN masks.
