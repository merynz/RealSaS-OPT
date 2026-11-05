# Corpus370 V18g — compositing consequence falsification of physical graphics groups

Date: 2026-11-05 (UTC). Status: **RESEARCH / RETROSPECTIVE SOURCE DIAGNOSTIC; NOT NEW DRAWABLE TRAINING TRUTH**.  
Branch `corpus370-v18b-reactive-stain-source-edit-court-v1`, PR #343.  
V18g run: `42116712393`, https://github.com/merynz/RealSaS-OPT/actions/runs/42116712393 . At initial commit the two original-source jobs are underway; this document deliberately does NOT assume their independent outcomes.

## Origin and exact previous measurements

- V18a original SHA-pinned 3-source mechanical perturbation proposal (`42097722762-1`); **no artist graphics-owner truth**. V18a thresholds `0.045/0.10/0.18`; Cat group counts `320/288/280`, Triceratops `469/22/1`, Cowfish `336/67/17`. Mechanical similarity and union-find merging produced legitimate research candidates but not drawable layers.
- V18d original-source instrument failures: Cat and Triceratops transparent shader removed faces incorrectly due Cycles `transparent_max_bounces` exhaustion; both independently falsified and repaired under physical hiding / 256 transparency bounces (run `42106976442-1`).
- V18e true physical mesh BMesh-deletion canary: exact original 2-source run `42111807283-1`, 3/3 PASS, 8 arms/view, four original camera views, original pose2. Both source/soup parity and source empty alpha PASS, exact independent original-source face census. No owner/training authority.
- V18f full physically removed group original-source suite run `42114401233-1`, **3/3 SUCCESS**, **416 actual Blender render arms**: 208 each Cat + Triceratops, 4 original cameras, 6 fixed V11 target queries × 3 frozen V18a thresholds. Both real independent courts PASS and original saved source RGBA replay SHA readback:
    - Cat `9fba1f6f74ff5fd9a60ffb842f49ff8ba0a4169e2c8cf3ed504bec0374c26276` (court receipt SHA), 14/24 observable fixed single-triangle proxy cases. For thresholds `0.045/0.10/0.18` group-isolate effect pixel instances 7,095 / 8,819 / 8,819; group-remove effect pixel instances 199 / 199 / 199. No unsupported artist labels.
    - Triceratops `6c578d92f8f7bc4d867632afd629d989b79f045f38a8edead4e75ab14b1b5f38` (court receipt SHA), 24/24 observable fixed single-triangle proxy cases. For thresholds `0.045/0.10/0.18` group-isolate effect pixel instances 6,723 / 94,572 / 95,736; group-remove effect pixel instances 1,726 / 88,488 / 95,736. Strong evidence that 0.10/0.18 proposals change extensive presentation on the original source; **not a proof that those edits violate any artist-intended layer boundary**.
    - Every candidate's single-face reference effect was contained in its containing group's effect (zero missing proxy pixels), a natural nested-set consequence, **not independent validation of drawable correctness**.
- For both sources V18f independent court: true source vs V14/true soup/no-op, physically empty soup, exact face counts per operation qualified. This qualifies the **instrument** for retrospective source-compositing analysis, not the candidate owner.

## V18g research question

For a fixed source-derived anonymous candidate group G and the actual physical complement R, at a fixed original source camera and pose, does standard alpha compositing from isolated G / isolated R reconstruct the original full source image? This tests part of the consumer presentation contract: not merely mechanical similarity or visually plausible clusters, but actual separation/recomposition in screen-space.

Consider source rendered straight linear RGBA `S`, physically isolated candidate `G`, and physically rendered complement `R`. Evaluate *premultiplied* Porter-Duff:
`G OVER R = (Cg*ag + Cr*ar*(1-ag), ag+ar*(1-ag))`
`R OVER G = (Cr*ar + Cg*ag*(1-ar), ar+ag*(1-ar))`
against premultiplied `S` at original immutable tolerance `0.025`. Also evaluate `min(pixel_error(G OVER R), pixel_error(R OVER G))` **as an oracle diagnostic only**. This pixelwise oracle is **not** a compiler policy, student input, model truth, or globally consistent layer order.

Require **actual common alpha support AND over/under predictions differing at a pixel** before interpreting order. If no overlap, K=1, fully invisible complement or a trivially covered whole character, there is *no empirical draw-order evidence*. If neither order reconstructs a pixel, this is a counterexample to this **two-layer premultiplied-alpha reconstruction under the exact source rendering conditions**, not necessarily a guarantee that the underlying partition is wrong; lighting, transparency, shadows, mixed-depth geometry and artist 2D compositing models can differ.

Because the same V18a group can be probed by multiple frozen V11 targets, V18g deduplicates by `(threshold, view, original group id)` and strictly requires duplicate renders to have identical results. Otherwise an apparent aggregate could double-count a single group or mask temporal ambiguity.

## Original source authority, frozen before run

- V18f original run `42114401233-1` **and exact independent court SHA** supplied as a required argument and checked against the downloaded original independent court.
- V18f original `PRIVATE_ORIGINAL_V18F_FULL_PHYSICAL_EDIT_RENDER.npz/json` SHA-bound to the parent court; V18a `PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz` + exact V18a original independent source court SHA.
- **No Blender required**. Read-only source pixel archives are opened on GitHub Actions host; no remote code can replace a court without invalidating the frozen hash.
- Source and camera identity, 4 heldout original views, 52 arms, 72 prior interventions, and source-face IDs exactly inherited from qualified V18f rather than reoptimized.
- Every read rejects source/original/NPZ schema mismatches, invalid alpha, nonfinite RGBA, changed original source/face group indices, missing result, and forbidden authority grants.
- Only private aggregate JSON with source-private hashes is stored in the pinned Drive experiment folder and SHA-readback checked.

## Explicit outputs and allowed interpretation

Per view/target/threshold: group and remainder alpha support, overlapping/strictly order-discriminating pixel counts; both fixed-layer orders' premultiplied residuals, pixelwise-oracle residual; how often only one order is supported, both orders fail, or both pass. Unique source group/view summary per threshold is required to avoid multiplying repeated targets.

**Only hypotheses produced**:
- If both layer orders fail even with pixelwise oracle: two-layer alpha compositing is not sufficient under this source intervention, camera, physical geometry and material setup.
- If pixelwise reconstruction works but no constant global order works: potential view/pixel-dependent partial draw order; stronger source-ray K-hit evidence required.
- If no overlap and no order-discriminating pixels: **ABSTAIN / UNKNOWN**, regardless of good reconstruction.
- If group is all faces or complement invisible: non-identifying, never positive drawable proof.
- Even if reconstruction passes at all measured pixels, **editable local boundaries, alpha holds, actual source appearance authority, and persistent identity across other poses are still unproven**.

**Never mint** `artist_same/different_drawable`, `graphics_owner_authorized`, `iris_training_authorized`, `tessa_training_authorized`, `materialization_authorized`, or `training_authorized`. When conflicting possible partitions are indistinguishable, keep equivalent candidates or abstain; do not infer a global artist class.

## Code and evidence

- `tools/corpus/court_corpus370_v18g_operational_compositing_diagnostics.py`
- `tests/corpus/test_corpus370_v18g_operational_compositing_diagnostics.py`
- `.github/workflows/corpus370-v18g-original-physical-compositing-diagnostics.yml`

V18g's diagnostic builds the **evidence side** of a future signed/partial-order Graphics compiler representation; it does not yet fit Mutex Watershed or Lifted Multicut. Those optimizers are downstream of valid positive/negative edit-relation evidence and must be compared under a separately heldout source consequence court.

Main, original V18a/V18b/V18e/f artifacts, Corpus370 #327 and the IRIS/TESSA head allocation remain unchanged.
