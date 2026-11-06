# Cat V3 native UV → reference puppet physical-parity preregistration (conditional)

Date: 2026-11-06; research stack #351 → #354 → #355 → #356 → #357. **This is a preregistered future measurement, NOT a completed run.** Execution is blocked until original GLB-to-teacher face-corner UV binding has an independently verified witness; no guessed importer name/face remap permitted.

## Inputs and authority separation

- **Source teacher-only:** original Cat GLB SHA `1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850`; source original PNG SHA `2f6d0032e51c2d49b13b8e689d59eb39aa3ee6c32f2a1851213e11fc7593177d`; native teacher SHA `c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603`.
- **Source oracle, withheld from candidate generator:** V3 original physical 64 Cycles sample array SHA `b2bb4fa9fc0b777bb8418aebc729123616b774b55eeb215ba17651e7f3448e0d`; independent V3 source court SHA `0171e9ec836d7b3c44ec64abcc59f712ebd33a87c2ad4edf6188976dc08e272c`.
- Source editor uses the three **anonymous** original top-left UV atlas 64x64 RGB-invert rectangles and leaves straight alpha unchanged. This proves neither three drawable layers nor source artist ownership.
- Reference puppet consumes only original source mesh/posed vertices, face-corner UV, unmodified source atlas, original physical camera parameters, edit rectangles and **required operations**; it does not read source target pixels until after compile+render.
- Candidate packettings share one depth/UV/CAA reference implementation; any parity demonstrates only reference rendering, *not* independently implemented drawable alternatives or a deployable BUSTED GPU/runtime path.

## Frozen experiment before source comparison

- Original source authored teacher poses: ordinals **0 and 14** only; views **V0,V2,V4,V6**, source Cycles offsets **0 and 16**, four original arms **BASE,SINGLE_0,SINGLE_1,SINGLE_2**. Original physically rendered array shape `(2,4,2,4,192,192,4)`.
- Use source-world posed XYZ as canonical face-vertex position input with unchanged teacher faces and proven corner UVs. Use source camera basis (origin/right/screen_up/forward/half_extent), orthographic 192px screen, **no fitted camera or per-frame warp**.
- Use current `OperationalPuppetRuntimeV0` → `render_caa_reference` as-is: 2x2 coverage samples, K4 depth layers, premultiplied linear RGBA, source-over alpha, the verified top-left atlas convention. Do not retune render parameters or threshold after measuring discrepancies.
- Emit one predicted render for each pose/view/arm: **32** deterministic reference frames. Compare each prediction independently to **both** original source Cycles sample offsets: **64** physical target comparisons.
- For original Cycles samples, convert recorded straight linear RGB to premultiplied `[RGB*A, A]` as in the independently established V3 court. For compiled predictions, use the CAA renderer's premultiplied output; **do not** independently color-correct/rescale against oracle targets.
- Compare all pixels and all channels: max-channel absolute error per sample-pixel; frozen threshold **0.025** premultiplied RGBA. Publish aggregate failure counts, per-pose/view/arm/offset breakdown, maximum-channel difference, and source-private first counterexample.
- Separately record per-frame K4 layer overflow, exact-depth ambiguity and visibility-coverage counts. If any required support is unqualified, mark family/candidate `UNKNOWN` regardless of numerical closeness. Unknown pixels are **not** silently excluded from the full-image mismatch count.
- Expected source V3 sample stochasticity and CPU-vs-Cycles pixel filter differences are legitimate potential falsification/implementation mismatch causes, not a license to reselect views or mask edge pixels.

## Pass/fail interpretation

**SATISFIED** only for declared *source-backed teacher-mode reference-render scope* if every required original physical comparison is within tolerance, all source address/geometry SHA gates pass, and zero unqualified visibility/compositing support. **VIOLATED** for a measured comparator exceedance in admitted scope; retain its pixel- and operation-level witness. **UNKNOWN** for missing source binding/artifacts, unqualified raster support, or unavailable required operations. Preserve distinct reason fields; a violation and an unknown may coexist.

No source artist ownership; no full puppet/attachment release/secondary motion, heldout morphology or one-image student inference; no training/materialization authority. Cost remains `UNMEASURED_COST`.

## Safe execution order

1. Finish original glTF primitive/scene UV inventory ↔ Blender-native v4 teacher face order + rest geometry alignment, using immutable original bytes.
2. Independently re-open original V3 physical source array and court receipts; all SHA joins exact.
3. Compile candidate puppet with **no physical target arrays in inputs**; write predicted arrays and sealed runtime receipt privately.
4. Separate independent CPU court reads predictions plus physically rendered source arrays; computes all 64 comparisons and UNKNOWN gates.
5. Only after PASS consider Triceratops local-depth and different-morphology qualification. No integration into `main`, Corpus370 admission, IRIS/TESSA or product stage from this research PR.
