# Amendment A1: Blender Image.pixels bottom-left scanline origin in Cat V3 source court

Date 2026-11-06. **POST-HOC DIAGNOSTIC AMENDMENT**, not the original 0.025 preregistration, not an unseen heldout result. Stacked PR #359 on #358. Preserve [the original uncorrected V0 physical result](https://github.com/merynz/RealSaS-OPT/actions/runs/42285528834) and original private report SHA `90759dd7f431b89ccc697824a9f9485fa2cdef15360a89746d0978dbd1ae50b4`.

## Provenance of mismatch discovered AFTER first measurement

The first Cat puppet physical comparison compiled 32 source-teacher reference CAA frames and independently compared them to all 64 original Cat V3 Cycles samples at 0.025 linear premultiplied RGBA tolerance. Its uncorrected result: 1,004,587 / 2,359,296 mismatched sample-operation pixels, maximum error 1.0, and 38,928 K4 overflow pixels across 32 frames. Code execution succeeded, but scientific parity was `VIOLATED_WITH_UNQUALIFIED_VISIBILITY_SUPPORT`.

**Newly discovered interface problem:** Original V3 Blender physical-source recorder uses `bpy.data.images.load(OpenEXR).pixels.foreach_get(...)` then `reshape(height,width,4)` without a vertical row-origin declaration. Blender `bpy.types.Image.pixels` buffers begin at **bottom-left**, whereas the RealSaS `render_caa_reference` screen rows begin **top-left**. A post-hoc diagnostic over **BASE-only** examples found that vertical row reversal dramatically increased silhouette agreement (about 0.993–0.998 IoU in source views), implicating pixel-array serialization rather than a proved generic drawable split problem. This result was observed after V0, so it is not being mislabeled a pre-registered discovery.

Independent published descriptions of Blender's bottom-left `Image.pixels` contract:
- https://tmhsdigital.github.io/Blender-Developer-Tools/gallery/image-pixels-testcard/
- https://blenderscripting.blogspot.com/2012/08/adjusting-image-pixels-walkthrough.html

## Frozen correction and tests before full edited-target re-court

1. Before source data is loaded, confirm the exact Blender v4.5.14 `Image.pixels` bottom-left scanline behavior using a **synthetic multicolor 2x2 PNG marker** saved by Blender and decoded by an independent Pillow host reader, plus a `foreach_get` readback. No original Cat art or physical targets enter this marker.
2. The corrected original physical source readback is `source_original_rgba[..., ::-1, :, :]` along **the height axis only** (bottom-left buffer rows converted to top-left CAA array rows). This is a fixed full-frame index reversal applied identically to all 2 original source poses, 4 cameras, both Cycles offsets, and all four BASE/SINGLE arms. **No coordinate fitting, image warping, texture resampling, per-camera movement, gamma, alpha or threshold change**.
3. Existing compiled puppet reference render NPZ is immutable: SHA `55675886ae92c3b122647b1749290675dc8fbc13d1c617c9340ec42fe3e702d8`. Original physical source NPZ is immutable: SHA `b2bb4fa9fc0b777bb8418aebc729123616b774b55eeb215ba17651e7f3448e0d`. No new Blender Cat source render.
4. After fixed origin conversion, compare all **64** physical target sample images to the **same 32** reference CAA frames. For each pose/view/offset/arm compare every channel at the original precommitted **0.025** maximum premultiplied-linear-RGBA threshold. Report all mismatches, maximum-channel error, per-operation breakdown and private first witness. **No masking of border pixels or K4 overflow.**
5. Preserve K4 overflow, exact-depth tie and incomplete alpha/source domain as `UNKNOWN` obligations. Numerical parity alone cannot authorize `SATISFIED` where required visibility is unknown.
6. Since BASE physical targets were inspected to identify the origin mismatch, present them as **diagnostic/reanalysis**. The three SINGLE edit arms are an *uninspected edited-condition check of the origin correction*, not a fully independent new source dataset.
7. A new corrected court status can be measured; it must not overwrite or retroactively convert the historical V0 `VIOLATED` result. This is an admitted **reader-contract correction** scope, not retrospective tuning of the graphics model.

## Qualification boundaries

No source artist ownership, no redrawable partition necessity, no student one-image graphics inference, no full BUSTED product runtime parity, no heldout morphology, no training/materialization authorization, and no measured GPU/CPU efficiency. All source/teacher private bytes remain off GitHub. Main and Corpus370/IRIS/TESSA unchanged.
