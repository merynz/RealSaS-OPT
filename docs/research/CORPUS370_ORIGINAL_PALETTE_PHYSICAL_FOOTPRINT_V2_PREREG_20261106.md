# Corpus370 physical palette footprint court V2 — frozen follow-up before execution

2026-11-06. Research-only; extends PR #351. **All previous V1 failures retained.**

## Why a NEW version is necessary

The original physical palette edit V1 test on original Triceratops source rendered four views V0/V2/V4/V6 at 256 px, original unlit-alpha policy, Cycles CPU 16 fixed samples. Its independent court used the source Blender first-hit ray at a pixel center with a 4-direction 2-pixel cardinal margin and treated that as uniform pixel color ownership. This was a FALSE assumption: V6 x166 y134 was labelled native palette 1 by center and cardinal rays, but real edit of palette 2 changed its RGB. V1 properly FAILed (one collateral sample, zero alpha change) and was NOT converted to PASS.

A second source-private analytic cross-check with the ORIGINAL SHA-bound rest triangles, exact original camera and independent per-face palette mapping showed the same center/cardinal palette 1 hits AND a palette 2 foreground hit at diagonal subpixel offsets (e.g. dx=-0.25,dy=+0.25). **This falsifies the inference from cardinal-source-ray agreement to uniform raster-pixel coverage**, without making a claim that exact Blender Cycles sample positions were reconstructed.

## Frozen V2 policy — preregistered before NEW camera edit render execution

- Representation and scientific target unchanged: original rest-pose source, exact qualified constant-source-color palette groups, fixed original source shade-to-emission unlit policy, individual RGB recolor, no geometry editing, no artist ownership.
- NEW evaluation camera directions **V1, V3, V5, V7 only**. These directions were present in the previously existing 8-camera original source bundle, but **no physical palette edit results for these directions have been inspected in the V1 test**. They are held out relative to this V2 edit-consequence trial, NOT unseen images globally.
- Source input identity must be original Triceratops FBX SHA ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a, and face order and source material facts must be independently SHA-bound. No threshold retuning, material/object-name grouping or model training.
- Renderer resolution remains 256x256, Cycles CPU 16 samples, seed 0, original admitted source unlit-alpha replay, transparent film. All three color edit groups, one at a time, same frozen channel rule: channel >=0.5 =>0.15; otherwise 0.85. Alpha retained.
- Independent reference is **not Blender scene.ray_cast**: analytically intersect 3D rays with the original source rest-world triangles via direct Möller–Trumbore and pick the first depth-valid source triangle, preserving exact original per-face palette index. All source face geometry comes from the separate source-rest NPZ sealed before V18.
- Pixel-support domain: grid stride **4 image pixels**; for every pixel center with foreground alpha >=0.995, evaluate a **3x3 ray stencil** at x,y offsets (-0.375,0,+0.375) pixels. Accept the pixel as a *stencil-uniform interior sample* only if **all 9 rays hit a known same source palette key**. Unknown or mixed stencil => UNKNOWN, NOT an unselected palette negative.
- Frozen observed-pixel thresholds: per-edit selected interior max-RGB change >=8/255, collateral/unselected interior max-RGB change <=3/255, global original baseline-vs-edit alpha max absolute difference <=2/255, at least 8 selected and 8 unselected qualified interior samples accumulated across V1/V3/V5/V7 for EACH of 3 palette colors. These are bounded pixel-consequence tests, not arbitrary full-raster or silhouette truth.
- A PASS qualifies ONLY this particular source, rest pose, exact constant shader palette edit policy and 3x3 **sampled** physical edit locality at the measured resolution and four newly tested directions. It does NOT prove exact continuous pixel-footprint coverage or native artist drawable identity.
- A FAIL stays FAIL and its original source geometry, camera, source-color membership, observed pixel deltas and SHA-sealed private imagery must be retained. No post-hoc thresholds or selection edits based on these new views.

## Explicit open items

Original raw lit shading parity, transparent linked shaders, UV/textured source edits, arbitrary compositor alpha, all four source *poses*, pose2 edited consequence, collision, attachment, secondary motion, true graphics owner, unseen morphology, 1-view IRIS student targets, shared truth matrix training readiness.

**All graphics_owner_authorized, artist_owner_authorized, iris_training_authorized, tessa_training_authorized, materialization_authorized, training_authorized remain FALSE regardless of V2 outcome.**
