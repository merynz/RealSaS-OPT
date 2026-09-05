# RealSaS Living Compile V4

This is the recovered-and-promoted RealSaS product UI lineage from the V18.86/V18.95 Living Compile shell, rebound to the current V4 product contracts.

## Current inputs

The UI opens an E2E output root containing:

- `puppet/canonical_puppet_graph_v3.json`
- `proof/product_proof_bundle_ir.json`
- `proof/motion_bakes/*.json` for exact proof-owned motion preview
- bundle-local directional textures, preferably `native_texture_source/textures/view_0.png` ... `view_7.png`
- optional `native/realsas_runtime.rss`

Run:

```bash
python -m product.living_compile.server --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`, choose **Character input**, and enter the E2E output directory in **Existing canonical bundle**.

## Editor surface

Recovered V18.95 capabilities:

- rig pose rotation/translation
- mesh vertex edits and local topology validation
- weight add/subtract/smooth brush
- visible binding reassignment

Promoted V4 authoring additions:

- bone add/delete/reparent user topology edits
- IK constraint authoring
- animation clip creation/override
- dope-sheet keyframes
- linear / step / cubic-smooth interpolation
- preview playback of user-authored keys

All edits are written to a sibling directory `<bundle>.user/layers/`. The canonical bundle bytes are not changed. `RealSaS.UserPuppetEditLayer.v3` has `can_promote_product_directly=false`, `requires_compiler_requalification=true`, and `deployment_status=BLOCKED_PENDING_DYNAMIC_REVALIDATION`.

## Authority rules

- Browser/server never use mechanical `P.xy` as directional raster position.
- Rest mesh display comes from qualified surface support -> admitted raster bindings.
- Runtime preview consumes qualification-owned motion bake frames. It does not run a second motion solver.
- A non-PASS ProductProofBundle cannot launch runtime preview.
- User authoring is never Compiler truth until a future recompile/requalification step accepts it.

## Deliberately not promoted yet

The **compile-from-UI** buttons are fail-closed because the current scientific one-family optimizer is still externally orchestrated. Opening a completed V4 E2E output is supported now; direct UI ownership of training/fit jobs will be promoted only after the optimizer itself becomes a current product service.
