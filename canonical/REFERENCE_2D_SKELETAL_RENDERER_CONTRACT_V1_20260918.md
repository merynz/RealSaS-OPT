# RealSaS — Reference 2D Skeletal Renderer Contract V1 — 2026-09-18

**Status:** `ACTIVE_REFERENCE_BASELINE__NOT_PRODUCT_PASS`  
**Execution authority:** self-hosted `realsas` runner only for playback/render diagnosis  
**Purpose:** define the non-negotiable graphics-engineering baseline before RealSaS-specific compiler/runtime extensions are judged.

## 1. Separation of concerns

This contract separates three questions that must not be collapsed into one gate:

1. **Renderer correctness** — given an exact posed triangle set, UV/source binding, composition and texture, does the runtime rasterize it correctly?
2. **Dynamic geometry quality** — did the Compiler/model pipeline produce posed geometry that remains useful under motion?
3. **Product visual quality** — does the complete assembled puppet look correct to the founder/artist?

A renderer is not allowed to reject otherwise valid triangle input merely because the drawable domain is not a single globally simple manifold boundary. Conversely, a correct renderer does not certify that a dynamically deformed mesh is aesthetically or mechanically good.

## 2. Reference-correct raster/deformation baseline

| Invariant | Reference requirement |
|---|---|
| Topology during playback | Face/index topology is fixed for an attachment during a clip unless an explicit attachment switch occurs. |
| Shared vertex geometry | Every triangle referencing the same geometric vertex consumes the same posed coordinate. |
| UV seams | Render-only UV vertex duplication is allowed; geometric duplicates must replay the exact same posed position. |
| Vertex deformation | One authoritative posed-vertex result enters the renderer. The renderer does not re-solve skinning. |
| Primitive | Triangles. Triangle soup and multiple components are legal. |
| Global manifold boundary | **Not required for raster correctness.** |
| Triangle winding | Both windings rasterize unless culling is explicitly requested by the authored/runtime contract. |
| Pixel sample | Pixel-center convention is explicit and stable. |
| Shared-edge ownership | Half-open/top-left edge ownership or an equivalent crack-free convention. |
| Barycentrics | Computed from the same edge functions used for coverage. |
| UV interpolation | Barycentric interpolation from admitted per-vertex/per-corner UVs. |
| Texture sampling | Explicit filter and address mode; current baseline is clamped bilinear. |
| Alpha | Source alpha is sampled; alpha==0 may discard a fragment. Epistemic confidence may not silently erase a covered fragment. |
| Draw order | Explicit slot/layer order; not inferred from triangle index or hash order. |
| Depth | If used, semantics are explicit and deterministic. 2D semantic order remains an explicit tie-break where intended. |
| Backface culling | Off by default for deformable 2D art unless an authored policy explicitly enables it. |
| Clipping/masking | Explicit authored/runtime feature, never implicit uncertainty handling. |
| Completion | Forbidden unless explicitly enabled by a separately qualified product policy. |
| Dynamic folds/stretch | Product/deformation-quality concern; diagnostic or qualification layer, **not** a prerequisite for the triangle rasterizer to execute. |

## 3. Required conformance fixtures

Before a Mage render is used diagnostically, the native reference raster should have deterministic fixtures for:

- two-triangle quad with shared diagonal;
- both triangle windings;
- UV-seam duplication with identical geometry;
- triangle soup with disconnected components;
- canonical pinch vertex (multiple fans sharing one vertex);
- skinny but nondegenerate triangle;
- triangle inversion with culling disabled;
- subpixel translation across pixel centers;
- alpha edge and fully transparent texel;
- equal-depth semantic draw-order tie;
- rigid-over-body composition;
- interpolation between stored posed frames.

These are deterministic conformance tests, not empirical architecture experiments.

## 4. RealSaS-specific work starts after this baseline

RealSaS may add:

- multi-view source ownership;
- Compiler-qualified evidence and lineage;
- source-only continuity underlay;
- view-local attachment activation;
- fail-closed completion policy;
- model-evidence -> Compiler-authority promotion;
- runtime package identity and source hashes.

Those extensions must not silently redefine the standard triangle coverage rules above.

## 5. Retired prerequisite

`compiler/realsas_compiler_core/playback_directional_assembly_cert_v1.py` was retired from active Runtime-v4 render admission on 2026-09-18.

Reason: it bundled useful source-authority checks with a stronger global boundary/manifold + continuous-embedding theorem and made that theorem a prerequisite for native rendering. A single-simple-boundary theorem is not a requirement of a correct 2D triangle rasterizer and blocked the real Mage triangle domain before the renderer could be evaluated.

Dynamic geometry diagnostics remain necessary, but they must be owned by Compiler/product-quality proof and evaluated independently of rasterizer executability.
