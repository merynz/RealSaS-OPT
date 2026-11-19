# RSS.vNext: Canonical User Edit -> Native Graphics/UV candidate preview

The base `.rss` and its Graphics source/inference evidence are immutable.
PR #624 creates a `.rsedit` authoring revision for Mesh vertex positions,
AXIS joint rest positions and MIRA skin weights. This step renders that exact
edited canonical triplet with the native C++ 2D Graphics runtime, without
promoting the edited appearance to original source truth.

## Identity and transport

`compile_rss_vnext_edited_graphics_preview_v1` independently reopens both
the original `.rss` and the authored `.rsedit`. The base .rss SHA-256 must
match the edit receipt. Face topology, canonical vertex, joint, skin and
control addresses are preserved by the edited revision court. This permits
face-local UV barycentric addresses and Graphics fragment ownership to be
*transported as a visual candidate*, not reused as source-certified color.

The original `RSGFX001` binary is boundedly decoded and its exact original
mesh and runtime hash binding verified. Original registered view selections
are discarded. A real, rehashed `RuntimeGraphicsV1` is emitted with the
**edited** mesh lineage and all eight views widened to canonical fragments
with only posed depth ordering. Source camera registration remains false.

The original `RSUVT001/RSUVT002` texture is strictly reparented to the
edited canonical mesh and preserves the face-local atlas UV coordinates and
inferred H(x) color candidate. Both original source RGB layers, their
source masks and their conflict mask are ERASED from the edited preview.
Consequently even requesting a source-view hint from the native renderer
cannot grant authority=1 on the moved surface. Unknown pixels remain
unknown rather than becoming transparent source background.

The canonical user-edited Mesh/Rig/Skin, pose, controls, Graphics, texture
and manifest are reassembled into a normal eight-entry native preview
archive (1- or 8-camera profile). The real LBS poses are recomputed,
including reselecting all eight **virtual** cameras when necessary. The
new content hashes are signed by the new authoring provenance chain, and
the exact output can be regenerated from the two immutable parents.

## Authority

The output is a `DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION`, not a new source
image or graphics teacher. The manifest and typed receipt explicitly hold:

- `graphics_uv_binding_qualified=0`
- `source_uv_requalification_pass=0`
- `source_rgb_surface_ownership_qualified=0`
- `original_source_camera_observed=0`
- `user_edited_source_color_layers_erased=1`
- `product_release_admission=0`

A successful C++ render proves that edited canonical mechanics reaches the
native 2D pixels. It does not prove artistic appearance accuracy, original
camera alignment, color completion quality, hidden-surface coverage,
motion quality or release suitability.

## Negative courts and limitations

Tests compile an actual two-source canonical preview and multi-joint edited
revision, then invoke the real C++ player on the recompiled preview. They
measure both canonical ownership and color authority sidecars. Rehashed
raster/manifest corruption, stale base source, fake ProductPASS and
source-color authority laundering are rejected.

This pass supports vertex-position editing but not face connectivity
surgery, cross-topology UV remeshing, new joint creation or manual UV
painting. Original source appearance cannot be declared valid again
without independent camera/source/temporal Graphics courts. No dataset or
teacher files are read or modified.
