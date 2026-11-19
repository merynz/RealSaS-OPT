# BUSTED native C++ post-Mechanics raster sidecar capture

The existing canonical `.rss` developer preview and live Mechanics
`RSPOSE01/02` stream already render real final vertex positions, including
XPBD. The native C++ player already provides separate first-hit face-owner,
color-authority and unresolved-occlusion outputs. The new
`execute_native_cpp_candidate_raster_capture_v1` connects all four output
buffers to a stable typed binary product **candidate** receipt.

## Input and execution

- Exact immutable `.rss` template from the existing two-source canonical
  Compiler path, independently reopened to check Mesh/Rig/Skin/Graphics/UV.
- Real live Mechanics `RSPOSE01/02` bytes and source-unqualified receipt
  from `encode_rss_vnext_live_mechanics_native_pose_v1`, exactly bound to
  that original template through
  `bind_rss_vnext_live_pose_to_native_preview_v1`.
- A separately pinned **SHA-256 of the actual C++ executable**. The runtime
  never builds or silently downloads a different player.
- Explicit canonical frame and camera view indexes. Native C++ raw
  rendering is invoked with `source_view=-1`: the template has only a
  **virtual, unregistered** camera. It cannot impersonate source view 0/1.
- Fresh private working directory. Exact expected byte lengths, frame/view
  counters, nonfinite and resource bounds, valid face-owner range and typed
  color-authority/UNKNOWN sidecars must all pass. The template is untouched.

The result is a compressed typed NPZ with exactly these keys:

```
native_rgba_u8              uint8[H,W,4]
native_front_face_i32       int32[H,W]
native_color_authority_u8   uint8[H,W] values 0/1/2
native_unknown_occlusion_u8 uint8[H,W] values 0/1
```

All four arrays and the NPZ have digest witnesses; the receipt also binds
the C++ executable, original `.rss`, final pose byte stream, canonical
Mesh/Rig/Skin, Graphics texture and camera. Reopening verifies all typed
sidecars and refusal flags, including
`independent_source_camera_registered=False`,
`source_color_or_visibility_qualified=False`, and
`product_release_admission=False`.

## Authority boundary

This component provides **observed C++ execution output**, not artist
appearance or independently qualified source camera truth. The output is
useful for post-XPBD visual diagnosis, measured native pixels, and as the
renderer half of the eventual source-registered Graphics court.

The trusted raw Graphics replayer (#623) still requires an **out-of-band
independently verified** source witness registry plus a separately approved
native execution registry. The candidate capture does NOT manufacture or
sign those registry rows, cannot clear source registration, cannot infer
hidden surface color and cannot mint ProductRevision.

Existing `product.proof` remains FAIL until independently re-run raw
source/native evidence and full behavioral coverage have been admitted.
Unknown occlusion is not transparent background, and a predicted or sampled
pixel cannot be relabeled source evidence merely because native C++ wrote it.

## Regression scope

Tests invoke the real C++ executable against the real two-source canonical
preview and a true XPBD-deformed final M* frame; each of the four output
sidecars is cross-compared with an independent direct C++ invocation.
Tampered executable/preview SHA, altered live pose, forged source-registration
or ProductPASS and changed binary bundles fail closed.

These are technical regressions. Neither the test inputs nor the candidate
capture are actual licensed source/artist validation assets.
