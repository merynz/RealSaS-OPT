# Canonical user edit -> actual native C++ execution consequence court

This court completes a missing *execution* interlock after #624's canonical
Mesh/Rig/Skin `.rsedit` roundtrip and #626's face-preserving, source-unqualified
Graphics/UV transport. #627 independently inventories source-registered
known-pixel frame samples, but this new court does not create those samples.

## Inputs and parent identity

The host supplies immutable original `.rss`, authored `.rsedit`, derived
user-edited Graphics preview `.rss` and one pinned C++ native renderer binary.
Each file is hashed under explicit byte limits. The `.rsedit` is independently
replayed against its original archive. The edited preview is fully regenerated
and byte-compared against the original + edit. The actual edited canonical
Mesh/Rig/Skin IDs and face topology must match; the regenerated native pose
must have the expected selected virtual camera resolution. Eight-way
virtual-camera edits are allowed to reproject after a geometry edit.

The same SHA-pinned C++ executable is run twice with `source_view=-1`:
once on the original, then on the revised `.rss`. No source RGB or original
camera can be claimed by this path. The returned RGBA, front face-owner,
color-authority and unknown-occlusion sidecars are exact typed bytes. The
raster archive has a deterministic `RSEDPX01` ABI, an exact eight-array
ordering and bounded image sizes, and its SHA is tied to the immutable input
parents and renderer SHA.

## Independent replay

`independently_replay_canonical_user_edit_native_cpp_v1` does not accept a
merely rehashed raster receipt. It validates field-by-field authority flags,
sidecar types, face IDs and color authority, then reopens the source and edit
and invokes the **same pinned native executable again**. Exact binary raster
bytes and the full typed court must match. A modified pixel, frame, renderer,
archive or forged ProductPASS fails.

The court distinguishes two things: (a) a user edit is persisted and actually
executed by the real native renderer, and (b) the resulting image differs
from the original at one or more sampled pixels. (b) is reported as a count,
not required to be positive because some valid edits may lie outside the
selected view.

## Authority and remaining gates

This is an **execution court only**. `edited_source_rgb_qualified=False`,
`edited_source_camera_observed=False`, `continuous_animation_proven=False`,
and `product_release_admission=False` are mandatory, even on successful
replay. Observed original-source pixels and real artist animation remain
outside this output. The initial corpus, teachers and source rights are
untouched. A production ProductProof consumer must separately bind this
court to independently admitted source, native-execution and release witnesses
before authorizing a ProductRevision.
