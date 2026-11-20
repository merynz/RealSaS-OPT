# ProductProof: host-only independent C++ user edit sample

## Product path

ProductProof consumes a **single source-bound product attempt**, not a
standalone editor screenshot. The upstream `product.graphics` has emitted the
original immutable `RealSaS.RSSvNextNativeInteractivePuppetPreview.v1` CAS
object, itself independently recompiled from the real original two-image
source, canonical M*/AXIS/MIRA and candidate Graphics. `product.motion`
supplies the same attempt's canonical motion.

An editor user edit supplies exactly four optional CAS inputs:

| Artifact type | Format | Authority |
|---|---|---|
| `RealSaS.RSSvNextCanonicalUserEditRevision.v1` | immutable `.rsedit` | candidate user change |
| `RealSaS.RSSvNextEditedGraphicsPreview.v1` | resulting `.rss` | candidate no-source preview |
| `RealSaS.RSSvNextCanonicalUserEditNativeRasterBundle.v1` | bounded `RSEDPX01` binary | untrusted native raster |
| `RealSaS.RSSvNextCanonicalUserEditNativeExecutionCourt.v1` | typed JSON | untrusted execution claim |

Partial, duplicate or schema-mismatched sets fail closed. Their raw CAS bytes
are checked for size, SHA-256 and exact source identity. The original
`.rss` must match **both** the active canonical Mesh/Rig/Skin and the
existing ProductProof original source-native preview CAS SHA (which was
independently regenerated from the original upload).

ProductProof reconstructs `.rsedit` canonical Mesh/Rig/Skin and byte-for-byte
rebuilds the full edited Graphics/UV archive from the original source preview.
It reopens the typed raster archive, verifies all sidecars and actual changed
pixel counts, and rejects falsely claimed source RGB, source camera, complete
animation and ProductRevision. Merely rehashing the court or substituting
source clips cannot pass the lineage court.

## Host-only independent execution

The worker optionally receives
`trusted_native_edit_replayer=make_trusted_native_edit_replayer_v1(...)`,
configured **outside** the user request by the execution host with a real
SHA-256-pinned native C++ executable. No executable path, runnable command,
host callback or source testimony may be supplied via `capability_metadata`
or the request body. The trusted callback receives only independently reopened
CAS bytes and a typed court; it rematerializes and reopens the same parents,
runs the native executable **again**, and compares complete raster bytes and
the typed court. Host failure propagates as proof failure, not ABSTAIN-as-PASS.

If the host has not configured this callback, the CAS court remains a
reopened **candidate** and
`independently_native_replayed_edit_frame_count=0`. Once independently
replayed, one bounded edit frame may be reported, but not continuous motion.

The actual platform worker now provisions the optional callback only from
**operator-controlled environment**:

- `REALSAS_PRODUCT_PROOF_NATIVE_RENDERER_PATH` — local installed C++ binary
- `REALSAS_PRODUCT_PROOF_NATIVE_RENDERER_SHA256` — exact trusted binary digest

Both must be supplied together or the worker refuses the incomplete pin
configuration. Neither setting can be supplied by user request or CAS
metadata. The worker verifies the executable SHA before use; each replayer
execution checks it again and runs isolated SHA-verified file snapshots.
With neither variable configured, deployed proof remains untrusted-candidate
only and fail-closed. Provisioning a binary does not turn on ProductPASS by
itself.

## Deliberate fail-closed boundary

One such successful frame proves original-versus-edited *native execution*
at one explicit **virtual** view/time/frame on the same canonical lineage. It
does **not** prove the original source RGB/camera, hidden appearance, artist
clip motion, contact/attachment truth, or the full release edit consequence.
The `EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN` gap intentionally remains in
`missing_independent_witnesses`. ProductProof returns `FAIL` with no
product outputs until independent source graphics, physical behavior and
release admission also close.

No model, teacher, dataset, source truth, asset right, or physics numerical
policy is changed in this patch.
