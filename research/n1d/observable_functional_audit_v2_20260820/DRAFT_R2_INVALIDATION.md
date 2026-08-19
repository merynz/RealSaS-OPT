# Observable Functional Audit V2 — Draft R2 Invalidation

**Status:** `INVALID_PRE_FREEZE_DRAFT__DO_NOT_EXECUTE`

R2 never reached preregistration, never opened evaluation truth, and never produced a scientific result.

It is invalidated because the attempted GitHub persistence method encoded the ZIP as manually transported base64 text parts. Blob-level verification showed that at least `source_bundle_r2_b64/part00` did not match the intended local bytes. The transport method is therefore rejected as an authority mechanism.

The replacement R4 authority uses ordinary GitHub source files with blob verification. Immutable legacy source dependencies are treated as separately hashed input artifacts, and the first-party experiment source is mirrored as a deterministic ZIP in Drive and Library.
