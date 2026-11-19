# ProductProof: independent temporal source Graphics sample audit

The source/native Graphics courts introduced in #622 and independently
raw-replayed by #623 can establish one known-pixel frame under an independently
registered source camera. Neither #625 native C++ candidate capture nor
a set of matching frames establishes time-continuous 2D appearance.

The new temporal auditor consumes **only positive, trusted-host raw replay**
objects from ProductProof. It is never given original unverified CAS-only
claims. Each court is tied to the active canonical Mesh/Rig/Skin/Graphics/Motion
parents by the existing ProductProof gate. The audit checks:

- Single source-asset identity, stable native camera identity per view,
  stable native renderer binary identity per view.
- Unique source view / frame index / source target-kind addresses.
- Artist-approved motion times within the canonical clip duration and
  strictly increasing time **and** frame index.
- Presence of original rest frames by source view, observed motion timestamps,
  known-pixel counts, and largest intervals between sampled frames.
- A positive court must keep its negative ProductRevision, hidden-surface and
  unobserved-motion authority flags even if the court JSON was rehashed.

The diagnostic includes actual per-view sampling spans and unknown-occlusion
measurements, with `complete_temporal_coverage_proven=False` and
`product_release_admission=False` in every case.

This is an inspection/consistency gate, not an authorization to use model
predictions as artist source observations. Rehashed receipts are not standalone
trust anchors. Actual raw image/renderer witness registries stay external to
MODULE metadata. No dataset, pretrained weights or source teacher records
are changed.

## Remaining independent acceptance gaps

1. Original source camera calibration and registered native projection
   with a real source raster on each claimed view.
2. Explicit temporal sampling/coverage policy and qualified animated targets
   for dynamic pose, silhouette, visibility, occlusion and revealed surface.
3. Independently replayed contact, attachment, penetration, time-step and
   XPBD/rig deformation consequences.
4. Product-level edited Mesh/Rig/Skin -> native Graphics -> saved executable
   puppet result under independent source/image and runtime admission.
5. GPU/large-mesh performance, checkpoint/model rights and release controls.

The existing ProductProof result remains FAIL until all independent release
witnesses exist. The reviewer must not convert positive sample counts or
coverage diagnostics into a ProductPASS boolean.
