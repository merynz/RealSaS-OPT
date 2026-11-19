# BUSTED: independent source-pose Graphics evidence, not a second Graphics head

**Purpose:** join the *same canonical* M* → AXIS/MIRA/XPBD frame → native
C++ Graphics RGBA/face-owner/unknown-occlusion outputs to independently
registered source pixels, while preserving the two-image product input
contract and fail-closed release behavior.

## Three authorities (not interchangeable)

1. **Source 2D truth**: two actual user-provided original RGBA drawings,
   observed pixels and artist-verified postmotion/hidden-reveal exceptions.
   Source camera is uncalibrated. A fitted silhouette camera or a generated
   six-view hypothesis is *never* independently observed or source-qualified.
2. **Mechanics execution truth**: the *final* canonical vertex positions of
   the active AXIS/MIRA control and XPBD state, exact rig/skin/mesh lineage,
   timestamps, pose bytes and camera policy SHA. The pose receipt explicitly
   refuses to claim source color or ProductRevision authority.
3. **Graphics pixel authority**: native C++ output RGBA + first-hit face
   owner + per-pixel color authority + unknown-occlusion sidecar under that
   exact active pose, registered camera, runtime Graphics graph and UV atlas.

`evaluate_registered_source_pose_graphics_v1` admits **one** source-known
frame only when independent source and native renderer registries agree on
all content digests, source asset, native camera registration, final pose,
canonical parent hashes, source-witness ownership, renderer executable hash,
known-mask digest and previously frozen threshold policy.

The source registry must be independently authorized (not student-generated).
The native registry must attest executing the exact pose with the real native
renderer, including owner/unknown sidecars. A trusted caller must supply
the two registries; a CAS content hash alone is not a signature or authority.

## Measured invariants

- Source-known RGB MAE on known foreground, known alpha MAE and visible
  alpha silhouette IoU. The known mask includes explicit source-known
  background to avoid a contour-only numerator trick.
- Exact visible *canonical face index* accuracy on source-known foreground,
  using C++ first-hit face IDs rather than semantic/z-order labels.
- Any known-pixel unknown occlusion, predicted/inferred color, wrong camera,
  absent view registration, forged source-owner mapping or source→renderer
  self-teacher is independently rejected or measured as a failure.
- New surface revealed by motion is **UNKNOWN**, not an empty/transparent
  source target. It is only evaluated after an artist-verified new witness.
- Artist motion target may require an approved postmotion RGBA+visibility
  intervention; a model-synthesized six-view is never a qualified target.

`SourcePoseGraphicsCourtV1.source_registered_pixels_passed` is *local*
qualified known-pixel evidence only. It does NOT assert invisible source
textures, physical cloth truth, all motion clips, ProductProof or the exact
source image style when registration is absent. Every receipt has immutable
`product_release_admission=False`,
`hidden_surface_color_proven=False`,
`unobserved_motion_quality_proven=False`.

## Independent raw source/native raster replayer

The additional `independently_replay_registered_source_pose_graphics_v1`
opens TWO bounded binary raster bundles, not a prior JSON verdict:
a source-qualified RGB/alpha/face-owner/known-mask bundle, and an
independently executed native C++ RGBA/front-face/color-authority/
unknown-occlusion bundle. These are typed NPZ payloads with no pickle and
strict resource limits. Both bundles must have independently pinned
SHA-256 hashes and an explicit source extraction or C++ execution witness
in registries supplied **out of band by the trusted host**.

The replayer recomputes the identical local court using the actual pixels,
final pose and canonical hashes; it compares the result to an optional
claimed CAS court and rejects any rehashed drift. Absence of an independent
native renderer execution record, a missing source issuer, a changed camera,
unknown mask, wrong face ID or tampered raster archive is fail-closed.
This is the first raw source-pixel independent replay capability; it
is not a global release proof or a new dataset pipeline.

## ProductProof consumer

`product.proof` now optionally reopens these typed CAS receipts and checks
exact current canonical mesh/rig/skin/Graphics/motion parents, duplicates and
false authority flags. It *cannot independently replay* the two trusted
registries and raw source/native pixel arrays from a JSON receipt alone,
so it reports local claims as diagnostics **without clearing**
`SOURCE_APPEARANCE_AND_POSE_VISIBILITY` or minting a ProductRevision.

The production follow-up is a trusted qualification-owned proof worker that
reopens the source and native RGBA/owner/unknown bytes, reruns this exact
court, verifies source+native issuers and complete temporal/view coverage,
then binds release-specific source visibility and hidden-reveal programs.
Until that exists, ProductProof remains fail-closed.

## Trusted ProductProof invocation

The optional `execute_proof_capability_v1(...,
trusted_source_native_pixel_replayer=...)` parameter is an **operator-injected
callable**, never a value in `capability_metadata`. It receives the reopened,
current-parent-matched typed court, independently reopens source and C++
native raster bundles using the approved SHA registries, reruns the real
source/Graphics measurement, and must return an exactly equal court object.
Missing or forged replay fails closed. Raw raster replay count and positive
locally known-pixel frame count are diagnostic-only: they cannot, alone,
clear `SOURCE_APPEARANCE_AND_POSE_VISIBILITY`, source hidden-reveal,
contact/attachment, edit consequence, or final release admission.

This deliberately prevents a service caller from attaching a self-hashed
`source_registered_pixels_passed=true` JSON and minting a product.
Only trusted infrastructure may provide the two independent registries.
The executable C++ renderer's owner/unknown sidecars are required alongside
RGBA; style and pose constraints do not collapse into alpha alone.

## Data-side handoff (not an operation by this change)

Dataset/source foundry must deliver rights-qualified original source
raster witnesses, source-to-native view registration, first-hit native face
ownership and selected artist-approved postmotion/reveal references with
explicit UNKNOWN masks. Mechanics/Compiler supplies final M* transport,
timestamps and independent rendered sidecars. **No** new per-asset
eight-direction drawings, no art fabricated from the model's own renderer,
no name-based source face ownership, and no backfilling unknown-as-negative.

All unit fixtures are synthetic adversarial regressions, not proof of
learned quality on the 10k+ production corpus.
