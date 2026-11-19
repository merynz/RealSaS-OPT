# RSS.vNext editable canonical authoring revision: Mesh / Rig / Skin

BUSTED's puppet is the product; source image → model → final canonical
Mesh/Rig/Skin cannot be "editable" merely because JSON exists. The actual
authoring revision must persist, reopen, retain stable identity, recompute
the real Mechanics output, and not counterfeit its original source teacher.

## Source vs user-authored authority

The admitted base `.rss` remains **immutable**. The new typed `.rsedit`
revision references its exact archive SHA-256 and preserves all canonical
vertex IDs, face topology, rig joint/control IDs, and skin row addresses.
A revision may edit three owner fields:

1. **Mesh vertex coordinates** (same topology and canonical vertex IDs).
   Requalification blocks nonfinite coordinates, collapsed triangles and
   invalid vertex IDs. The mesh lineage hash changes; a dependent Rig and
   Skin receive that new parent identity.
2. **AXIS joint rest positions** (same parent graph and joint IDs). The
   deform-structure, control-program and rig lineage hashes are recomputed.
   MIRA Skin is invalidated/rebound to the new deform-structure identity.
   No silent reuse of original model's rig-source truth is allowed.
3. **MIRA per-canonical-vertex skin weights** (same vertex and joint
   identity, nonnegative weighted simplex). Weight-only edits do not
   rename or invalidate the unchanged Mesh or Rig; the Skin lineage
   changes. Missing/invalid influences, unknown joints and no-op edits fail.

`compile_rss_vnext_canonical_user_edit_revision_v1` reopens the base
`.rss` first, applies *user-authored* changes, builds derived canonical
Mesh/Rig/Skin IR values and **re-runs the real LBS solver** for every
preexisting animator control frame. It seals the derived typed JSON, pose
bytes, patch scope, new and original canonical lineage IDs and source
archive SHA in a deterministic authoring archive.

`reopen_rss_vnext_canonical_user_edit_revision_v1` independently reopens
the original archive, verifies exact parent bytes and all typed derived
IR/patch scope/lineages, and **recomputes Mechanics projected pose bytes**.
Manipulating pose.bin or rehashing a forged ProductPASS cannot bypass the
independent replay gate. These are testable edits, not a second puppet
mesh or a video/spritesheet.

## Critical boundary

The original Graphics/UV archive is tied to the ORIGINAL canonical mesh
and original inferred source appearance. A mesh-rest-pose change **does
not magically source-qualify or rebind** its per-face Graphics/UV projection.
Therefore this `.rsedit` is intentionally a standalone typed authoring
revision, not an overwritten source `.rss` or a product-released native
archive. Its manifest explicitly states `graphics_uv_binding_qualified=0`
and all original source geometry/rig/skin qualifications cannot be reused
as approval of the edited result. A subsequent Graphics transport step
must rebind the derived Mesh while retaining texture candidate provenance
and independently test the new rendered pixels before ProductPASS.

**Scope:** vertex positions but not face connectivity surgery, joint
rest positions but not source-authored causal chain rewrites, and weight
painting but not joint creation/deletion. All three edit **fields** are
saved and independently reopened; multi-revision edit stacks and native
Graphics re-render after geometric changes are separate remaining work.

## Dataset boundary

No native teacher, image, rig library or dataset manifest is read/written
by this operation. All regression assets are synthetic fixture inputs
through the existing two-image Compiler preview, not new training data.
