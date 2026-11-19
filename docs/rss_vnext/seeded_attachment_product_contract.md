# RSS.vNext — production seeded attachment face teacher contract

Scope: neural source-native **HELD → HOLDER** face correspondence on the
**same generated-six-view H(x), same TESSA M*, same AXIS program** as the
seeded product rig/skin trainer. This contract does not create dataset truth,
physics policies, source-authored sockets or verified runtime contact.

## Activation

Trainer profile `PRODUCT_SEEDED_IRIS_TESSA_V1`:
- `product_axis_mira_weights.enabled = True`;
- `axis_training_policy = PRODUCT_MESH_MATCHED_REQUIRED`;
- nonempty `trusted_anchor_face_registry`;
- `product_anchor_face_weight > 0` (finite and nonnegative);
- an enabled, source-qualified AXIS program in the same batch.

When the field is absent on an otherwise qualified asset, leave all three
`source_anchor_face_*` batch fields empty. The geometry/rig loss still runs
without a contact loss. Never label absent or unobserved contact as NO_FACE.

## Qualified batch fields

| Field | Required on a labeled asset | Notes |
| --- | --- | --- |
| `source_anchor_face_targets` | tensor `[A,2]`, int64 | HELD then HOLDER, exact **final M*** faces |
| `source_anchor_face_mask` | tensor `[A,2]`, bool | FALSE is UNKNOWN, not a negative |
| `source_anchor_face_admission_key` | string | Key into trusted source registry |
| `axis_teacher_sequence` / `axis_teacher_admission_key` | AXIS source teacher | Causal event sequence with source socket anchors |

`F` (number of student M* faces) means **verified NO_FACE** on a labeled
endpoint. A mask of FALSE does not permit inference of NO_FACE. Some
equivalent source faces may all be valid attachments; this head currently
requires an exact source-qualified M* reindex, not fabricated nearest-face
labels. Such ambiguous labels must be separately resolved or masked.

## Trusted attachment registry entry

Existing source-qualified face-loss fields are mandatory:
`admission_status=QUALIFIED_SOURCE_NATIVE_ATTACHMENT_FACE_REINDEX`,
`source_semantics=NATIVE_RIG_ATTACHMENT_HELD_HOLDER_FACE_OR_VERIFIED_NO_FACE`,
`source_asset_sha256`, `dataset_manifest_sha256`,
`tessa_mesh_topology_sha256`, `axis_teacher_sequence_sha256`,
`source_face_targets_sha256`, `source_face_mask_sha256`, and
`source_reindex_proven=True`.

Seeded production additionally requires:
- `source_qualification_key`;
- `observed_rgba_sha256`, `observed_camera_sha256`;
- `source_binding_sha256`, `combined_generation_checkpoint_sha256`;
- `candidate_mstar_topology_address_sha256` equal to the ACTIVE final
  `seeded_mstar_teacher_address_sha256_v1(mesh.seeded_surface)`.

This last identity includes original faces and TESSA refinement parents,
final face indices, components and layers: neither a new M* face count nor
a source-space nearest-vertex match can silently recycle old labels.

## Architecture boundaries

- Output is unqualified neural evidence until Compiler validates attachment
  direction, joint/socket frames, live relative motion, and source witness.
- The loss touches source labels only at the loss boundary; model input
  remains exactly two actual images + six generated conditioning views.
- The attachment loss is reported in `anchor_face_correspondence_loss` and
  the seeded gradient court as `product_anchor_face`.
- The weight is saved as a distinct checkpoint objective policy. A change
  requires a new training phase rather than an unmarked resume.
- This closes only the **static directed face correspondence training
  connection**. Temporal attachment/pin intervals, multi-pin surface
  domains, solver selection, and XPBD final frame → Graphics still require
  separate runtime/source-qualified contracts.
