# Product Shape hybrid signed/open M0 — explicit request contract

The model and Compiler implement source-free signed+OPEN UDF candidate
construction. To make it available to the actual product two-image pipeline,
the Shape request must explicitly opt in. The default remains signed-only.

## Raw product.shape request

The existing `capability_metadata` object may include BOTH fields:

```json
{
  "hybrid_signed_layer": {
    "name": "body",
    "keep_roles": ["primary_form"]
  },
  "hybrid_open_policy": {
    "open_layer_name": "cape",
    "open_presentation_roles": ["outer_covering"],
    "resolution": [24, 24, 24],
    "query_chunk": 4096,
    "min_open_probability": 0.75,
    "min_kind_margin": 0.20,
    "min_slot_margin": 0.10,
    "max_output_vertices": 65536
  }
}
```

These are source-independent **model-layer requests** and geometric
resource/abstention policies. A user can request an OPEN layer, but cannot
supply a source garment component/face index, force the selection of a
student slot, make up source asset truth, or certify ProductRevision. Invalid,
partial or overlapping role requests fail closed. The policy is validated
against the released model's semantic-role schema.

The Shape worker passes the policy to
`run_uncalibrated_two_image_neural_scene_v1` →
`run_two_source_neural_construction_v1` →
`RSSvNextUnifiedCharacterModel.propose_teacher_free`.

Only two original input images enter the model. Six generated views remain
construction hypotheses. The model's own component-UDF kind logits select
an OPEN slot only if probability and margins pass. Otherwise the proposal
ABSTAINS; a missing cloth is not silently invented or turned into a
closed-FlexiCubes surface. ONE signed+open M0 becomes ONE refined M*
for TESSA/AXIS/MIRA and Compiler CAS (where signed/open kind is sealed).

## Other entrypoints

`run_two_view_to_compiler_proposal_v1`,
`run_canonical_two_view_to_compiler_proposal_v1`, and
`run_two_source_neural_construction_v1` accept the same explicit typed
`hybrid_open_policy` and `hybrid_signed_layer` pair. An absent member,
legacy research mode or an unpinned two-stage generator is rejected.

## Non-authority

This capability does not assign physically plausible cloth, open/closed
source-truth, pinned contact, 2D drawable RGB, or source hidden appearance.
Production hybrid *training* is separately source-qualified under #618.
Release ProductProof still needs independent source/pose/graphics courts.
