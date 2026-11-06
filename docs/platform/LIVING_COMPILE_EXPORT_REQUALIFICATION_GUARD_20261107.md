# Living Compile export: prevent stale edited-puppet authority

**Source audit:** main at `cd2b9afd1df3ad9fc46a06b07002466a0c27f2d5`.  
**Scope:** product/living_compile API export guard, *not* Compiler qualification or runtime native package admission.

## Confirmed product mismatch

`save_user_layer` writes each authoring transaction under `<bundle>.user/layers/view_XX.json`, setting `can_promote_product_directly=false`, `requires_compiler_requalification=true` and `requires_dynamic_revalidation=true`, while canonical product and native `.rss` remain unchanged.

Before this repair, `POST /api/export` returned `<bundle>/native/realsas_runtime.rss` whenever the file existed, even when the user had edited rig, mesh, weights, bindings, topology, IK or animation. That makes the old, unedited .rss appear to be the edited puppet.

The endpoint also did not itself load the ProductProof to check source product hash and `overall_status=PASS`. A stale, missing or failing proof could therefore leave an unjustified export reference.

## Bounded safety behavior

- If **any** pending local authoring-layer JSON is present for the selected bundle, export fails with `EXPORT_USER_EDITS_REQUIRE_COMPILER_REQUALIFICATION`. A malformed JSON layer cannot override authority by lying about its flags.
- If no pending edit, the server loads the canonical product and its exact source-state-bound proof; `overall_status` must be `PASS` or export fails with `EXPORT_PRODUCT_PROOF_PASS_REQUIRED`.
- Existing native `.rss` presence requirement remains. No file is mutated or deleted; user work is preserved.
- The general `GET /api/blob` route now applies the **same** gate to `.rss` payloads; symlink/relative path resolution cannot trivially bypass it, and undeclared `.rss` blobs are rejected. Non-RSS source artwork remains readable for editing.
- Dedicated regression tests exercise the public dispatch API using actual stored layer operations, including edits on nondefault views, weight edits, wrong-source proof, FAIL/UNKNOWN proof, fake layer flags and missing package.

**Important remaining limitation:** this API returns an existing package *path*, not a byte-level native-product hash/ABI proof or a new qualified edit revision. In particular, this change alone does not establish that an arbitrary .rss file in the bundle is valid or corresponds cryptographically to that proof. That belongs to the future portable runtime/export authority court. This PR must not be described as completed .rss qualification.

## Product next step

A proper user editing workflow must bind `source ProductRevision + control/mesh/skin/graphics addresses + edit transaction`, produce a new immutable candidate revision, rerun affected Mechanics/Graphics/Proof, and only then expose a new `.rss` export. A local provisional preview is allowed but must remain clearly unqualified.

This fix does **not** change trained models, research Graphics truth, Corpus370, or registered Go product capability contracts.
