# RSS.vNext — explicit editor approval of learned OPEN pin proposals

**Scope:** one indexed TESSA M*, AXIS/MIRA canonical Rig/Skin, explicit
user-authored pin selection, existing XPBD surface solver. This is an
engineering executable/editor boundary; it is NOT a new cloth network,
not a source-native pin witness, and not neural/release quality proof.

## Authority flow

1. **Neural review only**:
   `propose_open_pin_canonical_editor_review_v1` produces a canonical
   candidate with the full M* topology, oriented face, vertex, Rig/Skin
   lineage and host-chain hypothesis. It can also report
   `REVIEW_NEEDS_EXPLICIT_HOST_PIN_BINDING_REPAIR`.
2. **Immutable compiler DTO**:
   `lower_open_pin_neural_review_to_editor_candidate_v1` creates an
   `EditorOpenPinNeuralReviewV1` with a deterministic review hash. It
   never calls the XPBD solver. The untrained model head stays off
   in ordinary inference; the diagnostic caller must explicitly opt in.
3. **Editor authoring action**:
   the editor/controller must capture the actual user's approval as a
   distinct `EditorOpenPinAuthoringCommandV1`, binding exact review
   hash, Mesh/Rig/Skin lineage, revision identifier, action identifier,
   approved hem vertices and chosen host chain. Its hash protects
   internal command integrity **but does not authenticate a human**.
   This Python module does not itself implement a browser button,
   session permission, electronic signature or source-fidelity gate.
4. **Compiler structural court**:
   `prepare_editor_approved_open_pin_domain_v1` validates the two
   independently addressed face components, complete connected OPEN
   manifold sheet, boundary-only pins, bounded pinned fraction,
   actual host chain and positive pinned-vertex MIRA skin support.
   A cloth pin can be attached to the host without binding the rest
   of the garment to that chain; however the current XPBD v1 execution
   requires a real kinematic host influence at each pin.
   Missing host support is a guided repair, not an automatic LBS
   rewrite. A Skin edit changes the lineage and requires a fresh
   review + approval.
5. **Declared but disabled physics**:
   the returned `XPBDSurfaceDomainV1` comes from the existing
   `prepare_declared_xpbd_surface_domain_v1`; it does not run physics,
   assert material coefficients or call `enable_declared_xpbd_surface`.
   The actual session caller must separately select an explicit
   `XPBDSurfacePolicyV1` and activate the domain. Native Graphics
   post-physics validation remains an independent ProductProof gate.

## Strict separation from native source admission

This editor-authored path **never invokes**
`prepare_source_qualified_open_surface_xpbd_v1` or uses that function's
trusted source/transport receipts. A user-approved edit can be useful and
executable without becoming source-native training truth.

Both the model review and the editor command have:
`source_pin_qualified=False`, `automatic_xpbd_activation=False`,
and `source_material_truth_claimed=False`. The resulting domain
retains `product_release_qualified=False`.

No dataset values, license claims, host-only interventions, generated
missing-view pixels or physics material constants are fabricated.

## Acceptance and pending work

Synthetic Compiler/runtime tests cover exact Mesh/Rig/Skin identity,
tampering, manual pin overrides, cross-component rejection, source-truth
separation and actual deterministic XPBD session stepping. This proves
the adapter is executable, NOT that a model learned where to attach a
cape or that users can already approve it from the web editor.

Remaining: wire a real Studio interaction and persistent user revision
with provenance/permissions, dedicated host-pin binding repair UX,
real licensed cape pin supervision, held-out anatomy tests, XPBD body
contact and post-physics native RGB/alpha parity, and final ProductProof.

**Core files:**
`compiler/realsas_compiler_core/mechanics_editor_open_pin_approval_v1.py`
`models/rss_vnext/product_neural_open_pin_editor_review_v1.py`
`tests/compiler/test_mechanics_editor_open_pin_approval_v1.py`
