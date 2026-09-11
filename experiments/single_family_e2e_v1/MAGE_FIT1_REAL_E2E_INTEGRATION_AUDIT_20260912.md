# Mage FIT1 real E2E integration audit — 2026-09-12

**Status:** `IN_PROGRESS__RIGGING_CORE_CLOSED__DOWNSTREAM_SEAMS_PRESENT__EXACT_V5_QUALIFIED_SKIN_SERIALIZATION_GAP`

This is an integration audit, not a new scientific closure and not a `PRODUCT_PASS` claim.

## Goal

Run the already-promoted Mage FIT1 mechanical result through the current product stack without replacing, approximating or visually repairing any scientific artifact:

`real 8 images/cameras -> promoted S/G/W -> exact 8 directional M/B -> appearance -> directional joint binding -> motion/proof -> Living Compile -> native runtime`

If an input or qualification-owned artifact is absent, the real E2E path must stop or expose an explicit `ABSTAIN/BLOCK`. Preview-only reconstruction may not be relabeled as canonical product evidence.

## Audited current state

### Promoted rigging core

The canonical Mage FIT1 closure already records:

- `RiggingSurfaceIR`: 950 nodes / 2813 topology edges;
- `QualifiedSkeletonIR`: 22 Compiler-qualified controls;
- Arachne V5 -> Compiler-qualified skin: 950/950 rows;
- V5 GSA row-L1 p95 `0.04237784981177733`;
- V5 deformation ratio `0.019856400787830353`;
- V5 articulated deformation ratio `0.002780771814286709`;
- Compiler total skin correction L1 `2.9468642839168442e-05`.

No unseen-family or product-pass claim follows from those FIT1 numbers.

### Downstream source seams already present

The repository already contains qualified/closed source mechanisms for:

1. exact-eight directional MWB2 mesh/mesh-skin construction from observed local surface relations;
2. observation-derived per-corner appearance under native `PIXEL_CENTER_XY` semantics;
3. Compiler-owned `DirectionalJointViewBindingSetIR` and per-view joint pivots;
4. qualification-owned directional motion bake with current rotation-only evaluator scope;
5. proof-owned bake -> native-v2 projection -> sealed C++ runtime interlock;
6. Living Compile V4 product/editor shell.

These mechanisms are not evidence that the promoted Mage witness has already traversed the complete product path.

## Concrete integration defects/gaps found

### LC-01 — Living Compile rig overlay projection

**Observed defect:** Living Compile derived control anchors from joint support-surface raster centroids / nearest surface nodes instead of consuming the Compiler-qualified directional joint pivots used by the motion evaluator.

**Disposition on integration branch:** fixed. `DirectionalJointViewBindingSetIR` is now exported as a typed bundle artifact; Living Compile consumes its exact `raster_xy` pivots. If the binding is absent, the rig overlay is withheld instead of guessed.

### LC-02 — proof-blocked bundles could not be inspected

**Observed defect:** the UI requested runtime clips during bundle open; `/api/runtime/clips` rejected every non-PASS proof, preventing even static inspection of a truthful unqualified product state.

**Disposition on integration branch:** fixed. Static scene/mesh/rig/weight inspection remains available under `PROOF BLOCK`; runtime clips remain empty and `runtime_frame` remains strictly PASS-gated.

### ART-01 — exact V5 QualifiedSkinIR bytes were not persisted in the light closure evidence folder

The promoted V5 evidence folder was enumerated during this audit. It contains the V5 decoder delta, source, preregistration, composite manifest, closure report/seal, evidence-light archive and visual plates. It does **not** contain a serialized `SkinProposalIR` or `QualifiedSkinIR` for the final V5 Mage closure.

The closure metrics prove that Compiler qualification occurred, but those metrics are not a substitute for the exact 950-row product artifact.

A previously reconstructed V5 weight matrix from the best10752 field-token snapshot is **not authorized** as the exact promoted skin: it reproduces the closure closely but not byte/metric exactly because the full promoted V4 backbone execution is not represented by that snapshot alone.

**Required repair:** execute the promoted composite checkpoint (`V4 backbone SHA-256 95c441...` + `V5 decoder delta SHA-256 133441...`) on the sealed Mage conditioning input, emit `SkinProposalIR`, run the current Compiler `qualify_skin`, and persist the exact `QualifiedSkinIR` with content hash and source checkpoint identities. No optimizer/backward/update is required.

## Real E2E trigger contract

The eventual real trigger must fail closed unless all of the following hold:

- exactly 8 admitted 1024 RGBA observations and exactly 8 bound cameras;
- promoted/expected S, G and W lineages/hashes match the selected witness;
- no teacher mesh, teacher skeleton or teacher skin is consumed at inference/product assembly;
- no `mock`, `synthetic`, demo-tessellation or reconstructed-weight artifact can satisfy a real-artifact requirement;
- MWB2 never bridges typed UNKNOWN/UNOBSERVED geometry;
- visual completion, if ever used, is separately typed and qualified and never mutates S/G/W truth;
- per-view rig overlay uses the same qualified directional pivots as motion evaluation;
- missing deformation or motion frame evidence remains `ABSTAIN`, never synthetic PASS;
- native export remains forbidden until the exact current product has fresh PASS proof;
- every emitted artifact records source lineage/content hashes in one reproducible bundle manifest.

## Next execution order

1. Persist/re-emit exact promoted V5 `SkinProposalIR` + `QualifiedSkinIR` for Mage.
2. Feed exact S/G/W through current MWB2 for all eight directions.
3. Bind exact observation-derived appearance; leave unsupported regions visibly absent.
4. Qualify directional joint/view binding and verify S/E/N/W rig overlays against the real raster silhouette.
5. Materialize a proof-blocked inspection bundle first if required; inspect IMAGE / MESH / RIG / WEIGHTS separately in Living Compile.
6. Add only qualification-owned motion evidence; then run proof.
7. Export native runtime only after actual PASS.
8. Wrap the full path in one fresh-process real E2E trigger with the no-fabrication checks above.

## Claim boundary

Until ART-01 is repaired and the real witness actually traverses the downstream path:

`MAGE_FIT1_RIGGING_CORE = CLOSED`

`MAGE_REAL_PRODUCT_E2E = OPEN`

`PRODUCT_PASS = NOT CLAIMED`
