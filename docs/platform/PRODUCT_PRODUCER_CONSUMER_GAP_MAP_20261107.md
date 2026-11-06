# BUSTED — source producer / consumer / runtime closure audit

**Audited base:** main `7efb85e73413c05bdb5c42eff7f6fe817f7ebb38` (Motion V2 landed) on 2026-11-07.  
**Scope:** read-only source and registered-contract audit, no Platform release, Attempt, ProductRevision, model inference, source qualification, deployment or Graphics research changes.  
**Reproduce:** `python -m tools.product_producer_consumer_audit_v1 --strict-registered`.

## 1. What actually produces and consumes today

| Producer | Output | Next current registered consumer | Actual state |
| --- | --- | --- | --- |
| `product.input` | `ShapeInput/v1` | `product.shape` | Registered, typed edge matches |
| `product.shape` (IRIS/TESSA) | `ShapeProposal/v2` | `product.mesh` | Registered, typed edge matches; no newly qualified production ModelRelease implied |
| `product.mesh` | `CanonicalMeshIR/v1`, `RiggingSurfaceIR/v1`, `TESSAMaterialSupportFieldIR/v1` (and other qualifiers) | `product.rig` + `product.skin` | Actual CAS `_seal` emissions verified by source AST |
| `product.rig` (AXIS V5.4.1) | `CanonicalRigIR/v1` | `product.skin` | Registered producer/consumer version matches |
| `product.skin` (MIRA V5.5) | `CanonicalSkinIR/v1` | No current registered `product.mechanics` | Registered v1 prefix; product continuation stops here |

**Not equivalent to generic product completion:** the registered prefix is developer-only, `product_pass_authority=false`. The current compiler contains `CanonicalRigIR.v2`, `CanonicalSkinIR.v2`, `CanonicalMechanicsIR.v1`, `CanonicalMotionIR.v2` and baseline solvers. Presence of modules on GitHub does **not** make the current v1 Rig/Skin producer outputs consumable by Mechanics v2.

## 2. Candidate v2 composition requires real producers

**Target wire:** `Mesh.v1 → Rig.v2 → Skin.v2 → Mechanics.v1 → {Graphics.v1, Motion.v2} → Proof → ProductRevision`. Graphics and Motion are siblings depending on Mechanics; no accidental Graphics→Motion or Motion→Graphics authority shortcut. CAA stays under Graphics.

The original V6 producer worker files reference AXIS and MIRA runtime/model modules absent from this audited main. They must be imported with their exact checkpoint/model closure and guarded deserialization (original V6 runtime checkpoint loaders call `torch.load(..., weights_only=False)`). There is no permission to mint trained checkpoints, infer unobserved controls, convert v1 artifacts to v2 by relabeling, or promote research models to production.

The Mechanics CAS worker [#368](https://github.com/merynz/RealSaS-OPT/pull/368) is now merged to main at `173664070e4e2b81d4aa0e79d8c8d4248dbbfe9e` but remains **rest-qualification only**. Go/Platform cannot register it behind the currently producing v1 Rig/Skin artifacts. Interactive drag, constraints, deform, 6D grip and actual source behavior need distinct runtime and independent proof.

## 3. Runtime and product revision are still a legacy consumer path

Current Go `FinalizeCompile` explicitly promotes:
- Stage `40_MOTION_COMPILE_RUN`: `RealSaS.QualifiedMotionArtifact/v1` as `motion:qualified`
- Stage `43_RSS_MATERIALIZE_COMPACT`: `RealSaS.RuntimePackageArtifact/v1` as `runtime:package`
- Stage `46_PRODUCT_CLOSURE_SEAL`: product closure qualification

`ResolveRenderRequest` validates the bound revision/motion/runtime identities and sends CAS objects to the Python render tail. That tail extracts `clips[].frames` and invokes the C++ .rss player per frame. It **does not** consume `CanonicalMotionIR.v2.control_tracks`, solve live Mechanics from it, or reproject newly edited geometry/Graphics into .rss. This is an existing, valuable legacy render route, not the new interactive puppet route.

Native `.rss` parser/writer and C++ code exist, but a new **artifact-bound runtime ABI/admission test** must exercise the intended v2 combined output and reopen/edit/serialize/reopen invariants. It must verify actual mesh/skin/control/graphics/motion binding hashes, clip interpolation, view visibility/order and attachments without relying on an old baked-run witness.

## 4. Editor / edit consequences are currently separate

`product/living_compile/server.py` explicitly reports `compile_available=false` and rejects compile-from-UI routes. The editor opens an existing compiled bundle; authoring saves optional user-layer edits with `requires_compiler_requalification=true` and `can_promote_product_directly=false`. This is correct fail-closed behavior, **not** persistent product editing with immediate qualified playback.

The missing pathway is a versioned, immutable edit transaction bound to a source ProductRevision and exact control/mesh/skin/graphics addresses, validated by Compiler/Mechanics/Graphics and promoted as a **new qualified ProductRevision**. Save/load and undo/redo must never overwrite the source revision or bypass Proof. Editing may produce local provisional preview without falsely labeling it product PASS.

## 5. Engineering delivery gates (no invisible green)

1. **Producer gates:** AXIS/MIRA V2 actual checkpoint loader/API import, exact typed worker dispatch and `ModelRelease` match; missing releases fail visibly. Safe `torch.load` semantics and no source-model weight promotion by code merge.
2. **Consumer gates:** register developer-only Go roles only after the exact producer versions match. Mechanics typed execution and separate runtime behavioral capability matrix, including unsupported operator/DOF/attachment abstention.
3. **Graphics + Motion:** real independent source/appearance/occlusion/draw-order/attachment proof; `CanonicalMotion.v2` sampled into Mechanics with real mesh deformation and Graphics under the same ProductRevision identities.
4. **Runtime ABI:** output one sealed, portable `.rss` preserving identity, configurable controls and clip tracks; native reopen, edit, deform, compositing, deterministic re-open; prove no legacy Stage40 substitution.
5. **Product UX:** one-image frontend when authorized/ready, upload/start/status, durable job progress, edit transactions, explicit failure handling, persistence, exports and repeatable identity.
6. **Qualification:** cold new-subject compile, warm revision render without re-inference, timings/p95/resource budget, real morphologies, source-backed interventions, independently held-out unseen, and exact ProductRevision qualification. No launch claim from current synthetic CI.

The accompanying audit tool protects **registered** Python/Go descriptors, worker route existence, primary producer→consumer type+schema edges, and Mesh's secondary artifact emissions. It intentionally reports missing v2/product interfaces as BLOCKED without turning them into an unrelated CI failure. Its own `registered_contract_status=PASS` is a narrow source-wiring result, never product PASS.

**Safety:** This PR changes only audit code/tests/docs/workflow. It does not mutate Corpus370, Graphics scientific results, model weights, Platform state, release policy, native runtime, Go controller or website.
