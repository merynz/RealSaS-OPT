# BUSTED — Mechanics CAS worker mainline handoff

**Audit:** [Issue #361](https://github.com/merynz/RealSaS-OPT/issues/361), read against main `ffcbaa9ca68a31bda0c07511fcb886556941c1a0` on 2026-11-07.

## Precisely what is integrated

- Copy the pre-existing, unchanged V6 `mechanics_executor_v1.py` from pinned #297 commit `dd3089957e3126d93db4ad2804e30511cfdca348`. The original Git blob is `a1f4a66e2e67fbce150649a4feaae426cfa966d0`.
- In the current Platform worker, handle the *explicit* `RealSaS.ProductMechanicsCompile.v1` MODULE dispatch. Existing module handlers and stage fallback remain intact.
- The executor reads exactly one canonical Mesh v1, Rig v2 and Skin v2 CAS object, verifies their semantic lineage via the canonical Compiler and seals CanonicalMechanics v1.
- Regression courts check actual in-memory CAS exchange, canonical output readback, independent wrong-skin lineage, duplicate/missing schema inputs, forbidden model releases, and asynchronous worker dispatch.

## Explicit non-capabilities and scientific boundaries

This slice does **not** register `product.mechanics` in the Platform/Go capability DAG. Current registered `product.rig` and `product.skin` still emit **v1** IR and cannot supply this executor's **v2** Rig/Skin inputs. That mismatch is fail-closed, not adapted through imaginary conversion.

`status=PASS` means **rest qualification for a specifically supplied typed Mesh/Rig/Skin triple**, not grip solving, IK interaction, graphics compositing, animation playback, generic unseen success, source↔canonical equivalence, published `.rss`, or production release. The output diagnostics explicitly set `product_revision_pass_claimed=false`; no ProductRevision/ModelRelease/Attempt authority is minted.

Motion v2 (#365) and Mechanics research (#353) remain separate. This PR does not touch Graphics/CAA, Corpus370, IRIS/TESSA, AXIS/MIRA training, source truth, renderer or site.

## Next product-critical chain

1. Land Motion V2 (#365) only when its own full required CI closes.
2. Audit and port V2 Rig/Skin producer worker routes and their typed ModelRelease gates; no V1→V2 reinterpretation.
3. Register the *complete* developer-only, no-product-authority product graph after all upstream contracts can deliver V2 IR. Separate `rest_execution` from `behavioral_capability` in the actual qualification receipts.
4. Qualify an actual artifact-bound rest→edit→deform→Graphics evidence path; keep #353 multigoal and V7 interaction research opt-in until real source/behavioral courts justify promotion.

**Merge gate:** dedicated tests, platform-Go, current-mainline and affected broad regressions succeed on exact PR head; review ownership of added dispatch. Do not bulk merge #297.
