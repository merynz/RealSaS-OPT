# RSS.vNext open UDF → indexed mesh: DCUDF2 backend decision

**Date:** 2026-11-17. **Scope:** source-independent candidate extraction and
TESSA indexed geometry address; no GPU production qualification, dataset
mutation or checkpoint training performed.

## Implementations and license

- Paper: Chen et al., *DCUDF2: Improving Efficiency and Accuracy in Extracting
  Zero Level Sets From Unsigned Distance Fields*, IEEE TVCG 2025,
  https://arxiv.org/abs/2408.17284.
- Primary original implementation: https://github.com/jjjkkyz/DCUDF2,
  MIT, Copyright (c) 2023 jjjkkyz (as in upstream LICENSE; retain license
  and copyright notices when shipping copied code). The pinned **Git blob
  hashes** for mesh_extraction.py, mesh_cut.py, VectorAdam.py and LICENSE
  are declared in `models/rss_vnext/dcudf2_open_backend_v1.py`.
- Historical DCUDF upstream: https://github.com/jjjkkyz/DCUDF, MIT.
- DCUDF README acknowledges https://github.com/junshengzhou/CAP-UDF,
  also MIT (its own copyright still applies).
- MeshUDF's published code is GPL-3.0. **Do not copy that source into a
  closed-source BUSTED binary without separate legal clearance.**
- Third-party runtime dependencies must be licensed and version-pinned
  independently before deployment; MIT on DCUDF2 does not license all
  transitively imported software.

## What is implemented

1. An **opt-in local unsigned-gradient surface mesher** in
   `open_udf_surface_v1.py` used for regression/ablation only. This
   local method is *not* DCUDF2 and is not a production-quality substitute.
2. A wrapper for the exact MIT DCUDF2 implementation in
   `dcudf2_open_backend_v1.py`. The external checkout is compared with
   pinned Git source blobs before import. The implementation **forces
   is_cut=True**; its original default is false.
3. A post-extraction court rejects failed cuts that return a double-covered,
   closed, or degenerate surface, as well as nonmanifold edges, dangling
   open boundaries, orphan vertices and large UDF residuals. It never
   substitutes signed FlexiCubes or silently treats a closed double cover
   as a valid single-layer garment.
4. `hybrid_signed_open_iris_seed_v1.py` builds one *disjoint indexed M0*
   from signed and open candidates tied to the exact same H(x) scene SHA.
   Body/cape XYZ coincidence cannot weld their vertex IDs. The resulting
   M0 is the one TESSA will refine before AXIS/MIRA.
5. A **dedicated isolated CUDA process** must call the original optimizer.
   Its fixed `experiment/optimizer` progress exports require a unique
   workspace and no process-global `chdir` in shared model workers.

## A critical additional learning requirement

A single global `unsigned_distance(x)` head is **NOT sufficient** to
extract a particular open cape/hair card from a crowded body scene.
`unsigned_distance(x)` measures distance to the nearest surface across
potentially overlapping roles. Supplying an external string such as
`open_layer_name='cape'` does NOT transform it into a cape-specific UDF.

Therefore the DCUDF2 runner requires
`query_open_component_udf` and a declared `source_role_conditioned=True`.
This declaration alone is never a source qualification; a learned
component-conditioned UDF query, corresponding independently admitted
native source role-and-open-topology supervision, per-asset causal component
identity, and open-vs-closed routing receipt **remain required** before
the product runtime is allowed to invoke it. The present unified model has
a global UDF head and no independently qualified component UDF decoder.

Likewise, the DCUDF2 optimization returns a detached mesh. An end-to-end
gradient through topology selection and global double-cover cutting is NOT
proved by allowing its internal vertex optimizer to use torch gradients.
The production trainer must separately supply a source-qualified field
loss and source-native topology/mechanics targets to the shared IRIS and
TESSA heads. The gradient of a local surrogate is not a certified
differentiable DCUDF2 extraction.

## Independent release requirements — all FAIL CLOSED until measured

- GPU: pinned official DCUDF2 actually succeeds on learned open UDFs of
  capes, skirts and hair cards, with zero silent cut failures.
- Geometry: source-native open hem identity, no double cover, no missing
  component, layer matched Chamfer/normal/area and all-face graph courts.
- Topology: bounded face count, no cracks/T-junctions or unwanted
  cross-component bridging, no uncontrolled smoothing of thin features.
- Dynamics: exact M0→TESSA→M*→AXIS/MIRA, G3 and G3B on full source-qualified
  pose envelope plus independent Stage35 actual clips; optional pinned
  XPBD poses when qualified and collision constraints are present.
- Graphics: deforming shared M* retains UV/appearance, source-visible
  occlusion and correct 2D draw-order/depth behavior.
- Licensing: record transitive notice and reproducible build of the
  exact external source, not just its GitHub README.
- Evaluation: compare official DCUDF2 with local UDF baseline and
  signed/FlexiCubes fallback on *the same open-native-source cases*.
  Compare topology, UV, dynamic metrics and runtime. Do not call one
  a replacement before the experiment.

**Status:** integration scaffold + testable fail-closed courts, NOT
production UDF or animation-ready acceptance. No source corpus or
checkpoint has been touched.
