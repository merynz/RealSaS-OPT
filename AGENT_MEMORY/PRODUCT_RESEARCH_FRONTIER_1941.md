# Product Research Frontier — 1941-Commit Knight Trunk

> **NON-AUTHORITATIVE FORENSIC MAP.**
> This file describes the latest observed product-research frontier and how it relates to the normalized candidate. “Latest frontier” does not mean “scientifically final truth”.

## Identity

The user-remembered 1941-commit product-research trunk is:

`6ed060351512a5f76a206acce011b585f48b1768`

Relative to the current rolled-back `main`, it is exactly **1941 commits ahead** at the time of reconstruction.

Two late branch heads share this trunk:

- `research/vf11-certified-adaptive-20260921` = trunk + one render-rerun commit;
- `audit/knight-rest-visual-owner-20260930` = trunk + 17 visual-forensic commits.

The branch name “VF11” is historically misleading. From 21–29 September the lineage evolved into the full Knight product-research spine: geometry, GSA, mesh, mechanics, appearance, source-owned visual presentation, motion, runtime and orchestration.

## Central reconstruction result

The normalized backup:

`backup/main-normalized-20260930-3529344`

preserves almost all **living compiler/runtime product logic** from the 1941 trunk.

Tree comparison shows:
- compiler production code difference: **one file** — `runtime_v2.py`;
- native runtime difference: **zero files**;
- key geometry/mechanics/appearance core files are byte-identical;
- most remaining differences are research workflows, audits, canonical evidence, tools and historical IRIS v3–v5 research implementations.

Therefore the 1941 trunk should not be wholesale-merged. It is primarily the scientific/provenance explanation for why the normalized code looks the way it does.

## Living frontier already preserved by normalization

### Geometry / topology provenance

The late trunk established:

- Stage35 may not delete failing faces as a hidden repair;
- repairs are represented as typed repartition directives;
- exact dense-face provenance survives through Stage15;
- Stage18 uses exact face provenance and may not mint triangle cliques from pairwise edges;
- Stage14 normals must be topology-local under policy v3;
- Stage35→Stage17 repair requires authorized child-attempt semantics.

Representative commits:
- `bb860547...` Stage35 face deletion → repartition directive
- `f7b9b84f...` exact dense-face provenance through Stage15
- `ee299e8d...` Stage18 exact-face-provenance authority
- `191d8ed2...` topology-local Stage14 normal policy v3
- `0e975fd1...` authorized Stage35→17 repair child attempts

The relevant `mesh_v2.py` and `iris_geometry_v2.py` blobs are byte-identical in the normalized backup.

### Mechanical seam / topology closure

Late product research adds:
- component-harmonic Stage18 seam support;
- Stage35 source-edge probe seeds;
- Stage17 repartition closure;
- no face deletion;
- no cross-component support leakage.

Production replay court:
`KNIGHT_PRODUCTION_TOPOLOGY_REPLAY_COURT_V1_20260929.json`

Observed:
- direct source-edge seeds: **1417**
- final separate constraints: **1818**
- Stage17 components: **73**
- Stage18: **16,031 vertices / 28,331 faces**
- face deletion count: **0**
- mechanical transfer: `COMPONENT_HARMONIC_DIRICHLET_V1`
- idle/run/slash: **edge_gt_4 = 0; edge_gt_10 = 0**
- worst observed motion edge ratio: about **3.616**

Production G3 motion court:
`KNIGHT_PRODUCTION_G3_MOTION_COURT_V1_20260929.json`

- G3: PASS
- failure invariants: none
- 169 G3 probes
- same 0 edge>4 / 0 edge>10 actual-motion signature
- measured court total about 65.6 seconds

These are strong Knight product-research measurements, not universal product authority.

### Appearance / CAA

The last living appearance sequence:
- completion components derive from source topology;
- CAA graph topology is decoupled from mechanical vertex duplication;
- Stage24 qualifies on the same source-topology graph as compile;
- local harmonic completion is primary;
- cross-view donor is a gated fallback;
- Stage24 holdout mirrors the same harmonic-first fallback;
- donor fallback is constrained by frozen compatibility cuts;
- incompatible cross-view donation fails closed.

Representative commits:
- `0e370046...`
- `63bcd2b7...`
- `463a4163...`
- `587dac3b...`
- `0c8d5647...`
- `53f85888...`
- `39051343...`
- `70cbd671...`

The core implementation and subject-free compiler tests for this sequence are byte-identical in the normalized backup.

## Source-owned visual presentation: working research path, not yet canonical runtime

The 1941 trunk contains a real working visual render path, but it is explicitly marked demo/rescue evidence rather than product authority.

The path uses:
- Stage18 `VisualMeshSetIR`;
- original source RGBA;
- Stage23 appearance asset;
- Stage28 skeleton;
- Stage32 skin;
- motion preset/source;
- visual→mechanical binding;
- ARAP2D visual deformation;
- C++ source-owned visual mesh renderer.

The strongest operational prototype is:

`tools/demo/render_knight_solution_visual_resume_v1.py`

Its report contract says:

```text
PASS_RENDER_ONLY_RESUME
stage17_25_recomputed = False
source_owned_visual_mesh_rebuilt = False
source_texture_rebuilt = False
mechanical_candidate_rendered_directly = False
renderer = CXX_VISUAL_MESH_V1__ARAP_2D
```

This is valuable product behavior: render-only reuse is already demonstrated.

### Why normalized Stage42 fails closed

The 1941 trunk’s canonical `runtime_v2.py` does **not** understand source-owned visual presentation geometry; the working visual path bypasses Stage42 through demo tooling.

Normalization changes the canonical runtime to detect source-owned visual mode and fail closed with:

`RUNTIME_V2_SOURCE_OWNED_VISUAL_PRESENTATION_BINDING_REQUIRED`

This is a correctness improvement, not a regression.

Correct recovery principle:

> **Preserve normalized fail-closed honesty, then productize the trunk’s proven visual execution primitive as a typed Stage38→42→runtime-package→native carrier.**

Do **not** remove the fail-close and silently fall back to mechanical render topology.

## 1958-tail finding: mechanics closed, remaining smear is visual/presentation

The 17-commit tail ending at:

`643714de28999a7de9d0b25733a290136dc374c7`

contains audits/workflows/tools only; it does not introduce new production compiler-core semantics.

### Rest-pose ownership

Corrected rest audit:
- V0 extra alpha: **1.44%**, recall ~**99.48%**
- V2 extra alpha: **2.07%**, recall ~**99.47%**
- most extra pixels have identity-face contribution, not only generated-face contribution

So even rest-pose excess cannot be explained purely by generated topology.

### Renderer mechanical-support counterfactual

With the canonical mechanical support contract:
- idle/run/slash: **edge_gt_4 = 0**, **edge_gt_10 = 0**
- worst edge ratios remain around 2–3.6

With the old/wrong renderer support:
- hundreds to thousands of edge_gt_4 failures
- extreme edge stretch roughly 40–166×

Therefore a major catastrophic-smear owner was renderer skin-support mismatch, and the canonical support contract fixes that mechanical failure.

### Remaining visual smear

After canonical mechanical support is used, dynamic extra alpha remains substantial:

- idle: V0 **5.33%**, V2 **51.93%**
- run: V0 **29.26%**, V2 **46.73%**
- slash: V0 **48.41%**, V2 **47.11%**

while mechanical edge_gt_4 / edge_gt_10 remain zero.

Most extra pixels are attributed to non-seam faces; many representative owners are rest-visible.

Current frontier interpretation:

> catastrophic mechanical deformation is no longer the dominant owner; residual error belongs primarily to visual presentation / projection / visibility-exposure semantics.

This is an **open frontier**, not a solved product claim.

## Final 1958 forensic workflow did not falsify the visual hypothesis

Run `36640358982` successfully:
- checked out the audit branch;
- installed dependencies;
- built the native source-owned renderer.

It failed before the forensic rendering logic because:

`RESCUE_PARENT_ARTIFACT_MISSING: .../18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json`

for the requested parent run.

This is an **artifact continuity/orchestration failure**, not a scientific visual-proof failure.

It is also direct evidence for the post-main platform mandate: a valid research result should not become inaccessible merely because the next run cannot discover/reuse its immutable Stage18 artifact.

## Historical model research that should not be blindly restored

The normalized tree omits many `models/iris/v3`, `v4`, and `v5` files present on the 1941 trunk.

Current product adapters use current typed execution/checkpoint authorities and historical checkpoint admission where explicitly allowed. The old research implementations are not automatically active execution owners.

Disposition:
- preserve as Git/provenance evidence;
- do not restore as co-current product implementations unless an exact current dependency proves they are required.

## Latest product-research frontier classification

### LIVING_AND_NORMALIZED

- topology-local Stage14 normals
- exact dense-face provenance
- Stage35 repartition directives instead of face deletion
- Stage17/18 source-edge partition closure
- component-harmonic mechanical support
- source-topology CAA
- harmonic-first completion
- compatibility-gated donor fallback
- normalized source-owned visual ownership split through Stage38

### LIVING_BUT_NOT_PRODUCTIZED

- source-owned VisualMeshSet ARAP/native rendering primitive
- render-only sealed-artifact reuse path
- visual→mechanical binding execution
- source RGBA / source-owned visual presentation geometry

These work as product-research primitives but are not carried by canonical Stage42/runtime package.

### POST_NORMALIZATION_GENERIC_FIX

- Stage35 G3B stress must bind to sealed deformation-envelope joint ranges:
  - `25e765e8...`
  - `16b8cb72...`
- independently revalidated subject-free on 2026-10-01:
  - **436 compiler PASS, 4 skipped**
  - no Knight coupling

### HISTORICAL_RESEARCH_ONLY

- old VF11 experiment machinery
- historical IRIS v3–v5 research implementations not required by current product adapters
- hundreds of one-off Knight court workflows/tools once their result is captured in the frontier ledger

### OPEN_FRONTIER

1. typed source-owned visual carrier Stage38→42→package→native;
2. residual visual projection/exposure ownership;
3. repository wiring-test contract drift;
4. native runtime source-seal drift;
5. Quaternius distinct-take qualification;
6. professional artifact continuity and cross-run reuse.

## Main recovery implication

The recovery base should be the **normalized engineering lineage**, not a wholesale replay of 1941 research commits.

The 1941/1958 history supplies:
- why the normalized core decisions exist;
- which late decisions were empirically won on Knight;
- which working primitives were not yet productized;
- which apparently-red runs were orchestration failures rather than scientific falsifications.

The target recovery is therefore:

```text
normalized lineage
+ post-normalization generic Stage35 envelope fix
+ canonicalize the proven source-owned visual render primitive
+ resolve residual visual presentation/projection frontier
+ exact-current gates and seals
= proposed canonical recovery head
```
