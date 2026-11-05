# Corpus370 — Blind Original Artist-Layer Gold Benchmark V1

Research design: 2026-11-05. PR #327 is a DRAFT. The independent partition court and negative tests exist; **no real artist-authored project has passed this new benchmark**. Never equate source mesh-object labels with artist drawable ground truth.

## Core experiment

Question: after flattening an ORIGINAL human-authored, genuinely layered character into a single anonymous merged surface/appearance representation, can RealSaS recover a geometrically and behaviorally equivalent editable partition, without ever seeing the author's original layer names/count, rig identifiers or slot membership?

There are TWO separate independent authorities:

- ALGORITHM INPUT: anonymous merged triangles/appearance (or image-only on the consumer track) stripped of source layer/slot/attachment/bone/mesh-object identities. Source layer count and boundary flags hidden. Input/artifact/seed/algorithm frozen before unsealing reference.
- COURT-ONLY REFERENCE: the untouched original layered artwork with artist edits, true layer opacity/alpha, draw order, attachment/slot changes, optionally original authored animation and source mesh face mapping. Immutable source SHA, explicit license permission and verified author identity; no generated pseudo-labels.

Source mesh objects, UV islands, material names, bone controls and triangle connectivity may be useful geometry baselines but CANNOT stand in for real artist 2D drawables. Native rigged 3D source asset object IDs must be treated as mechanical source proxy only. Existing CAT/TRIC/COW pilot is a mechanics/native edit pilot, not genuine author-layer ground truth.

## True source cohorts, never conflate

A. Human-authored 2D editor rig/source: Spine JSON+atlas+original part layers, or authored cutout project, with real slots, attachments, clipping and draw order, when legally usable. The exported Spine slot/attachment is not one-to-one with bone controls or a fixed number of artist layers; attachments can change over time.
B. Layered human-authored PSD/KRA/Aseprite: true source art masks and alpha, optional layer ordering. **2D-only** if no source mesh/rig. Do not invent a 3D correspondence.
C. Blender/GLTF/FBX native components: a SECONDARY mechanical/3D object source baseline. It cannot be scored as author drawing-layer reconstruction by default.

Possible rights/quality-screening leads (not imported or training-authorized):
- Official Spine sample/source exports: https://us.esotericsoftware.com/spine-skeleton-viewer ; licensing needs separate review.
- Hand-drawn free separated-part layered PSD, no Spine rig: https://grabitready.itch.io/starter-platformer-hero ; personal/commercial license, cannot assume model training permission or permission to redistribute original PSD.

In the connected repo and Drive, no source has yet been independently verified to supply the exact artist-layer-to-merged-face mapping. Do not invent one.

## Flattening and leakage protections

1. Archive the true original source and source author rights. Hash exact raw bytes. Preserve original visibility, per-layer alpha, original order and source pose/skin.
2. Export true author-layer ID masks, optional source triangle attachment ID, edit/visibility and per-pose relations into a COURT-ONLY immutable file.
3. Produce merged anonymous triangles and/or full composite without source layer names, IDs, count, bone/mesh names or slot metadata. Preserve only explicitly admitted input-domain geometry, UV, image and camera. Prove layer-composite == merged input render with alpha, including overlays and occluded artwork. Fail closed when hidden source art was not recoverable.
4. Lock the algorithm branch/version/parameters/predictions before opening the private artist reference. For image-only product tests, restrict model to product image input, NOT privileged merged author geometry.
5. Require exact common face/pixel address binding for reference-vs-prediction court. Opaque numeric IDs and SHA receipts only. Existing court rejects 3D object-only sources presented as human 2D layer truth.

## Implemented first scoring contract

Code: tools/corpus/court_artist_blinded_partition_v1.py
Negative/CI tests: tests/corpus/test_court_artist_blinded_partition_v1.py

- Hungarian permutation-invariant mean IoU and matched source area. Layer IDs can be arbitrary without changing the score.
- Area-weighted variation of information, source author fragmentation/oversegmentation, predicted merged unrelated author parts, and unknown-area coverage. A method that labels only easy regions gets low coverage even if its conditional IoU is high.
- Reject leaked author layer count and names, wrong merged-surface/source-address SHA, false human author ground truth, uncommitted predictions, corrupt label/weight masks, and any implicit training authority.
- The initial court measures STRUCTURAL similarity only. It is not independently certified behavioral equivalence, authored attachment, draw order or materialization.

## Required second court: behavioral edit equivalence

Use matched artist-vs-model source regions (not fixed K), and perform independently rendered hide/remove, isolate, local move/deform, draw order swap, attachment reveal and heldout source-authored animation states.

Compare RGBA + alpha, occlusion/exposure, seam coverage, reverse/move persistence and change-under-novel-pose against actual source artist edits. Same motion does NOT mean SAME_DRAWABLE and differing deformation does NOT mean DIFFERENT_DRAWABLE. Multiple visually/behaviorally equivalent valid partitions may differ from the author's editorial grouping. Report BOTH exact author agreement and behavior, never a single magic layer accuracy score.

Run baseline geometry-only connected components, UV/material islands (labeled explicitly as proxies), fixed K, prior V7-V12 geometrical partition, and source native V13b response partitions on identical permitted readsets. Hold out WHOLE AUTHORS / source packs / character families, not merely random frames from a familiar artist.

## Completion and authority gates

A real original human-art authored layered asset and license must be verified; exact flatten reconstruction/parity and hidden reference must be sealed; the source adapter must prove face/pixel membership and alpha; predictions are locked before artist labels; independent court produces split/merge/IoU/coverage and behavioral-edit comparisons; unseen authors/families and occlusion cases evaluated separately.

**Research benchmark pilot only, no artist-layer recovery result yet.** No source ownership, IRIS/TESSA training, materialization or ModelRelease authority until independent proof gates are met. The V14 source native 3D-to-RGBA mechanical court remains complementary: it establishes whether causes change images, whereas this new benchmark asks whether those effects reconstruct the actual artist editable representation.

## Directly inspected external true layer source candidate — NOT an admitted bench subject

The official source export in EsotericSoftware/spine-runtimes (4.3 reference,
path spine-android/app/src/main/assets/spineboy-pro.json) was read directly
through GitHub as an independent original authored rig *candidate*. Verified
source Git blob SHA: 4a58fecd32a97bad83a9169056c44db3f17dba4e.
Decoded metadata: 52 original slots, one default skin, 12 mesh attachments,
66 region attachments and 11 animation names. Those are NOT 52 art layers or
proof of artist drawable correspondence; some attachments are alternatives
within a slot and appearance changes over time. No original source artwork,
runtime code or sample files were copied into the RealSaS repository.

Source metadata reference:
https://github.com/EsotericSoftware/spine-runtimes/blob/4.3/spine-android/app/src/main/assets/spineboy-pro.json

Official runtime repository says its examples can be evaluated free of charge;
that is NOT proof of rights for redistribution or ML training. This is a
candidate for private evaluation only, after an explicit source/project,
atlas/original-part material and licensing check. To avoid humanoid-only
sample bias, a nonhumanoid authored rig/source from a different asset family
must be tested as a heldout subject in the same blind benchmark.
