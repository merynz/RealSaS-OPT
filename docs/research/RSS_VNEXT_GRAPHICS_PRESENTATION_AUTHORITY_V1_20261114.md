# RSS.vNext — Graphics-Owned 2D Presentation Authority

**Architecture decision: 2026-11-14.** This is the active two-image product
path decision, not a retroactive rewrite of historical Knight CAA experiments.

## Decision: Graphics is the consumer/owner

`3D Mechanics + game-profile-dependent 2D Presentation` is the contract.
**Graphics** owns the complete compiled appearance/presentation program:
which canonical drawable face is painted, from which texture/provenance
layer, at which pose/view/depth/order, and how alpha is composited.
**CAA (Complete Appearance Authority)** is **NOT** an additional
competing model head, independently privileged compositor, canonical-mesh
owner or separate output authority. Its useful residual obligations
(source-preserving color/alpha, deterministic completion, provenance,
unknown coverage, atlas integrity, immutable source pixels, shipping
baked assets) are **policies/courts INSIDE Graphics**. Historical
`compiler/realsas_compiler_core/appearance_authority_v2.py` and Knight
reference renderers remain as evidence/reference, not the active
`product.graphics` execution authority.

The old historical CAA contract's mandatory 8-camera output assumption
does **not** constrain the current product. A side-scrolling game may
render one direction, an isometric game another, or the user may select
an eight-yaw profile. All share the same editable 3D Mechanics identity.

## Concrete ownership map

| Producer | Actual emitted data | Consumer decision | Authority |
| --- | --- | --- | --- |
| Two uploaded original images | Front/back RGB/alpha and digests | Preserve pixel color/alpha; no fabricated third observation | Direct original **image-plane** observation only |
| IRIS shared H(x) and Graphics head | Per-vertex RGB/density/uncertainty candidates | Persistent face-addressed UV candidate completion for unseen surfaces | Predicted appearance, **not source surface ownership** |
| Semantic head on same H(x) | Named K-way presentation-role logits | Candidate face-indexed drawable strata with UNKNOWN/abstain | Suggested partition ONLY; not contact, opacity, z/order or visibility |
| TESSA / CanonicalMesh | Face indices, vertex IDs, structural adjacency | Stable UV charts, deformed triangle addresses and first-hit surfaces | Qualified mechanical mesh where separately admitted |
| AXIS / MIRA / Mechanics | Rig, skin, controls, LBS/IK/FK poses, sockets | Per-frame projected 3D triangles and depth; attachment via independent court | Mechanical action, not automatically appearance source truth |
| Graphics Compiler | Source-projected RGB candidates and same-H UV, typed region relations, pose/view policy | Present/occluded faces, explicit qualified order constraints, alpha and compositing | Sole **product 2D presentation** decision maker |
| Native C++ renderer | Mesh-indexed UV candidate, physical depth and typed controls | Exact 2D pixels, pixel face owner and provenance label | Executable preview; not source or release Proof by itself |
| Product Proof | Independently replayed source CAS, model CAS, mesh, controls, UV and .rss | Accept/abstain from shipping ProductRevision | Independent final product authority |

The original views are **uncalibrated uploads**. Projection of those RGB
samples onto an inferred mesh is a **candidate correspondence**; a native
pixel marked `source_candidate` must NOT be mislabeled verified
`source_3d_surface_ownership`. H(x)-generated colors of hidden faces are
`inferred`, not historical observations. Unsupported pixels remain
unknown/transparent rather than silently reclassified.

## Graphics-layer executable policy

1. **Canonical appearance carrier**: one immutable mesh-addressed UV
   chart per face; two independent original RGB/alpha candidate layers
   and one inferred same-H appearance layer. CAA-like source preservation
   checks (mask, conflict, no cross-face UV bilinear leakage, unknown
   quarantine) act before native packaging. Explicit contact/source
   surface matching is a separate qualification and is NOT inferred
   from sparse original images.
2. **Semantic is advisory**: K-way `presentation_role_logits` come from
   the same scene snapshot and TESSA vertex address order as Graphics
   logits. Aggregate vertex softmax by the exact face winding; retain
   K probabilities, candidate role, explicit `255` abstain. This
   does not create a fourth source view, person-specific segmentation,
   depth, draw-order precedence or a new 3D mechanical structure.
3. **Optional drawable views/layers**: the default full puppet never
   hides a face based on an uncertain learned role. When a user explicitly
   requests a role stratum, the runtime renders only that role's
   **visible contribution**, retaining the COMPLETE physical depth
   and alpha occluder stack, including unselected strata. Thus a rear
   face never leaks through an opaque front face merely because the front
   belongs to another candidate role.
4. **Moving frame visibility**: recompute first-hit ordering from every
   frame's **posed/deformed** canonical triangles and the selected
   orthographic camera. UV/color indices remain face-index stable.
   Explicit draw-order exceptions need separately admitted evidence;
   anonymous learned source-region logits alone cannot override
   physical depth or become source contact.
5. **Persistence**: native `RSUVT002` extends legacy `RSUVT001` with
   per-face K probabilities, model-only role selection and role-schema
   digest. Legacy archives continue reading. `.rss` control edits
   reproject the same mesh/rig/skin, never rerun image diffusion. Native
   player can render a role pass with `--role=K`; default rendering
   remains full-puppet. Role 255 abstains rather than dropping the face
   from default rendering.
6. **Independent replay**: `product.proof` regenerates source-projected
   candidates, RGB+role atlas, binary texture and preview from exact
   source/scene/checkpoint CAS, then byte-compares output. Model candidate
   roles cannot pass missing independent source appearance/visibility,
   attachment/contact or ProductRevision courts.

## What is and is not proved

**Executable regression scope:** source-vs-inferred color separation;
independent front/back sample masks; exact face-chart sampling;
source RGB unchanged by changes in Semantic role logits; face-indexed
candidate roles and abstention; native physical occlusion under motion
and orthographic depth; optional layer compositing with intact physical
occluders; native preview edit/reopen; independent Proof CAS replay
and fail-closed unresolved source authority.

**Still not qualified:** unseen reconstruction fidelity; watertightness
of arbitrary morphology; complete newly-exposed underside textures;
independently grounded hidden-surface source truth; exact learned
drawable/cut/attachment boundaries when faces cross semantic regions;
per-pose nonphysical 2D illustration order exceptions; source-contact
and secondary-motion behavior; the source-camera mapping for an
uncalibrated upload; or full ProductRevision release.

The first-party native renderer is a bounded **CPU correctness/reference**
renderer, not proof of a production GPU renderer. No training, inference
quality or released-game-editor integration is implied by green CI.

## Paper/product cross-check

Spine separates mechanics (bones), visual attachments (slots) and
animation-dependent **draw order**. Live2D Cubism differentiates ArtMesh
deformers, clipping, opacity and draw order that can depend on parameters.
These support giving Graphics independent 2D presentation authority,
without throwing away canonical 3D mechanical first-hit depth. RSS.vNext
does not copy either implementation or treat their authored information
as available from a mere two-image upload.

- https://en.esotericsoftware.com/spine-slots
- https://docs.live2d.com/en/cubism-editor-manual/draworder/
- https://docs.live2d.com/en/cubism-editor-manual/clipping-mask/

**Before trainer freeze**: convert remaining **independent source
admission** needs into supervision masks/qualified receipts, not new
unverified labels. Training objectives can improve H(x) proposals but
must not make unknown source authority look observed.
