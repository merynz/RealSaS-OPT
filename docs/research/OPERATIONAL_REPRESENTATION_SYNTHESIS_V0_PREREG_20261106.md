# Operational Representation Synthesis v0 — preregistration and authority boundaries

Date: 2026-11-06. Base: PR #355 (Cat source-authored pose physical experiment). Draft research only.

## Question
Given **E** (source observations and provenance), **C** (operations required of an editable puppet), and **P** (optional representation cost preferences), can one generic procedure construct nontrivial candidate representations, execute source-addressed edits under deformed 3D geometry and depth compositing, and preserve observationally indistinguishable alternatives without inventing artist-owner truth?

## Immutable pre-implementation choices
- No subject-specific branch, anatomy labels, or source-derived artist owner tags.
- Retain existing `CanonicalMeshIR`, `CanonicalSkinIR`, `DrawableSurfaceIR`, `CompiledGraphicsV1`, `visibility_v2` and `appearance_render_v2`; only *add* a research adapter. Do not activate a product pipeline stage.
- Distinguish **face packetting** (drawable) from **atlas edit controls** (independent address handles) and from **pose transport** (barycentric UV carried by canonical faces), and from **per-pixel compositing** (existing CAA depth K4).
- Strong packings: addressed aggregate, source-group-shared-carrier (when present), bounded face chunks, per-face stream. All preserve independent address controls and depth compositor; no strawman mandatory flat order.
- No observed original owner labels, one-image student inference, or Corpus370 / IRIS / TESSA training admissions. Physical source arrays remain private. Teacher-mode metadata must be explicit.
- The independent source court is not the synthesis algorithm. Source original holdout render arrays may enter a separate qualification process only *after* candidates are compiled. No target-pixel leakage into candidate construction.
- The full puppet contract is **UNKNOWN** until a runtime artifact passes independent source consequence tests. A successful reference-render contract is not proof of a shipped GPU/web runtime.
- Nonbinary alpha, layer overflow, exact-depth ties, attachments and secondary motion must remain UNKNOWN or fail closed unless source evidence plus supported operations qualify them.
- Different package layouts with the same observable capabilities form an equivalence set. Structural packet counts may be reported, but absent independent timing measurements all latency/VRAM/draw-call metrics remain `UNMEASURED_COST`.

## Milestones / falsification
1. Contract + generator: E/C/P typed separately, reject malformed source binding, generate three-or-more generic structural layouts, preserve equivalence if operations agree; ensure missing capability is UNKNOWN, never PASS by default.
2. Runnable reference puppet: apply RGB invert by *top-left* atlas address (alpha unchanged), deform by supplied posed canonical positions, use existing UV + CAA depth rendering, preserve hidden edit bits and emit per-operation diagnostics. Test on authored synthetic two-pose geometric fixture with independent expected semantic outcomes, and both opaque and alpha-sensitive cases.
3. Source original: bind Cat's exact source teacher geometry, texture, two authored poses and camera/sample receipts to independent physical source arrays; compare each compiled candidate to independent targets at preregistered 0.025 premultiplied RGBA tolerance, without retuning. Then Triceratops local-depth case. These tests are NOT claimed done by Milestones 1-2.
4. Heldout: morphologies outside development; attachment/release and alpha tests before product promotion.

## Interim research qualification
The existing compiler CAA renderer itself is reused, not independently reverified here. Passing its own tests is *executor consistency*, not independent Cycles parity or unseen inference. Every candidate remains `research_only=true`, with `training_authorized=false`, `owner_authorized=false`, `materialization_authorized=false`.
