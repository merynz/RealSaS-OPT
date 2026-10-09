# Texture seal and the connected 2D presentation boundary

Date: 2026-10-09. Classification: code/source review and implementation decision.
Inspected RealSaS snapshot: `fa482aa44e6efd3c89c8c0b0c2a68a36f69b875b`.
This review does not qualify a new asset, issue a scientific Attempt, or change
mechanics, appearance policy, production operators, or the PR #66 verdict.

## Decision

Seal the **complete appearance bundle**, using the existing CAA artifact and
qualification chain. Do not introduce another texture extraction stage or a
second appearance authority. A PNG digest alone cannot establish that the
required material exists, that its UV correspondence is correct, or that the
presentation consumer uses the qualified asset.

The useful product boundary is an immutable appearance asset bound to its
addressing, admitted support, provenance and qualification. Presentation may
move and arrange that material; runtime may only consume it. A new motion can
reuse the same asset when the new dynamic exposure proof passes. Newly exposed
unsupported material requires appearance work and requalification, rather than
runtime hole painting or a mechanics rerun.

## Existing implementation, and what each seal proves

| Existing authority | Relevant implementation | Guarantee and limit |
|---|---|---|
| Canonical compile field | `appearance_compile_v2.py`, `appearance_canonical_completion_v1.py` | Directional observed appearance stays source-backed; all-view-unseen canonical samples may receive deterministic offline completion. Calling the solver is not shipping qualification. |
| Stage22 `CAACompileSealIR` | `appearance_authority_v2.py`, `appearance_v2.py::seal_caa_compile_stage` | Pins compile/preregistration/NPZ identity and checks immutable direct samples and explicit provenance. A seal can retain unsupported samples; its report distinguishes total admitted support from total potential appearance. |
| Stage23 `CompleteAppearanceAssetIR` | `appearance_authority_v2.py`, `appearance_v2.py::bake_complete_appearance_stage` | Binds the compile seal, mechanical mesh, SurfaceAddressing, appearance domain, V0–V7 direction set, texture bytes, UV bytes, provenance bytes and atlas layout. |
| Stage24 qualification and Stage25 rest proof | `appearance_v2.py::qualify_complete_appearance_stage`, CAA support-admission contract | Establish source fidelity/completion quality and qualified rest visibility within the declared domain. Their denominator and mode must accompany a totality claim. |
| Stage42/package consumer and Stage45 dynamic proof | `runtime_v2.py`, `runtime_package_v2.py` | Must consume the exact qualified bundle. Canonical dynamic proof counts visible unsupported/padding provenance and completion exposure; a new motion/view scope needs its own proof. |

Normative references are
[`COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`](../../canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json),
[`CAA_V2_DEFORMATION_VS_DISOCCLUSION_CONTRACT_V1_20260927.json`](../../canonical/CAA_V2_DEFORMATION_VS_DISOCCLUSION_CONTRACT_V1_20260927.json),
and [`CAA_V2_RENDERABLE_SUPPORT_ADMISSION_CONTRACT_V2_20260927.json`](../../canonical/CAA_V2_RENDERABLE_SUPPORT_ADMISSION_CONTRACT_V2_20260927.json).

In particular, `C(p)` belongs to a canonical surface point, not a frame-local
screen hole. Motion can reveal its already baked value. It does not trigger
generation. The implemented all-view-unseen solver uses source anchors as hard
constraints and a surface-graph Dirichlet solve; it never modifies a sealed
mechanical surface or turns deformation into new appearance.

“100% defined” means no undefined visible samples within the qualified support
and declared scope. It does not prove the original, unseen design is known with
100% accuracy. Completion quality, source immutability, support admission and
dynamic exposure are separate predicates. Deliberate transparent artwork also
differs from unsupported material: alpha alone is not a totality certificate.

## The current scoped Knight consumer is source-direct

The completed downstream Attempt
`707e3b9b-91c6-4247-97dc-02b52ba0164a` used immutable execution code
`b936454362a7f5dcdba53920c6aa88a936bf508b`.
Its active V3 adapter delegates Stage37 and base Stage42 to
`presentation_research_v1.py` through the safety adapter.

`compile_source_domains_stage` reads the eight original `cfg.views[*].source`
PNGs and verifies their handoff hashes. `compile_projection_stage` places those
same textures into the projection. Its `appearance_asset_binding_hash` is the
hash of the texture digest list; its `appearance_qualification_binding_hash`
is the source handoff digest. These are scoped source-input identities, not the
asset hash and qualification hash of a completed canonical CAA bundle.

Consequently, the current `PASS_DEMO_ONLY` does not prove consumption of hidden
canonical completion. The current V3 Stage45 also does not replay the canonical
completion exposure/unsupported-support predicates. Its actual owner and parity
checks remain valid for their narrower predicates.

The full CAA source-owned visual branch similarly sets
`total_defined_fraction=1.0` only with the explicit totality domain
`SOURCE_OWNED_VISUAL_MESH_ONLY`. Completion and cross-view appearance are unused
in that mode. Do not reinterpret that value as complete material underneath
every armor overlap or on the unseen side of the latest mechanical carrier.
Conversely, an older canonical CAA receipt cannot be substituted without exact
carrier/addressing/domain/asset identity qualification.

Before claiming appearance closure, the downstream consumer must record its
mode and exact appearance/support/qualification dependencies, and Stage45 must
measure exposure against that same domain. Source-direct diagnostics can remain
source-direct; they must not implicitly claim completed hidden-surface coverage.

## Paper review

Primary source: Smith, He and Ye,
[Animating Childlike Drawings with 2.5D Character Rigs, arXiv:2502.17866v1](https://arxiv.org/html/2502.17866v1).
Reviewed the supplied HTML's method, evaluation and limitations; no 2025 solver
implementation or supplementary video was executed.

Sections 4.2–4.3 separate prepared textures, directional pose retargeting,
ARAP deformation and explicit triangle ordering. Section 4.3.4 addresses
projection instability. Section 5.1 reports a held-object inconsistency during
limb remapping. Section 7 limits the method to upright bipedal drawings without
overlapping parts. It is not a hidden-material correctness certificate.

## RealSaS implementation consequences

The following are our design decisions, not claims that the paper proves Knight
correctness or provides a drop-in solver:

1. Derive one connected presentation pose from the existing frozen hierarchy
   and witness. Decide projected direction and per-view authored length together;
   preserve the root transport and parent/child attachment relations. Do not
   independently fit each pivot while discarding foreshortening. The existing
   [structural counterexample](SPINE_RUNTIME_GAP_AUDIT_20261009.md) shows why that
   operator can split a joint even under identical rigid mechanical motion.
2. Keep canonical left/right joint and prop identity stable. Sword and shield
   remain bound to their qualified target slots across V0–V7; view styling must
   not silently exchange equipment ownership. Prove setup identity separately
   from animated frame0, and check transitions and loop boundaries.
3. Treat near-zero projected direction as an explicit presentation ambiguity.
   Any bounded resolution/temporal policy must be declared and independently
   checked; it cannot invent another mechanical rig. Do not add per-limb plane
   optimization without proving the shared contacts it may affect.
4. Reuse the existing canonical visual coefficients and qualified seam/contact
   relations. UV cuts cannot create another motion owner. Bounded triangle
   repairs must preserve those relations. A replacement ARAP skin solve is not
   authorized by this review.
5. Compile object and, where needed, within-object drawing ownership. Mechanical
   depth may inform the policy, but a geometric depth field is not itself the
   final 2D art contract. Preserve independent overlap/tie/cycle/transition proofs.
6. Separate missing material from missing screen geometry and wrong occlusion.
   A complete texture cannot fill an absent triangle or reconnect independently
   translated charts. Stage45 must measure qualified joint/contact residuals,
   coverage loss, unsupported appearance exposure and semantic occlusion, in
   addition to healthy triangles and native/reference parity.

Implement and compare downstream-only child Attempts on the same upstream
artifacts/probes, and use the existing controlled owner-attribution service only
where matched interventions justify attribution. Fresh native IDLE/RUN/SLASH
across all eight views follows that proof. This review is preparatory evidence;
this review did not implement the connected pose or expanded Stage45. The later
[connected pose intervention](CONNECTED_POSE_INTERVENTION_20261009.md) implements
the palette and independent relational gate, while final chart contact, dynamic
coverage and semantic overlap qualification remain open. No fresh render or new
scientific verdict is claimed by this preparatory review.

PR #65 stays merged. PR #66 stays open for visual closure. Mechanics, rig, skin,
IRIS, AXIS and MIRA stay sealed. Appearance quality and absolute product/G3
authority remain distinct boundaries; optimize after accepted visual results.
