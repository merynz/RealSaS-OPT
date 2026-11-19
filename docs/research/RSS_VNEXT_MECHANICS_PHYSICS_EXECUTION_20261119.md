# RSS.vNext: Mechanics.Kinematics / Mechanics.Physics execution contract

**Decision 2026-11-19.** Product path. This freezes responsibility and
safety boundaries; it is not a new physics teacher nor an unseen-release proof.

## Representation and execution

`IRIS/SLR -> TESSA M* -> AXIS rig/controls -> MIRA skin -> Compiler` is one
canonical puppet. No parallel cloth mesh or fifth independent neural expert.

**Kinematics** consumes the admitted AXIS control program and Motion control
tracks / user overrides. FK/IK, constrained attachment goals, joint limits
and solved pose are deterministic Mechanics responsibilities.

**Physics** acts ONLY after a qualified primary pose. It may own:
- `DECLARED_OSCILLATOR`: bounded secondary joint lag, NOT physical cloth truth;
- `DECLARED_XPBD_JOINT`: compliant secondary joint chains with declared
  inverse masses/colliders and canonical skin follow-through;
- `DECLARED_XPBD_SURFACE`: stretch/bend/area garment or membrane triangles
  on canonical indexed M* with explicit pins, dynamics policy and contacts.

Runtime order: Kinematics -> optionally joint Physics -> same canonical MIRA
LBS -> optionally surface Physics -> Graphics. Primary pose, canonical vertex
and surface identities never change. Surface pins are kinematic boundaries;
dynamic surface particles have one and only one physics owner.

`mechanics_physics_ownership_v1.py` rejects overlaps between **both**
secondary joint implementations and XPBD surface, using actual canonical
skin influences, not semantic proximity or a guessed physical threshold.
The previous runtime guarded XPBD joint/surface overlap but not the
oscillator/surface combination. It now preflights declared ownership before
expensive FK/IK and records the declared ownership modes in diagnostics.

## Wings, garments, straps and other auxiliary anatomy

A source-observed deformable wing can contain articulable rigid struts plus
a thin flexible membrane. AXIS chain semantics describe how the struts are
jointed; source-native skin describes which M* vertices follow those joints.
Motion retargeting supplies intended wing gestures. Physics MAY add secondary
strut dynamics or membrane dynamics where explicitly admitted, but not both
on the same dynamic mesh vertex. Declared attachments are not inferred from
appearance or bone names.

`mechanics_semantic_garment_v1.py` already turns confident
`outer_covering` face **candidate evidence** into an XPBD domain only after
explicit flexible-garment approval and actual skin/anchor support. This is
a product execution capability, not proof that IRIS has learned flexible
garment ontology or real material coefficients.

## Teacher authorization hierarchy

- Native rig parent hierarchy and skin weights can teach AXIS and MIRA
  without clips when exact source face/skin and joint-order receipts pass.
- Source animation/pose matrices teach deformation/motion consequences,
  not physical material identification.
- A 2D role label, SAM3 mask, FBX bone name or geometric flatness alone
  **cannot** supply actual mass, friction, cloth compliance or source-observed
  attachment consequences. Those are separately measured or editor-authored
  runtime policy, with their own provenance.
- Typed Drawable/Cut relations remain disabled in production. Surface
  membership and mechanical-chain structure are distinct from Drawable.

## Open gates

1. Teacher foundry: license-qualified real wing/garment source rig/skin
   and approved source-chain semantic evidence, with held-out morphology.
2. AXIS long-sequence generation throughput, source-conditioned auxiliary
   topology and valid control/constraint program proposals.
3. Physics coupling under dense learned MIRA weights: canonical influence
   sparsification, declared domain ownership, contact reliability and
   multi-solver frame-time benchmarks; never discard low skin weights
   silently inside Physics to hide double-ownership.
4. Dynamic visual quality: retarget clips + post-physics native Graphics
   visibility, hidden layer presentation, texture/alpha and save/reopen.
5. Real unseen two-source product evidence: mesh, chain, skin, motion,
   physics and 2D presentation on the SAME persistent .rss puppet.

The tests supplied with this change are deterministic safety regressions,
not source-corpus generalization or physical accuracy measurements.
