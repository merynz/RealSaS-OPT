# RSS.vNext — Mechanical semantics and 2D puppet production contract

**Decision: 2026-11-19.** Parent: [17 November StdGEN++ architecture freeze](RSS_VNEXT_V1_STDGENPP_PRODUCTION_ARCHITECTURE_FREEZE_20261117.md).
**Scope of this change:** active product typed-relations retirement and a source-qualified
joint-to-chain-to-skin training objective. **Not** an unseen trained release, a
new teacher dataset, a new solver or an automatic garment/wing classification claim.

## Frozen product ownership

1. Two original images (never six invented observations) -> one shared IRIS H(x).
   IRIS/SLR owns 3D layer and appearance evidence, not physical cloth/material truth.
2. One IRIS-extracted surface identity -> seeded TESSA refinement -> ONE final
   indexed M*. Open hems and multi-component shells need not be watertight.
   No independent product autoregressive triangle mesh, and no remesh after
   rig/skin/Graphics are sealed.
3. AXIS generates an articulated *program*: deform joints, hierarchy, auxiliary
   chains, controls, frames, typed constraints, anchors. A learned body-part tag
   alone cannot authorize a constraint or attachment.
4. MIRA owns exactly the final M* vertex-to-joint weights. When a source-native
   chain partition exists, the qualified hierarchical identity is:
   W(v,j) = P(chain(j) | v) * P(j | chain(j),v).
   Unknown joint ownership and unqualified vertices are masked. This PR's
   chain objective supervises the EXISTING MIRA skin: it does not invent an
   independent model head, material, chain, bone label or rig program.
5. Compiler admits mesh, rig, skin, mechanics and Graphics evidence separately
   with provenance and ABSTAIN on unsupported source or runtime behavior.
6. Mechanics has two execution responsibilities:
   - Kinematics: FK, IK, joint limits, constrained attachments, canonical LBS;
   - Physics: deterministic XPBD for explicitly selected secondary chains,
     garment sheets, surface contact/self-contact and parameterized material
     policies. Runtime executes primary pose first, optional Physics second.
   Physics is a named responsibility inside Mechanics, **not** a fifth neural
   expert or separate unconstrained geometry identity.
7. Motion provides licensed clip import/retargeting, blending and procedural
   controls. A clip/semantic motion label is not a claim that any solver
   can reproduce artist-authored animation from two static views.
8. Graphics solely renders the post-Physics M* with persistent UV, projected
   face visibility, alpha and optional draw-order exceptions. Learned Drawable
   and editable 2D cut partitions are explicitly **out of scope**. 3D -> 2D
   presentation does not require Sprite/Spine-style Drawable generation.

## Retired typed Relations

The inherited three anonymous-slot channels meant:
`same_drawable_affinity`, `cut_affinity`, and
`directed_attachment_affinity`. None is an executable mechanical operator.
Product default `typed_region_relation_policy=PRODUCT_NO_TYPED_RELATIONS`:
- does not compute the slot-pair projection;
- prevents its output from conditioning seeded TESSA and AXIS;
- leaves source-observable region membership and presentation-role logits;
- prevents query-pair typed relation APIs from inventing 0.5 affinities;
- permits socket/face candidates without an unqualified relation prior, but
  source-independent attachment qualification still needs its own evidence;
- keeps `RESEARCH_SOURCE_QUALIFIED` explicit for existing scientific tests.
The archive carries a zero sentinel for ABI compatibility; it is never a
valid probability and must be rejected by query consumers.

Source 3D co-region pairs remain independent geometric supervision when
their 3D addresses are qualified. Bone parents, skin, motion and authored
constraints belong in AXIS/MIRA/Mechanics teacher contracts, not a new
unqualified generic region algebra. SAM3/manual boundary corrections may
prepare source labels but cannot confer deformation or physical truth.

## Qualified chain objective added in this change

`models/rss_vnext/qualified_chain_skin_v1.py` accepts exact predicted and
source-native skin [1,V,J] on the same final M*, explicit source joint->chain
IDs [J] (-1 UNKNOWN) and qualified source vertex mask [1,V]. It supervises:
- summed mass by explicitly admitted chain;
- conditional bone weights within source-active chains.
- vertices with nontrivial source weight on UNKNOWN joints are excluded;
  arbitrary source chain ID numbers are never semantic classes.

`models/rss_vnext/product_axis_mira_seeded_training_v1.py` wires this as
opt-in `chain_skin`, only after exact source-native barycentric face/pose
transport qualification and frozen MIRA codec check. The independent chain
registry binds manifest, source asset, source qualification, AXIS token
identity, skin tensor, chain map and vertex mask digests. Missing/incompatible
teacher data must fail closed. Checkpoint/resume binds the registry hash.
Default weight is zero until real permitted teachers are materialized and
the benefit is measured on held-out real morphologies.

## Source-native auxiliary chains are structural, not authored controls

The AXIS V6 `CHAIN` event addresses an ancestral start/end path in the
emitted deform joint parent graph. The new decoder `axis_mechanical_chain_partition_v1.py`
verifies ancestry, unique owner vs shared-root UNKNOWN, and equivalence of
source-native teacher partitions independent of arbitrary chain labels. A
source-native chain teacher must match the *same* causal AXIS event program;
it cannot be an unrelated metadata label.

Native structural `CHAIN` events may now be taught from a sealed chain
receipt without claiming a source-authored IK or constraint intervention.
Actual `CONTROL`, `CONSTRAINT` and `ANCHOR` teaching continues to require
separate source intervention qualification. This is an intentional fix to
the previously over-broad intervention gate.

AXIS V6 product free-running had a 1,024-token cap. A 190-joint reference
with 190 frames and four FSQ tokens per joint needs 1,148 events before
auxiliary chains and controllers. The product proposal cap is raised to
2,048; the Transformer architecture itself supports up to 4,096.
**This only removes a hard capacity impossibility:** the current un-cached
causal prefix recomputation is still a critical long-sequence runtime
performance/VRAM qualification gate. No throughput has been claimed.

## Open product gates, explicitly NOT satisfied by this PR

- StdGEN++ sparse high-res layer extraction and independently qualified hidden
  source appearance / native moving-frame 2D rendering.
- SLR/TESSA adaptive production mesh quality and unseen two-image M* admission;
  seeded route exists but quality proof is missing.
- Source rights and coverage for **actual** wing, cloth, rope, tail, armor
  source-native rigs and chains; synthetic variants are not independent assets.
- Complete AXIS auxiliary-chain causal generation and general retarget mapping.
- Deterministic Physics runtime on source-real garment, wing membrane and
  multi-contact examples with measured residuals and stable frame pacing.
- Training all intended losses on an admitted real dataset and verifying
  source-free model inference. Passing tiny regression tests is not qualification.
- Final `.rss` edit/reopen, original 2D art fidelity and independent Proof.

## Acceptance checkpoints

A. Unit: product typed-region affinity cannot affect structural/control
   memory, typed query fanout, or socket candidate scoring.
B. Unit: true source weight partition reconstructs chain/conditional skin;
   missing ownership is UNKNOWN; nonzero source-qualified gradients reach MIRA.
C. Integration: source teacher hashes/mesh/rig IDs are identical across
   trainer resume; missing chain authorization rejects.
D. Scientific: licensed holdout winged humanoids + rigged skirts/capes;
   compare source-controlled rest/posed surfaces, collision, projected RGB,
   clip retarget and cloth/wing dynamics.
E. Release: one persistent M* -> AXIS/MIRA -> deterministic Mechanics/Physics
   -> Motion -> native Graphics -> editable .rss -> independent ProductProof.
