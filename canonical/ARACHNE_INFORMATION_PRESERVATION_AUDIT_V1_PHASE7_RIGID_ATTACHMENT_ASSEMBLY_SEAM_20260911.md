# RealSaS — Arachne Information-Preservation Audit V1 — Phase 7 Rigid-Attachment / Assembly Seam

**Date:** 2026-09-11  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_RUNNING_A0_CHANGE__NO_A1_IMPLEMENTATION_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Continuation authority:** `main/CURRENT_STATE.md`

## 0. Why this phase was opened

The Mage front render visibly contains large non-body pieces while the compact GSA/skeleton visual can make some of them look absent or mechanically unexplained. A prior provisional audit note therefore left two possibilities open:

1. rigid attachments were intentionally kept out of deformation truth; or
2. they were silently omitted from the A0 teacher authority.

That ambiguity is now resolved for the current Mage A0 authority.

**Main finding:** the current full-source Mage A0 teacher authority **does contain the rigid attachments**, and the corpus extractor converts bone-parented rigid pieces to exact one-hot skin weights on their parent attachment control. The body-only teacher omission was discovered and superseded **before scientific A0 optimization** on 2026-09-08.

The remaining open issue is not missing A0 teacher truth. It is **product semantic typing/editability**: current `QualifiedSkeletonIRV2.assembly_root_binding` is empty, so rigid pieces can be represented kinematically by one-hot skin weights but are not yet demonstrated as explicit product assembly/attachment state.

---

## 1. Exact R6 Mage render classification

Recovered R6 pilot asset:

- pilot: `kaykit_01_Mage`;
- raw source SHA-256: `cf898585da33fab50c724d31605fb931eb2912e6d2280092141e98ca81ad507d`;
- raw source: KayKit Adventures `Characters/gltf/Mage.glb`;
- primary armature: `Rig`.

R6 subject gate admitted six skinned-primary pieces:

| Object | Classification | Weighted vertices |
|---|---|---:|
| `Mage_ArmLeft` | `SKINNED_PRIMARY_ARMATURE` | 330 |
| `Mage_ArmRight` | `SKINNED_PRIMARY_ARMATURE` | 330 |
| `Mage_Body` | `SKINNED_PRIMARY_ARMATURE` | 1275 |
| `Mage_Head` | `SKINNED_PRIMARY_ARMATURE` | 761 |
| `Mage_LegLeft` | `SKINNED_PRIMARY_ARMATURE` | 326 |
| `Mage_LegRight` | `SKINNED_PRIMARY_ARMATURE` | 326 |

Total skinned-primary vertices: **3348**.

It also admitted six rigid bone-parented pieces:

| Object | R6 class | Parent bone from raw GLB | Vertices | Triangles |
|---|---|---|---:|---:|
| `Spellbook` | `RIGID_BONE_ATTACHMENT` | `handslot.l` | 399 | 292 |
| `Spellbook_open` | `RIGID_BONE_ATTACHMENT` | `handslot.l` | 418 | 292 |
| `1H_Wand` | `RIGID_BONE_ATTACHMENT` | `handslot.r` | 158 | 150 |
| `2H_Staff` | `RIGID_BONE_ATTACHMENT` | `handslot.r` | 498 | 440 |
| `Mage_Hat` | `RIGID_BONE_ATTACHMENT` | `head` | 402 | 396 |
| `Mage_Cape` | `RIGID_BONE_ATTACHMENT` | `chest` | 56 | 84 |

Rigid-attachment total: **1931 vertices / 1654 triangles**.

The raw GLB's twelve character/attachment meshes therefore total:

- **5279 vertices**;
- **5683 triangles**.

R6 excluded `Icosphere` as `NO_PRIMARY_ARMATURE_RELATION`.

Important visual terminology correction: the left-hand rectangular object in the render must not be called a “shield” as an exact asset fact. The recovered Mage object names are `Spellbook` / `Spellbook_open`; no object named `Shield` exists in this exact Mage record.

---

## 2. The master-corpus extractor intentionally converts rigid attachments to one-hot skin truth

The audited Master Corpus Blender extractor iterates the source mesh objects, reads exact vertex-group weights first, and then has an explicit rigid-attachment rule:

```text
if mesh is parented to the primary armature as BONE
and parent_bone resolves in the source skeleton:
    for zero-weight vertices:
        W[parent_bone] = 1.0
```

It then normalizes nonzero rows and persists the full geometry + skin matrix.

This is not an accidental fallback. The source comment explicitly states that rigid bone-parented accessories are a legitimate skinning representation and may be converted to one-hot weights.

Thus two representations coexist cleanly:

- source semantic class: `RIGID_BONE_ATTACHMENT`;
- deformation truth used by A0: exact one-hot weight to the attachment's parent control.

For a purely rigid piece these are kinematically compatible.

---

## 3. Current normalized Mage authority proves that all six rigid attachments survived

Current normalized source authority:

- SHA-256: `528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f`;
- `vertices`: **5321 × 3**;
- `faces`: **5763 × 3**;
- `skin`: **5321 × 41**.

Current C4/full-source loader then bridges the 41 source controls to the exact selected 22 controls. It reports:

- selected-skin-positive vertices: `5321 - 42 = 5279`;
- selected-skin eligible faces: **5683**;
- zero-selected-skin vertices: **42**;
- excluded faces: **80**;
- excluded mass on the 19 non-bridge controls: **0**.

These numbers align exactly with the R6/raw-source decomposition:

```text
5279 character + rigid-attachment vertices
+ 42 unrelated Icosphere vertices
= 5321 normalized vertices

5683 character + rigid-attachment triangles
+ 80 unrelated Icosphere triangles
= 5763 normalized faces
```

The earlier P0 full-source supersession record independently identifies those exact 42/80 rows/faces as the final `Icosphere` object and defines the A0 teacher surface over the 5683 skin-supported faces.

Therefore the A0 teacher source does **not** discard the rigid attachments.

---

## 4. Exact rigid one-hot forensic

The raw GLB was parsed and each source mesh's world-space vertices were mapped back to the exact normalized-source geometry (maximum positional match error below `1.7e-6` after the glTF Y-up → Blender Z-up basis conversion). The normalized 41-column skin rows give:

| Object | Exact source weight result |
|---|---|
| `Spellbook` 399/399 | `handslot.l = 1.0` |
| `Spellbook_open` 418/418 | `handslot.l = 1.0` |
| `1H_Wand` 158/158 | `handslot.r = 1.0` |
| `2H_Staff` 498/498 | `handslot.r = 1.0` |
| `Mage_Hat` 402/402 | `head = 1.0` |
| `Mage_Cape` 56/56 | `chest = 1.0` |

Aggregate exact-one-hot counts provide an independent checksum:

- `handslot.l`: **817 = 399 + 418**, and no non-one-hot positive rows;
- `handslot.r`: **656 = 158 + 498**, and no non-one-hot positive rows;
- `head`: **1163 = 761 Mage_Head + 402 Mage_Hat**, all exactly one-hot;
- `chest`: exactly **56** one-hot rows attributable to `Mage_Cape`; body chest rows additionally exist as blended skin.

The exact Geppetto/Arachne source bridge selects source controls including:

- `handslot.l` = source control 8;
- `handslot.r` = source control 13;
- `head` = source control 14;
- `chest` = source control 3.

Since excluded non-bridge control mass is exactly zero, 41→22 bridging discards none of this attachment deformation truth.

---

## 5. Historical body-only A0 teacher bug was already found and superseded before optimization

This phase recovered the prior canonical evidence record `ARACHNE_MAGE_P0_FULL_SOURCE_SUPERSESSION_20260908.md`.

It states that the first P0 V1 teacher projection used a **3348-vertex body-only** teacher surface and therefore omitted accessory/attachment geometry. Crucially:

- this was found **before main scientific A0 optimizer steps**;
- the body-only target/cache was retired as optimizer authority;
- current full-source FS1 uses **5321 vertices / 5763 faces / 5321×41 skin**;
- it excludes only the 42 zero-selected-skin Icosphere vertices / 80 faces;
- it bridges 41→22 with zero excluded deformation mass;
- teacher/source geometry remains training/evaluator-only and is forbidden as shipping A1 input.

The first full-source diagnostic changed `9 / 950` GSA rows relative to the retired body-only target, demonstrating that attachment omission was not merely theoretical.

Thus the current live A0 token experiment is downstream of the **corrected full-source authority**, not the retired body-only authority.

---

## 6. Current GSA/A0 cache contains explicit attachment-control rows

Using the current sealed FS1 conditioning cache and exact 41→22 bridge:

- `handslot.l` has **7 supervised GSA rows** with exact weight `1.0`;
- `handslot.r` has **3 supervised GSA rows** with exact weight `1.0`.

In the source authority these two controls occur **only** on the rigid book/wand/staff meshes, with no non-one-hot positive source rows. Therefore these ten current GSA teacher rows are direct evidence that at least the hand-slot rigid-attachment classes are represented in the shipping-surface→teacher target binding.

The previous full-source-vs-body-only diagnostic changed 9/950 rows; the difference between 10 current exact hand-slot rows and 9 changed rows is not interpreted here without replaying the retired target ambiguity/fallback classification.

For `head`, the current cache has **434 supervised rows** with exact head weight `1.0`. Because both `Mage_Head` and `Mage_Hat` are one-hot head in the source, weight alone cannot partition these rows into body-head vs hat. The earlier `z>=1.3` diagnostic remains valid as a statement about **teacher head ownership**, but must not be presented as an exact hat-only segmentation proof.

For `Mage_Cape`, chest weight is not a unique object identifier because body vertices also carry chest influence. Exact cape-specific GSA carrier coverage remains a regional/component-census question, not solved merely from W.

---

## 7. Correction to the earlier provisional “hat may be absent from teacher source” concern

A provisional audit interpretation considered the possibility that R6 rendered rigid attachments but the normalized A0 teacher source omitted them. That possibility is now **falsified** for the current Mage authority.

Correct statement:

> The current Mage full-source A0 teacher authority explicitly contains rigid bone-parented accessories, encoded as exact one-hot deformation weights on their parent controls. The historical body-only omission was already repaired before A0 scientific optimization.

The earlier upper-head result should therefore be read as:

> upper surface rows in the current teacher field are rigidly head-owned;

not:

> every such row has been object-identified as `Mage_Hat`.

This is an evidence-precision correction, not a change to the running experiment.

---

## 8. What remains genuinely open: product assembly semantics

Current `QualifiedSkeletonIRV2` has an `assembly_root_binding` field, but current `qualify_skeleton_v2()` constructs it as `{}` and the exact promoted Mage qualified artifact also has an empty binding.

Therefore RealSaS currently has two different capability levels:

### Kinematic deformation capability — substantially present

A rigid accessory can be represented as one-hot skin:

```text
staff surface -> handslot.r weight 1.0
hat surface   -> head       weight 1.0
cape surface  -> chest      weight 1.0
```

Arachne + Compiler skin qualification can in principle carry this as ordinary valid weight rows. This is sufficient for the piece to move rigidly with its control.

### Explicit assembly/editing semantics — not yet demonstrated

The current qualified skeleton does not first-class say:

- “this disconnected component is a rigid attachment”;
- “its socket/parent is handslot.r/head/chest”;
- “preserve it as detachable/non-deforming assembly geometry”;
- “this root is assembly-only rather than deform-root.”

That matters for an editable product even if motion can be reproduced by one-hot weights.

**Verdict:** `KINEMATIC_ATTACHMENT_TRUTH_PRESENT__EXPLICIT_PRODUCT_ASSEMBLY_TYPING_PARTIAL/OPEN`.

This is not an A0 codec gate failure and does not revoke the 4-token GSA stable PASS.

---

## 9. Implications for A1

The A1 input/output contract must avoid two opposite errors.

### Do not drop the attachment evidence

Rich S contains disconnected/spatial surface structure and qualified G contains handslot/head/chest controls. A future predictor must be capable of assigning rigid regions sharply to these controls. One-hot rigid rows are legitimate Arachne outputs.

### Do not ask Arachne to invent assembly authority

If the product later needs typed detachable/rigid attachment semantics, that must be represented in an upstream/Compiler-owned qualified assembly contract. Arachne may predict skin/deformation semantics; it should not infer a second hidden parent/socket graph that competes with Compiler authority.

Potential later architecture consequence:

- ordinary A1 baseline can still predict W for all surface carriers;
- a future `QualifiedAssemblyIR` / non-authoritative component annotation may allow explicit rigid routing/editing without changing the mathematical skin field.

No such schema is implemented by this audit.

---

## 10. FIT-k / FIT8 coverage requirement

The recovered Mage is already a genuine rigid-attachment witness, so it is incorrect to say FIT1 is attachment-free.

However the **product feature** “explicit rigid assembly semantics” has not been accepted merely because Mage source truth contains rigid attachments.

Before a broad unseen/product campaign, corpus coverage metadata should distinguish at least:

- no attachment;
- rigid bone attachment;
- disconnected deforming secondary component;
- assembly-only/multi-root requirement;
- garment/secondary-motion requirement.

This is a **coverage label**, not a post-hoc selection criterion for the already frozen experiment.

---

## 11. Phase-7 verdict

1. **FALSIFIED:** current A0 teacher source omits Mage rigid attachments.
2. **PROVEN:** the historical body-only omission existed and was superseded before A0 optimization.
3. **PROVEN:** full-source normalized authority contains all six rigid attachments and maps them to exact one-hot parent-bone skin truth.
4. **PROVEN:** 41→22 bridge preserves all current Mage deformation mass.
5. **PROVEN:** current FS1 GSA cache contains exact handslot attachment-target rows.
6. **NOT PROVEN:** exact GSA carrier count for each visual object such as `Mage_Hat` or `Mage_Cape`; weight identity alone cannot always separate object identity.
7. **OPEN PRODUCT SEAM:** explicit qualified rigid-attachment/assembly semantics remain partial because `assembly_root_binding` is empty.
8. **NO CURRENT A0 CHANGE:** running 4/8/16/32 treatment remains untouched.

## 12. Next work

Phase 8 remains post-A0-arm work: latent interface + regional error + holdout deformation-signal forensic.

Before Phase 8 becomes available, remaining zero-optimizer work can additionally census compact GSA coverage of known raw-source components using legal diagnostic object masks. That census must remain **diagnostic only**; raw source object labels are teacher/source metadata and are not A1 shipping inputs.

**No A1 implementation, assembly schema implementation, current A0 treatment change or product PASS is authorized by Phase 7.**
