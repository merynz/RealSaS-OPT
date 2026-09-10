# RealSaS — Arachne Information-Preservation Audit V1 — Phase 7B Mage Component→GSA Diagnostic Census

**Date:** 2026-09-11  
**Status:** `DIAGNOSTIC_ONLY__TEACHER_SOURCE_METADATA_NOT_SHIPPING_INPUT__NO_RUNNING_A0_CHANGE`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`

## 0. Question

After proving that the current full-source A0 teacher authority contains Mage rigid attachments, quantify whether the current **950 shipping GSA carriers** geometrically reach recognizable source components. This is a forensic/coverage diagnostic only.

Raw source object identity is **teacher/source metadata** and is not legal A1 product input. The purpose is to inspect what the shipping surface contains, not to give object labels to Arachne.

## 1. Exact object ranges recovered

The persisted Mage `source_structure.json` binds the normalized source vertex/face concatenation in this exact order:

- Mage_ArmLeft: 330 vertices / 468 faces;
- Mage_ArmRight: 330 / 468;
- Mage_Body: 1275 / 1230;
- Mage_Head: 761 / 1027;
- Mage_LegLeft: 326 / 418;
- Mage_LegRight: 326 / 418;
- Spellbook: 399 / 292;
- Spellbook_open: 418 / 292;
- 1H_Wand: 158 / 150;
- 2H_Staff: 498 / 440;
- Mage_Hat: 402 / 396;
- Mage_Cape: 56 / 84;
- Icosphere: 42 / 80.

Total: 5321 vertices / 5763 faces.

## 2. Diagnostic assignment method

For each current GSA carrier rest-world point in `ARACHNE_MAGE_FS1_CONDITIONING_CACHE_V2.npz`, exact point→triangle Euclidean distance was computed to every source object face subset using the persisted normalized source geometry and exact source object face ranges.

Each GSA carrier was assigned **only for this diagnostic** to the source object with minimum geometric surface distance. No teacher weight, source object name, or object range enters any shipping model.

The diagnostic also records the distance margin against the second-nearest source object to expose ambiguous/overlapping component regions.

This nearest-component diagnostic is not the FS1 teacher-target algorithm. FS1 may use local ambiguity gates + support-view ray voting, so a carrier's final teacher W may legitimately differ from the nearest object's source skin in an ambiguous region.

## 3. Nearest-component census of GSA950

| Nearest source component | GSA rows | Supervised rows | Median source distance | Notes |
|---|---:|---:|---:|---|
| Mage_Head | 431 | 423 | 0.00528 | dominant upper/head surface carrier mass |
| Mage_Body | 297 | 295 | 0.00693 | torso/core |
| Mage_ArmLeft | 61 | 56 | 0.00832 | |
| Mage_ArmRight | 58 | 58 | 0.00573 | |
| Mage_LegRight | 46 | 45 | 0.00805 | |
| Mage_LegLeft | 37 | 37 | 0.00512 | |
| **Mage_Hat** | **11** | **11** | **0.00287** | clear explicit hat carriers |
| **2H_Staff** | **3** | **3** | **0.00170** | explicit staff carriers |
| **Spellbook_open** | **3** | **3** | **0.00471** | explicit book carriers |
| **Spellbook** | **2** | **2** | **0.00480** | explicit book carriers |
| Icosphere | 1 | 1 | 0.00544 | nearest-source label only; target is body/spine blend; see §5 |

No carrier's strict nearest source component is `1H_Wand` or `Mage_Cape` in this diagnostic.

This does **not** prove those objects are absent from the rendered/implicit surface. Both can be spatially overlapped/occluded by other source components, and the GSA carrier budget is compact. It only says no compact carrier lands closer to their triangle subset than to every other component.

## 4. Rigid attachment rows agree strongly with current FS1 teacher semantics

### Spellbook

2 nearest-source carriers. Both supervised. Both exact:

`W(handslot.l) = 1.0`.

### Spellbook_open

3 nearest-source carriers. All supervised. All exact:

`W(handslot.l) = 1.0`.

One row has a relatively large absolute source distance (~0.0674) and a small nearest-vs-second margin (~0.00080), so its object assignment is geometrically weak even though its FS1 target is the attachment control.

### 2H_Staff

3 nearest-source carriers:

- 2 rows are exact `W(handslot.r)=1.0`;
- 1 row is an FS1 ambiguity/vote case with confidence code 1 and final target approximately `hand.r=.95737`, `wrist.r=.04022`, `lowerarm.r=.00242` rather than the rigid handslot.

For this exceptional row the nearest 2H_Staff distance is ~0.00180 and the second-nearest 1H_Wand distance is ~0.00217; margin only ~0.000366. This is exactly the kind of spatially ambiguous case for which nearest-component identity and FS1's multi-view teacher arbitration should not be conflated.

### Mage_Hat

11 nearest-source carriers. All 11 are supervised and all 11 have exact:

`W(head)=1.0`.

All 11 also have a positive >0.001 nearest-vs-second component distance margin; 7/11 have margin >0.005. Their second-nearest component is consistently `Mage_Head`.

This is stronger than the earlier z-slice inference: the compact GSA substrate contains at least **11 geometrically explicit Hat-nearest carriers**, all with the correct rigid head teacher semantics.

## 5. The one Icosphere-nearest carrier is not evidence that Icosphere entered teacher skin authority

One GSA carrier is geometrically ~0.00544 from the excluded Icosphere and ~0.00644 from Mage_Body. Its final current FS1 teacher row is a valid body blend:

- spine ~0.69947;
- chest ~0.18872;
- hips ~0.11181.

This is consistent with the full-source authority rule that Icosphere's 42 vertices / 80 faces have zero selected skin authority and are excluded from teacher target faces. The nearest-object diagnostic is purely geometric and is permitted to label an overlap point “Icosphere-nearest”; FS1 target authority still correctly ignores Icosphere.

## 6. Component-distance robustness

For the body/limb components, the nearest component is generally strongly separated from the second nearest. Attachment components are naturally closer/overlapping with one another or their host body part.

Diagnostic margin examples:

- Mage_Hat: median nearest-vs-second margin ~0.00625; minimum ~0.00105;
- Spellbook: median ~0.00332;
- Spellbook_open: median ~0.00146;
- 2H_Staff: median ~0.00103.

This supports the policy that source-component labels are useful for **post-hoc regional error analysis**, but should never be elevated to a product predictor feature.

## 7. Revised answer to the visual concern

The screenshot concern can now be stated more precisely:

- the head/hat region is **not absent** from GSA; there are at least 11 compact carriers whose nearest exact source component is Mage_Hat, all target-owned by head;
- staff/book attachment geometry is also represented by explicit compact carriers;
- the current GSA compact surface is therefore not simply “body-only”;
- lack of a drawn bone through the full hat volume is not evidence of uncontrolled hat geometry;
- explicit product assembly typing remains separately open, as Phase 7 established.

The exact GSA representation of `Mage_Cape` and `1H_Wand` is not established by this nearest-component assignment because neither wins the nearest-object classification for a compact carrier. That is a compaction/visibility/component-resolution question, not proof of silent loss.

## 8. Implication for post-A0 regional Arachne diagnostics

Once the current token arms finish, a zero-optimizer regional error report should include these **diagnostic-only source masks/nearest-component groups**:

- Mage_Hat-nearest rows;
- Spellbook/Spellbook_open-nearest rows;
- 2H_Staff-nearest rows;
- core head vs body vs limbs;
- ambiguity-margin strata;
- pure one-hot vs blend rows.

This can identify whether the A0 representation errors concentrate on disconnected rigid components, without contaminating training or product inference with source object labels.

**No current A0 treatment, A1 interface, teacher authority, or product claim changes.**
