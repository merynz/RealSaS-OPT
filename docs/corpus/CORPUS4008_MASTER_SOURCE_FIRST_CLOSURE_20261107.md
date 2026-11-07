# Corpus4008 → RealSaS Master Corpus expansion: source-first closure V1

Date: 2026-11-07. Owner: Corpus/Truth Foundry. Status: **SOURCE AUDIT IN PROGRESS — NO TRAINING/ADMISSION**.

## Frozen order

1. **Build the Master Corpus** from original files and exact identities: legal/source lineage, dedup, actual character-vs-prop, modular components, attachment graph, rest/orientation and real source-correct render.
2. Only then build qualified Geometry / Semantic / Graphics and TESSA/AXIS/MIRA truth families from admitted source observations.
3. Only after independent courts: family-specific admission, split sealing, materialization and training.

Do NOT start from the assumption that the Master Corpus 2026-08 technical `ADMIT` means the current product's semantic, graphics, attachment, controllability, camera, or animation truth is admitted.

## What the previous 72-gate inventory establishes

Immutable historical master: **4,008** entries (3,765 Objaverse originals, 191 Quaternius, 37 KayKit, 15 Khronos QA). Historical decisions: 3,414 rig+skin, 484 rig/no-skin, 110 geometry-only. **169 historical EXTERNAL_HOLDOUT** rows are protected. Legacy gold/QA source IDs have only 24-char prefix tokens and must be bound to complete raw SHA-256 before duplicate promotion. **53** are likely Core370 source overlaps, not final certified duplicate lineages. Existing 72×4008 metadata gate audit is a **pre-audit**, not source/render/semantic inspection.

The Master Corpus V3 source tree is on the user-controlled Drive. `master/variants`, `metadata`, `sources`, original asset `renders`, and the legacy `MASTER_VARIANTS.jsonl` are distinct. Source originals are the authority, not the previous normalized NPZ or legacy cel RGB.

## New executable producers in this branch

- `tools/corpus/probe_corpus4008_source_scene_v1.py`: exact source SHA and source-token check, original GLB hierarchy/parent joint → unskinned mesh links, skin joint counts, native action inventory, image/material census. OBJ source groups/triangle boundary counts. FBX and BLEND **stop at exact raw-byte proof** and require pinned Blender. Source group != semantic component != drawable; native bone attachment link != runtime consequence.
- `tools/corpus/build_corpus4008_source_preflight_v1.py`: full 4,008-row readback against original mounted `sources/<provider>/...`; detects missing/ambiguous originals, preserves 169 historical holdouts and 15 QA-only records, emits versioned source receipts when original bytes are available. No network or secret access, no generated labels.
- Hosted-only synthetic regression tests and workflow. No self-hosted runner dependency for CI.

To use after a read-only Drive source tree is mounted and the original Master V3 ledgers are copied/verified:
```bash
python -m tools.corpus.build_corpus4008_source_preflight_v1 \
  --master /verified-drive/metadata/MASTER_VARIANTS.jsonl \
  --selection /verified-drive/metadata/CANONICAL_VARIANT_SELECTION.json \
  --sources-root /verified-drive/sources \
  --output /audit/corpus4008_source_preflight_v1
```
`--limit 100` performs a bounded source probe while preserving exact full-4008 census. `--limit 0` is all available sources. This script does not reconstruct exact legal rights from file presence; the legal and derivative-rights receipts remain separate.

## Three real original-source probes already performed locally

| Original source | Exact source bytes | Native imported evidence | Still unqualified |
|---|---|---|---|
| KayKit CC0 GLB `asset_551ea351b43a1787d0f55536` | full SHA verified | 12 mesh groups (12 individually open under import triangulation), 1 skin with 41 joints, 76 named actions, 6 bone-parented unskinned mesh candidates, 3,196 sampled weight rows with no sum mismatch | Character class, native attachment response, real rest pose, character up/front, camera, source RGB, semantic, training |
| Quaternius CC0 OBJ `asset_102c52015bc540c2fe3f7ef5` | full SHA verified | 8 import mesh groups (7 individually open), native OBJ has no rig/skin or animation | Any desired merged solid SDF, component semantic class, camera, RGB, training |
| Objaverse original FBX `asset_09178fa704865496d57c11be` | full SHA verified | Pinned FBX evaluator **required**; no source topology/joint/skin assertion | All scene/pose/render/semantic targets |

Important: these are *source-byte + import observations*, **not independent native Blender parity courts**. Open per-group surfaces are NOT automatic rejection: armor, clothes, props and modular assets can legitimately be open.

## Critical presentation audit

Two historical native-size 1024×1024 `V0/cel_clean.png` images were inspected. Both are cel-style derivatives, **not independently source-appearance matched RGB**. One KayKit source render includes multiple articulated/attached accessories occupying a large portion of its silhouette. It is unsafe to teach source-appearance `Graphics` from this RGB without original-material parity.

The old render contract `realsas.level_orthographic_z_orbit.v1` uses +Z world-up and 8 cyclic yaw angles. That cannot by itself certify anatomical up/front, camera correctness or 2D readability across arbitrary source file orientation. The new production renderer must document:

1. Exact source-space and native rest/pose transforms, preserved untouched.
2. Character-space up/front (or explicit ABSTAIN) and separate body/attachment bounds.
3. Invertible source→character→camera matrices and camera round-trip.
4. Eight native camera observation proofs: crop/occupancy, alpha, ray/primitive IDs, occlusion, hidden geometry, loose props and camera labels; at least one native action/repose if rigged.
5. Original material/alpha RGBA as `NATIVE_SOURCE_REFERENCE`; optionally a shader-only `STYLIZED_AUGMENTATION` that never overwrites source appearance authority.
6. Independent semantic class/region review; source object/material/bone names are teacher hints only.

**Do not force an upright camera onto a sideways native asset or hallucinate a front direction based on world +Z.** Do not infer attachment identity from proximity alone, or structural components from watertight island counts.

## Next source-preparation work (P0)

- Run entire 4,008 source-byte census on a Drive-mounted read-only worker. Review missing, ambiguous, source-name collisions and exact SHA drift before advancing.
- Add pinned Blender 4.0.2 FBX/glTF scene evaluator using existing Corpus370 native source adapter: world/local matrices, armature, bind, actions, skin, object parenting, sockets, constraints, open/closed islands, nonuniform scale and source textures. Require reproducible independent readback.
- Morphology- and provider-stratified contact sheets with original/candidate anatomical frame **side-by-side**, plus multiple animation poses for rigged subjects; independently review character vs non-character and functional prior candidates.
- Bind proposed component/source-face semantic teachers only to reviewed source regions; never promote descriptive source names to model labels.
- Extend existing 72-gate ledger using typed external source receipts, rather than replacing it with optimistic `PASS`.
- Keep 370 source families and active semantic/IRIS training prep untouched until this Master Corpus passes its own source gates.

There is no real 4,008-character-count qualification yet. The 3,000+ training target remains an ambition subject to actual character class, source rights, dedup, correct cameras and per-head truth availability.

## 2026-11-07 real remote-source census and recovery progress

The exact previously published Master V3 manifest and selection were verified by full SHA-256:
- `MASTER_VARIANTS.jsonl`: `475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`
- `CANONICAL_VARIANT_SELECTION.json`: `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

The [real GitHub-hosted Drive inventory](https://github.com/merynz/RealSaS-OPT/actions/runs/42371135711) observed **659/4008** source-path matches in the old Drive `sources/` root:
- Objaverse 416/3765
- Quaternius 191/191
- KayKit 37/37
- Khronos QA-only 15/15

The other **3349** are *not in that old Drive source folder under the expected identity*, rather than demonstrated globally destroyed. The [SHA-pinned metadata recovery audit](https://github.com/merynz/RealSaS-OPT/actions/runs/42371850821) joined **all 3349** one-to-one with old original-source records and upstream Objaverse linked annotations. All 3349 have 40-hex-commit-pinned GitHub blob URLs and historical legal `PASS` records; **135 remain EXTERNAL_HOLDOUT** and must never flow into train. Historical legal PASS is **not** fresh license or training authority.

The [3×(FBX, GLB, glTF) real original-byte pilot](https://github.com/merynz/RealSaS-OPT/actions/runs/42372470068) reacquired **9/9** and independently matched the complete expected SHA-256 for **85,587,565** bytes. Exact source bytes were placed in **PRIVATE Drive quarantine only**, not public Actions artifacts and not the official `sources/` root. This demonstrates a viable retrieval route, not guaranteed accessibility for the other 3340 files.

A bounded follow-up [8-shard source acquisition pilot](https://github.com/merynz/RealSaS-OPT/actions/runs/42372844889) uses frozen 32-way source ownership, at most 16 non-holdout historical-legal-PASS originals per running shard (at most 128 this first wave), 64 MiB per original, max 4 concurrent hosted workers, all exact SHA verified. Every failure/too-large/non-source response remains a typed failure, never a PASS or silent skip. The first wave must be inspected before increasing the sample window.

Exact private source metadata, recovery URLs and staged bytes live solely under the Master V3 `audits/source_inventory_v1`, `audits/source_recovery_v1`, `audits/original_reacquisition_pilot_v1`, and `audits/original_recovery_shards_v1` folders. No original-content data is exported to public workflow artifacts.

**Still open**: active-source current legal review, asset/variant/pose/frame import courts, orientation and camera, 2D presentation, attachments and secondary motion consequences, per-head supervision and admission. A verified source byte is **not** a qualified character or render truth.

## 2026-11-07 original-source recovery and actual Blender imported-scene validation

The [8×16 SHA-verified initial restoration](https://github.com/merynz/RealSaS-OPT/actions/runs/42372844889) completed all eight hosted jobs: **128 attempted**, **127 original source-byte SHA matches** (**304,666,927** accepted bytes), **1 HTTP 404** retained as an explicit failure. The preceding 9-source / 85,587,565-byte pilot overlaps one of those recovered 127 sources; the HTTP 404 is not that overlap, so the two waves provide **135 distinct exact-SHA recovered originals** in private quarantine, not 136 distinct assets.

The [pinned Blender 4.5.14 real-source import](https://github.com/merynz/RealSaS-OPT/actions/runs/42373335212) completed for all three original bytes, preserving rest import separately from animated import:
- Objaverse original FBX: 1 imported mesh object, 1 armature, **34 bones**, **1 imported action**, **30 indexed geometry components** inside that mesh, 592 indexed open edges and **4 missing external images**. Graphics source-RGB must stay blocked.
- KayKit original GLB: 13 imported mesh objects, 1 armature, **41 bones**, **76 imported actions**, **6 bone-parented mesh candidates**, 12 mesh objects with indexed open boundaries. The six attachment relations are not runtime consequence truths.
- Quaternius original OBJ: 1 imported mesh object, 0 armatures/actions, **20 indexed geometry components**, 96 boundary edges, no native OBJ skin or rig.
- Actual outputs are **importer observations**, not source-native FBX parity and not uncontested semantic classes, CAA, 2D appearance or correct anatomical camera frames. They are held on restricted Drive under `audits/blender_source_pilot_v1`.

A [32-shard bounded source-recovery job](https://github.com/merynz/RealSaS-OPT/actions/runs/42374046073) is initiated for **3,214 historical non-holdout** missing Objaverse original candidates. Each shard has between 85 and 123 source identities, runs only four concurrent hosted workers, and limits each source to 64 MiB. First-wave exact-byte cache is revalidated by full SHA before reuse; no source or training admission, no canonical `sources/` mutation. The 135 original historical external-holdout identities remain protected.

**Next strict Master Corpus gates:** finish and reconcile the 32-shard source receipts and any 404/oversize entries; separately revalidate current commercial source rights; qualify each original Blender/Git LFS/texture dependency, character-vs-prop, anatomical pose/up/front and 8-view native RGB/primitive alignment before building the Semantic / Graphics / TESSA truth targets.
