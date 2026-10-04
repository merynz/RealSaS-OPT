# RealSaS Truth Corpus V1 — Canonical Candidate Status

**Date:** 2026-10-04  
**Authority class:** corpus discovery / raw-truth status  
**State:** `SEALED_RAW__STRUCTURALLY_AUDITED__ADMISSION_PENDING`  
**Do not interpret as:** training-ready canonical corpus, unseen-generalization proof, or product qualification.

## Agent-facing rule

RealSaS now has a real sealed raw truth corpus for initial research. Future chats/agents MUST NOT state or assume that RealSaS has no corpus.

The correct shorthand is:

> RealSaS Truth Corpus V1 exists as a sealed raw, structurally audited canonical candidate. Canonical admission, representative render/import validation, lineage-safe split construction, and remaining FBX/Blend-only truth audit are still pending.

Do not widen this to "the corpus is finished" or "the corpus is training-ready" until the pending admission work below is closed.

## External raw corpus identity

The raw corpus is intentionally external to GitHub; large source assets are not committed to the repository.

Observed sealed bundle identity:

- bundle name: `RealSaS_Truth_Corpus_v1_20261004_2f15bf846944.zip`
- corpus root SHA-256: `2f15bf84694417cee3862c39a89e04bfa131f51331d6da5880739d40b7eab5ab`
- raw file count at seal audit: `3112`
- catalog records: `30`
- `SEALED_RAW`: `25`
- intentionally `CATALOG_ONLY_PAID_OR_GATED`: `5`

The manifest SHA, file-ledger SHA and corpus-root seal were independently rechecked against the exported audit package and matched.

## Confirmed GLTF truth subset

The first structural audit inspected the GLTF-bearing subset and confirmed:

- `395` GLTF files total
- `143` files contain skin + skeleton + animation truth
- `252` GLTF files are rigid/static at this layer
- after obvious Flat/Textured identity duplicates in the animated-mech family are collapsed: **139 confirmed unique rigged+animated GLTF subjects**
- `1803` animation clips across that confirmed animated subset
- `33` exact skeleton/hierarchy signatures

Skinning truth on the `143` skinned GLTF files is unusually clean:

- `641433` skinned vertices inspected
- zero-weight vertices: `0`
- maximum influences per vertex: `4`
- maximum observed weight-sum normalization error: approximately `1.19e-7`
- inverse bind matrices present on all confirmed skinned GLTF subjects

These numbers describe the audited GLTF subset, not a final deduplicated count for every FBX/Blend-only asset in the raw corpus.

## Confirmed animated-subject distribution in the GLTF subset

| Source pack | confirmed unique rigged+animated subjects |
|---|---:|
| Ultimate Monsters | 50 |
| Cute Animated Monsters | 21 |
| Ultimate Animated Animals | 12 |
| Ultimate Space Kit | 12 |
| Ultimate Modular Men | 11 |
| Ultimate Modular Women | 10 |
| Platformer Kit | 10 |
| RPG Characters | 6 |
| Animated Mech | 4 |
| Toon Shooter Kit | 3 |
| **Total** | **139** |

Additional FBX/Blend-only animated families exist in the sealed raw corpus but are not admitted into the `139` count until their binary truth is audited.

## Subject admission classes

Do not use semantic labels such as "character" versus "environment" as the primary admission rule. Use mechanical truth and intended court capability.

Current subject classes include:

- `FULL_TRUTH_ACTOR`
- `PARTIAL_TRUTH_ACTOR`
- `RIGID_PROP`
- `ARTICULATED_PROP`
- `ARTICULATED_ENVIRONMENT`
- `STATIC_AUX`

Examples already present in the corpus include small articulated non-character actors such as chest, lever, bouncer and spike-trap assets. These must not be discarded merely because they are environment/prop content.

Rigid assets such as weapons, rovers or spaceships may remain useful for presentation/turntable/vehicle courts even without skin truth.

## Leakage rule — critical

**Pack-level train/test splitting is NOT sufficient.**

The structural audit found exact skeleton/hierarchy signatures shared across nominally different Quaternius packs, including shared lineages across:

- Toon Shooter / Ultimate Monsters / Ultimate Space
- Cute Animated Monsters / Platformer Kit
- Ultimate Modular Men / Ultimate Modular Women

Therefore split authority must operate on leakage-connected lineage components, not merely pack names or rendered observations.

Required conceptual hierarchy:

`creator -> leakage-connected skeleton/topology/material lineage -> subject identity -> animation family -> rendered observations`

No FIT/LOFO/UNSEEN court may place members of the same leakage-connected lineage component on opposite sides merely because they came from different pack folders.

## Cross-source unseen candidate

KayKit free source bundles and Quaternius Universal Animation Library bundles are present in the sealed raw corpus as raw nested ZIPs with immutable ledger hashes. They should remain unopened/unconsumed by fitting where practical until the cross-source unseen policy is explicitly sealed.

A strong initial research strategy is:

1. Quaternius-based FIT8 / FITK development,
2. lineage-component-safe LOFO inside admitted Quaternius truth,
3. KayKit as a creator-level cross-source unseen court,
4. later add a harder third-creator unseen court.

Do not claim KayKit unseen validity until the exact split/seal is committed before model fitting.

## Initial representative admission smoke candidate

Before promoting this corpus to training-ready canonical status, run a representative importer/render/mechanics smoke that spans substantially different motion classes. A current candidate set is:

- RPG Warrior — humanoid
- Horse — quadruped
- GreenBlob — amorphous
- Dragon/Evolved Dragon — flying creature
- Mech — mechanical articulated actor
- Chest — articulated prop
- SpikeTrap — articulated environment
- Space Mech — nonorganic/mechanical actor

The exact subject IDs should be frozen from the raw manifest when the court is implemented; the names above are descriptive planning labels, not yet an executable court manifest.

## Admission work still required

The corpus is **not yet fully admitted**. Close these before calling it training-ready canonical truth:

1. representative source -> importer -> canonical mesh/rig/skin -> animation -> render smoke;
2. verify materials/source appearance survive import sufficiently for truth/render courts;
3. audit FBX/Blend-only animated families and fold them into deduplicated lineage identities;
4. construct topology/skeleton/material leakage-connected components;
5. freeze FIT8/FITK/LOFO manifests;
6. cryptographically seal UNSEEN before fitting;
7. preserve raw sources immutable; all normalization/retarget/repair/render-derived artifacts live outside the raw corpus.

## Current claim boundary

Allowed:

> RealSaS has a sealed raw truth corpus with a structurally audited GLTF subset containing 139 confirmed unique rigged+animated subjects and clean skinning truth, plus rigid/static and not-yet-admitted FBX/Blend families.

Forbidden until further closure:

- "139 is the final corpus size"
- "the full corpus is deduplicated"
- "the corpus is training-ready"
- "LOFO is safe at pack granularity"
- "KayKit unseen has already been proven"
- "render/import fidelity has already passed"

## Continuation

If a future chat/agent is asked to build or evaluate a generalized RealSaS corpus, start from this document rather than rediscovering whether a corpus exists. Preserve the external raw seal and perform all derived admission/audit work non-destructively.
