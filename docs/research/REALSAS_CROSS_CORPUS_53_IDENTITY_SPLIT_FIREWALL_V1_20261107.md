# Global Truth Foundry: Cross-corpus source identity and split firewall (2026-11-07)

**Result: 53 SHA-96-bit source-identity overlap candidates; 3 exactly byte-verified, including EXTERNAL_HOLDOUT and CAL examples. NO training admission, NO source deletion, NO finalized dedup.**

## Critical correction to previous zero-overlap observation

The earlier `0` count checked exact full SHA overlap between **3,192 Master SHA-recovered sources (all Objaverse)** and the historical Core370 full source SHA set. It excluded every Quaternius (191) and KayKit (37) Master asset because their existing Master inventory had blank full-source SHA and `LEGACY_PATH_MATCH_UNREHASHED` status.

The historical Master `candidate_id` for these two providers contains a 24-hex-character source-content SHA-256 prefix. This was independently checked by re-downloading three KayKit original source files and two Quaternius originals from the private persistent Drive and recomputing exact SHA-256 over the original bytes. All five files matched their candidate ID prefixes.

Comparing the Master legacy prefixes against **Core370 original SHA-256 prefixes** yielded **53 one-to-one pairs**:

| Provider | Cross-corpus source-pair candidates |
|---|---:|
| Quaternius | 48 |
| KayKit | 5 |
| **Total** | **53** |

The 53 pairs are unique on both Master and Core sides. They are **NOT** 53 certified unique-character reductions. The two cohorts' source alias mapping is unresolved for all remaining sources, and cross-format geometry/semantic identity is not closed.

## Split contamination risk

| Master historical split | Linked Core370 sources | Policy |
|---|---:|---|
| FIT | 37 | Quarantine until source cluster and Core split settled |
| DEV | 8 | P0 protected cross-corpus identity court |
| TUNE | 4 | P0 protected cross-corpus identity court |
| CAL | 3 | P0 protected cross-corpus identity court |
| EXTERNAL_HOLDOUT | 1 | P0 protected cross-corpus identity court |
| **Total FIT-external risk** | **16** | No cross-split admission until adjudication |

The protected Quaternius source `Zombie_Female.fbx` has:
- Master canonical ID `asset_746f9c7168ac22c1e444fd1a`, split `EXTERNAL_HOLDOUT`;
- Core370 subject ID `corpus_4cbc16c165b169871ce9`;
- re-downloaded actual-source SHA-256 `155f0ef09ce23a61d016f3e07782d7faa3ed7e75d9d1241bf4d649e5a2ed2569` exactly matches the historical Core original SHA.

The Quaternius `Knight_Male.fbx` CAL link is also **original bytes confirmed**:
- Master `asset_824b3498a6937e42d3a5ab93` (CAL);
- Core370 `corpus_2915c4f0f9c253c1ec1c`;
- exact source SHA-256 `002a5a7f89ba1e1f9bb79efa31acd2ac1083ba136480b1bd9b3932b368ec0aa6`.

A KayKit `Rogue_Hooded.glb` original file independently matched Core370 full SHA `93e6e25213009952276d9cf34f5d96a243767334c66f280db0433ddfabb91545` (the third byte-verified cross-corpus duplicate). **Actual training leakage is NOT proven:** only a potential conflict if the same source cluster were consumed across train and assessment.

## Quarantine overlay (not data rewriting)

The 53 source-pair identities correspond to 106 current provisional source records. With 37 intended head supervision targets each, **3,922 head-target routes** are put into a distinct **identity/split review quarantine**, including **1,184 routes** derived from 16 protected-split pairs.

The original V2 161,986 candidate routes remain untouched, with training and target loss masks all False. The 3,922-row overlay is only an additional, SHA-pinned negative-gate receipt, **not** 3,922 originally admitted examples being revoked.

## Production readiness gates

1. Rehash all 50 remaining source-byte overlap candidates (do not treat 96-bit prefixes as independent full SHA proof).
2. Recover full SHA for all 228 Master Quaternius/KayKit source records through immutable source file readback; preserve original creator/license receipts independently.
3. Reconcile source IDs and identity clusters across both cohorts (including other formats and repeated assets) and **freeze splits at identity-cluster level** before head training.
4. Run independent character/prop and morphology source scene courts. `VikingHelmet.fbx`, `Chef_Hat.fbx`, and non-character objects are valid possible negatives or QA, NOT guaranteed character training data.
5. Re-run head-specific truth admission for all six heads. Original numeric imports and historical PASS do not confer current source or model training authority.

## Implementation (separate draft PR, no main merge)

- `tools/corpus/audit_cross_corpus_source_identity_v1.py`: SHA-manifest-locked historical Master/Core import; SHA96 match; optional actual original source-byte full SHA check; protected-split priority, fail closed.
- `tools/corpus/build_cross_corpus_identity_quarantine_v1.py`: joins the private original 37-target routes; proves quarantine of both source records; rejects a forged PASS or missing counterpart.
- `tests/corpus/test_cross_corpus_source_identity_v1.py`: source SHA tamper, role/split guard, prefix caution, duplicate core collisions and training authority firewalls.
- `tests/corpus/test_cross_corpus_identity_quarantine_v1.py`: pair and route hash integrity, both sides present, all 37 targets masked.
- `.github/workflows/realsas-cross-corpus-source-identity-contracts-v1.yml`: synthetic-only CI; actual source inventories remain in private Google Drive, not in public logs.

**Technical metrics are provisional crosswalk evidence, not licenses, identity cluster completion, or approved supervised training targets.**
