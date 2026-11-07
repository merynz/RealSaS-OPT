# RealSaS Global Truth Foundry V4 — original-byte dedup closure of 53 source pairs

**2026-11-07. Status: 53/53 source-byte SHA-256 duplicate edges VERIFIED; corpus-level training and semantic dedup still CLOSED.**

This supersedes the preliminary counts in `REALSAS_CROSS_CORPUS_53_IDENTITY_SPLIT_FIREWALL_V1_20261107.md` which recorded only three actual original-byte SHA readbacks at that time.

## Full original-source rehash verified

We crosswalked the frozen private V1 source ledgers:

- `MASTER3993_SOURCE_TRUTH_INVENTORY_V1.csv`: 3,993 rows, SHA256 `15e90dbd5ffd470737fb230d9c34138a63b98fcc5057a09450d3239984db20d6`.
- `CORE370_370_MACHINE_SUBJECTS_V1.csv`: 370 rows, SHA256 `2c149c72a64f3a57d957b88f31a7becf4215beff46f3330ef537dae0b7d0d307`.

Earlier Master rehash work had 3,192 SHA-recovered **Objaverse** originals only. This left all 191 selected Quaternius and 37 KayKit Master sources in `LEGACY_PATH_MATCH_UNREHASHED`. Their historical `candidate_id` contains the first 24 hex characters of source SHA256, allowing a candidate join with Core370 full SHA.

**We then re-downloaded every source identified by these 53 candidate pairs from the immutable original Master Drive source archive**, and independently recomputed full SHA-256 on the actual file bytes.

| Original source court | Count |
|---|---:|
| Quaternius full-byte SHA match, mostly binary FBX | 48 |
| KayKit full-byte SHA match, glTF binary GLB | 5 |
| **Confirmed byte-identical Master ↔ Core370 source pairs** | **53** |
| Prefix-only overlaps still awaiting byte readback | **0** |
| Master+Core record IDs participating | 106 |
| Head-target routes in independent quarantine overlay | 3,922 |

The 53 Master and Core IDs match one-to-one; no ambiguous many-to-many pair in this court. Each exact digest is recorded in the private SHA-pinned pair ledger. All original bytes remain unchanged. This proves **physical source identity** for these 53 edges, *not* generic cross-format or semantic duplicates among other sources.

## Split authority and unseen integrity

| Historical Master split | Exact duplicate Core370 source pairs | Priority |
|---|---:|---|
| FIT | 37 | P1 — Core split/lineage unresolved |
| DEV | 8 | P0 — evaluation linkage |
| TUNE | 4 | P0 — protected tuning linkage |
| CAL | 3 | P0 — calibration linkage |
| EXTERNAL_HOLDOUT | 1 | P0 — unseen holdout linkage |

**All 16 Master non-FIT duplicates have been verified from real original source bytes**. The external holdout example is `Zombie_Female.fbx`, exact SHA256 `155f0ef09ce23a61d016f3e07782d7faa3ed7e75d9d1241bf4d649e5a2ed2569`. The CAL example is `Knight_Male.fbx`, exact SHA256 `002a5a7f89ba1e1f9bb79efa31acd2ac1083ba136480b1bd9b3932b368ec0aa6`. A FIT example is KayKit `Rogue_Hooded.glb` exact SHA256 `93e6e25213009952276d9cf34f5d96a243767334c66f280db0433ddfabb91545`.

**No observed training leakage claim**: detection of identical files across two *candidate* corpora does not prove they were ever used simultaneously in an effective train/test run. It does prove a very material leak risk if current/future head admission ignores source identity clusters.

The overlay maintains all 106 original source IDs and all 3,922 blocked head-target candidate routes, including 1,184 high-priority routes from the 16 protected pairs. It never merges original source rows or authorizes training.

## Identity accounting is NOT training count

All 4,406 known recorded entities (3,993 Master + 370 Core370 + 15 QA-only + 28 previously rejected candidate-only) remain in the original register. Of these, 4,378 have canonical asset IDs, but 28 historical rejects have only candidate IDs.

Applying **only the 53 proven same-byte source edges** yields **4,325 provisional graph components** among the 4,378 canonical-ID records. This number is neither the final unique source count nor the count of characters, admitted supervision or licensed assets. Cross-format duplicates, within-cohort repeats, prop/object filtering, QA status and source family lineage all remain unclosed.

For the 228 Master Quaternius/KayKit sources, this specific court has fully rehashed 48 + 5 = 53, leaving **143 Quaternius + 32 KayKit = 175** source records for later original-byte revalidation. The original source folder does contain the archived files (191 Quaternius root entries and 37 selected KayKit filename matches), but existence does not equal hash or Blender scene parity.

## Next engineering and admissibility gates

1. Extend immutable source-byte SHA rehash to the 175 remaining Quaternius/KayKit assets; independently source-recover remaining Objaverse and QA/reject cohorts without holdout contamination.
2. Build a deterministic physical source lineage cluster and split firewall covering *all* assets, including derivatives and semantic/geometry duplicates across different format/exports.
3. Audit true character-vs-prop morphology and original scene graphs. Exact overlaps include `VikingHelmet.fbx`, a reminder that the presence of an FBX is not a semantic character label.
4. Perform orientation, authored pose, camera, mesh topology, modular/open mesh, attachment, UV/texture, rig hierarchy, rest/bind, original skin and animation intervention checks using source SHA-pinned receipts.
5. Produce **head-specific supervision**: StdGEN++-inspired IRIS Geometry/Semantic/Graphics source-bound references; SkinTokens-inspired AXIS/MIRA exact rig/weight/deformation correspondence; TESSA mesh/topology and mechanical surface. Graphics must have independently approved stylized **2D art targets**, not mere source 3D RGB.
6. Permit loss masks only after independent source truth, legal rights, leakage-safe split and head-specific courts PASS. **All masks remain closed now.**

## Immutable implementation

- `tools/corpus/audit_cross_corpus_source_identity_v1.py` — source SHA-prefix join with optional actual original file-byte full rehash; no training PASS.
- `tools/corpus/build_cross_corpus_identity_quarantine_v1.py` — two-sided SHA-pin / all-target quarantine.
- `tools/corpus/build_verified_cross_corpus_source_clusters_v2.py` — full-SHA-only source identity grouping; frozen 4,406 source-roster SHA and original split firewall.
- `tests/corpus/test_cross_corpus_source_identity_v1.py`, `test_cross_corpus_identity_quarantine_v1.py`, `test_verified_cross_corpus_source_clusters_v2.py` — synthetic-only safety verification.
- `canonical/REALSAS_GLOBAL_COHORT_REGISTRY_V2.json` — latest source/holdout counts and explicit unknowns.
- Private V4 ZIP contains real data ledgers and SHA manifests; source asset bytes are NOT redistributed in the ZIP.

**No merges, source deletion, model edits, training, materialization, or silent PASS.**
