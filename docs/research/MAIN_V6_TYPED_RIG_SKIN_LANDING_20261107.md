# RealSaS / BUSTED — V6 typed Rig/Skin contract landing slice

**Date:** 2026-11-07  
**Target:** up-to-date `main` via a short review-only PR  
**Provenance:** exact, unchanged source code from V6 PR #297 head
`dd3089957e3126d93db4ad2804e30511cfdca348`.

## Why this slice exists

Issue #361 found a 137-commit V6 branch diverged from main by 339 later main
commits. Its successor V7 PR #352 and Mechanics PR #353 both depend on that V6
base. Bulk-merging the old branch would introduce unnecessary branch ancestry
and concurrent Platform/Graphics/Corpus risks.

The exact imports `canonical_mesh_ir_v1.py`, `carrier_skin_v1.py`,
`hashing.py`, and `types.py` were checked in both main and V6:
their Git blob SHAs **match**. This enables a zero-rewrite import of the
existing typed contracts, with no new producer or product runtime.

## Included (existing, unchanged V6 code)

- `rig_program_ir_v1.py`: qualified source proposal and causal rig-program
  contract. Does **not** learn, repair or execute missing controls.
- `rig_constraint_schema_v1.py`: five currently declared executable
  constraint operators `IK/AIM/FOLLOW/COPY_TRANSFORM/STRETCH`.
- `skin_code_ir_v1.py`: typed SkinCode proposal; does not decode codec.
- `canonical_rig_ir_v2.py`: distinct deform-structure and control-program
  hashes. Controls edits should not silently invalidate skin weight identity.
- `canonical_skin_ir_v2.py`: full mesh/rig-deform/codec evidence lineage;
  does not grant learned-model or ProductRevision authority.

Source-equivalent reviewed tests:
`test_control_ready_rig_v2.py`,
`test_rig_program_causal_linear_v1.py`.
New main-focused negative tests:
`test_v6_typed_rig_skin_main_integration_v1.py`.
An independent `v6-main-contract-integration-v1` GitHub Action runs
the exact tests without touching V7/Corpus/Graphics.

## Deliberately excluded

- V6 Rig/Skin/Motion/Mechanics/Graphics product workers and Go capability
  registration, and all external ModelRelease/Artifact Registry changes.
- V7 neural/codec code, training or Corpus370 state; no authority is promoted.
- Mechanics opt-in multi-goal solver, secondary motion and ready-to-ship claims.
- Any edit or merge of #297, #352, #353, #327 or Graphics research PRs.

## Verification and next steps

This is a **typed-contract-only landing**. CI passes only if source-equivalent
V6 tests and new hash/unsupported-op negatives pass on *current main*.
Do not label the existence of these classes as a product runtime.

Following a successful reviewed merge, create **separate, short, main-based**
PRs for:
1. CanonicalMechanics baseline + FABRIK/LBS, preserving Knight regression.
2. CAS workers/Go capability wiring with exact ProductRevision authority.
3. V7 research/model behind a separate release flag, only after dedicated
   `axis-mira-v7-research` workflow runs successfully.
4. Mechanics multigoal/dual-grip research opt-in with source and capability
   qualification, not by copying #353 history wholesale.
5. One Mechanics-owned surface picker: compare V7 raw skin-support picker
   with the Mechanics graph-reachability picker under negative controls;
   preserve source provenance and abstention.

This PR must never be interpreted as a source-truth, unseen-learning, product
FPS or end-to-end puppet release result.
