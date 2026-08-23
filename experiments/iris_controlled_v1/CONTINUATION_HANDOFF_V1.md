# IRIS Controlled V1 — Continuation Handoff

**Date:** 2026-08-23  
**Branch:** `g0-g1/single-pose-geometry`  
**Status:** FAST PREP PATCH READY / OPTIMIZER STEPS 0 / SEALED CLOSED

## If the user says only “devam et”

1. Read root `CURRENT_STATE.md` first; it is the single continuation authority.
2. Read `experiments/iris_controlled_v1/FAST_PREP_ADDENDUM_V1.md` before executing cache preparation.
3. Read `experiments/iris_controlled_v1/IRIS_CONTROLLED_V1_PREREG.md`.
4. Read `experiments/iris_controlled_v1/ARCHITECTURE_AND_BOUNDARIES_V1.md`.
5. Read `experiments/iris_controlled_v1/TRAINING_READY_REPORT_V1.md`.
6. Read `experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md` before making any claim that the observable substrate is product-sufficient.
7. Do **not** reopen the 56-asset repair thread unless a measured result depends on it.
8. Do **not** reopen privileged exact-owner M4 as the default target.

## Byte authority

The canonical base executable package is the Google Drive package:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1`

Drive folder ID: `15Du2plm2vHYe4Mmm-p1-L6emkxYucN8k`

Base `PACKAGE_MANIFEST_V1.json` SHA-256:

`e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098`

Base package seal:

- listed files: 23;
- optimizer steps: 0;
- sealed panels opened: false.

The fast preparation patch is additive and was frozen before optimizer step 1:

- `prepare_iris_controlled_v1_fast.py` SHA-256 `8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7`
- `launch_iris_controlled_v1_fast.py` SHA-256 `1f35904429912b626dab815ed6af871d3e67fd61729963d5e2973017e421c690`

Both live in the same Drive package folder and in GitHub for inspection.

## Immediate corpus

Selected canonical assets: 3993.

Excluded from immediate Controlled V1:

- 56 repair-pending assets, frozen and SHA-preserved after Stage-B6;
- 6 active non-Basis shape-key assets;
- 1 all-8 blank observation asset.

Usable controlled corpus: **3930**.

Frozen split:

- FIT 2935;
- TUNE 313;
- CAL 246 — sealed;
- DEV 270 — sealed;
- EXTERNAL_HOLDOUT 166 — sealed.

Open preparation/training count: **3248**.

No resplitting is authorized.

## Immediate scientific target

The evidence was **not pruned**. The problem was pruned.

Controlled V1 asks:

> Can eight controlled neutral-pose images support recovery of persistent observable surface geometry and uncertainty?

Neural outputs:

- P;
- geometric N;
- U/risk;
- Z_coarse;
- Z_fine.

Deterministic IRIS evidence layer owns support, top-k candidate retention, reciprocal/cycle checks, local reranking, set-valued ambiguity, common-frame fusion, reprojection checks and provenance.

No joint IDs, owner IDs, parent graph, skin weights, pose-B mechanics or GFDR mechanical fields are legal IRIS target authority.

## No-optimizer evidence already completed

Real corpus preflight:

- exact master asset consumed;
- physical surface tracks built from triangle+bary authority;
- 128 multi-view tracks;
- model → losses → backward finite;
- 5,657,863 trainable parameters;
- nonzero gradient;
- PASS.

Representation apparatus smoke:

- 390 cross-view pairs on the real preflight asset;
- exact P top1/top8 = 1.0 / 1.0;
- exact P+N top1/top8 = 1.0 / 1.0;
- heavy P/N perturbation retained top8 = 1.0 in this smoke.

This is apparatus validation, not the confirmatory gate.

## Operational incident: original serial prepare

The first all-in-one run reached only `20/3248` after about 16 minutes, approximately 48 seconds per asset and >40 hours projected before the ceiling.

This was diagnosed as an engineering/I/O design failure:

- serial Drive small-file reads;
- redundant source file SHA re-reads;
- per-asset recompression of sixteen 512×512 RGBA images;
- full 3248 preparation before a ceiling that needs only 64.

No optimizer step occurred and no sealed split was opened.

The partial original cache is preserved as lineage. Do not delete it. Do not continue the original serial fresh-run path.

## Current next executable action

If the old serial process is still running, stop it with `Ctrl+C`.

On the already-mounted NVIDIA Colab runtime:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python launch_iris_controlled_v1_fast.py --mode all --workers 8
```

The fast launcher does:

```text
verify frozen base package
→ build only deterministic 64 ceiling assets to /content local SSD
→ actual-source original-vs-vectorized correspondence parity check
→ exact representation ceiling
→ if PASS, expand to all FIT+TUNE=3248 using the same local cache
→ if full prep PASS, optimizer starts
```

The fast builder prints live throughput and ETA every 10 completed assets.

Staged alternative:

```bash
python launch_iris_controlled_v1_fast.py --mode ceiling --workers 8
python launch_iris_controlled_v1_fast.py --mode prepare-full --workers 8
python launch_iris_controlled_v1_fast.py --mode train
```

Training resume:

```bash
python launch_iris_controlled_v1_fast.py --mode train --resume
```

The derived training cache is intentionally local/ephemeral; parent source authority remains frozen on Drive and every derived cache artifact is SHA-sealed. Full cache manifest/seal/progress are copied into the Drive run provenance directory before training.

## After training

Inspect FIT/TUNE only. Do not open CAL/DEV/EXTERNAL automatically.

Classify failure before changing architecture:

- apparatus/data;
- representation/target;
- learner/optimization;
- persistence/evidence layer;
- consumer;
- genuine information limit only last.

A separate recorded authorization is required before any sealed panel is opened.

## Mandatory downstream question after / alongside Controlled V1

The M4 audit freezes an unsolved product-level question: does richer observable evidence with set-valued ambiguity actually provide enough substrate for Geppetto and a functionally equivalent 2.5D rig?

Run the E0–E5 program in the VERY IMPORTANT M4 audit before declaring product substrate closure:

- exact observable substrate ceiling;
- Geppetto oracle-substrate sufficiency;
- extractability;
- predicted-substrate sufficiency;
- ambiguity stress;
- RigAnything-like 2.5D functional equivalence.

Prior S0/B2 evidence is supportive but not a substitute for the full E0–E5 qualification.

## Deferred work — intentionally not on current critical path

- Stage-B7 publication/rerender of 56 repaired assets;
- source-appearance B render;
- stylized/product C render;
- camera jitter / unknown-camera frontend;
- artist-specific cross-view residual modeling.

These are not forgotten. They are deferred until controlled IRIS evidence justifies reopening them.
