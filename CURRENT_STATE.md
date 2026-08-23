# RealSaS-OPT — Current State

**Date:** 2026-08-23  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `IRIS_CONTROLLED_V1_REPRESENTATION_CEILING_PASS__OPEN_PILOT_NEXT__OPTIMIZER_PRODUCTION_0__SEALED_CLOSED`

## Read this first

This file is the **single continuation authority**. In a new chat, if the user says only “devam et” or “GitHub'a bak ve devam et”, continue from the exact next action in this file. Do not ask the user to reconstruct history.

## Scientific target

**Observable evidence was not pruned; the problem was pruned.**

Current IRIS target:

`8 controlled neutral views -> persistent observable surface geometry + uncertainty`

Legal evidence contract:

- common-frame P;
- geometric N;
- visibility/support;
- uncertainty/risk;
- coarse/fine persistence;
- reciprocal/cycle consistency;
- set-valued ambiguity;
- provenance.

Forbidden IRIS target authority: joint IDs, owner IDs, parent graph, skin weights, pose-B mechanics, GFDR hidden mechanics.

## Current corpus

Canonical selected: 3993.

Preserved exclusions:

- 56 repair-pending assets;
- 6 active non-Basis shape-key assets;
- 1 all-8 blank asset.

Controlled V1 usable: **3930**.

Frozen original split membership:

- FIT 2935;
- TUNE 313;
- CAL 246 — SEALED;
- DEV 270 — SEALED;
- EXTERNAL_HOLDOUT 166 — SEALED.

Open FIT+TUNE count = **3248**. No resplit authorized.

## Frozen package authority

Drive package:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1`

Drive folder ID: `15Du2plm2vHYe4Mmm-p1-L6emkxYucN8k`.

Base `PACKAGE_MANIFEST_V1.json` SHA-256:

`e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098`

Base package was sealed before optimizer; CAL/DEV/EXTERNAL remain unopened.

## Architecture

```text
8 ordered RGBA views + known controlled yaw
        ↓
deterministic input contract
        ↓
shared multiscale encoder
→ within-view axial reasoning
→ row-constrained multi-view fusion
→ dense decoder
→ P / N / U / Z_coarse / Z_fine
        ↓
deterministic top-k / reciprocal / cycle / local rerank
→ set-valued ambiguity
→ support + P/N fusion
→ reprojection/provenance
```

Authority: `experiments/iris_controlled_v1/ARCHITECTURE_AND_BOUNDARIES_V1.md`.

## REAL REPRESENTATION CEILING — PASS

Decision authority:

`experiments/iris_controlled_v1/CEILING_PASS_AND_OPEN_PILOT_DECISION_20260823.md`

Executed on 64 deterministic open assets (51 FIT / 13 TUNE), **65,830 cross-view pairs**.

Results:

- P_EXACT top1 = `0.9976302598`
- P_EXACT top4 = `1.0`
- P_EXACT top8 = `1.0`
- P_EXACT family top1 p10 = `0.9934295619`
- PN_EXACT top1 = `0.9942123652`
- PN_EXACT top8 = `0.9998936655`
- P noise 0.0025 top8 = `1.0`
- P noise 0.005 + N noise 20° top1 = `0.9206744645`
- P noise 0.005 + N noise 20° top8 = `0.9996658059`
- ambiguous-within-0.003 fraction = `0.0151754519`

Primary representation gate = **PASS**.

Interpretation:

- legal observable P/(P,N) can address persistent surface loci in the controlled exact-camera setting;
- this is **not** learner proof;
- P alone slightly outperformed naive fixed-weight P+N top1, so N must not be forced into matching with an uncalibrated constant coefficient;
- noisy top8 staying ~99.97% is strong controlled support for high-recall top-k/set-valued ambiguity rather than forced singleton matching.

## Operational cache incident

Original serial prepare was deprecated after `20/3248` took ~16 min (>40 h projection).

Fast V2 ceiling cache completed 64 assets in 311 s and passed correspondence parity exactly.

During full 3248 expansion, initial rate burst above 1 asset/s but decayed toward ~0.26 asset/s due Google Drive random-read pressure, implying roughly 3+ h cache preparation.

This is an **operational I/O issue, not a scientific failure**.

Partial local fast cache artifacts are preserved in the live Colab `/content/IRIS_CONTROLLED_V1_FAST_CACHE`. Do not delete while the runtime is alive.

## Training implementation correction before real training

A checkpoint-selection bug was found before production optimizer launch:

- epochs 0–3 optimize P/N/U only;
- persistence terms enter at epoch >=4;
- old trainer compared TUNE `total` across those changing objective definitions.

That could make a warmup checkpoint look artificially best simply because it contained fewer terms.

Mandatory corrected trainer:

`experiments/iris_controlled_v1/train_iris_controlled_v1_v1_1.py`

It preserves FIT warmup but evaluates TUNE selection with the **same full post-warmup objective at every epoch**.

SHA-256 of Drive-uploaded v1.1 trainer:

`961b6469b54e74222bd3055de68bfa7aa96042a04b59f2f97f1ee586aeb982d3`

The original trainer remains historical lineage and is not the current selection authority.

## NEXT SCIENTIFIC GATE — OPEN PILOT LEARNER

Do **not** spend hours finishing the full 3248 cache yet.

Use the already completed 64-asset ceiling manifest:

`/content/IRIS_CONTROLLED_V1_FAST_CACHE/IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json`

Pilot:

- 51 FIT train;
- 13 TUNE evaluate;
- 8 epochs;
- CAL/DEV/EXTERNAL untouched;
- random-init witness recorded first;
- compare trained state against identical open TUNE evaluation.

Diagnostic PASS requires:

- TUNE P Euclidean error reduced by at least 20%;
- coarse persistence top8 improves by >5 percentage points over random init;
- fine persistence top8 improves by >5 percentage points over random init.

Executable:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python run_iris_controlled_v1_open_pilot.py --epochs 8
```

Before running pilot, interrupt any still-running full cache process with `Ctrl+C`. The existing 64 manifest remains valid; additional local cache assets are harmless and preserved.

After pilot, inspect `PILOT_SUMMARY.json` before authorizing full cache completion or architecture changes.

## Production training remains NOT STARTED

The bounded open pilot may take optimizer steps, but it is **diagnostic**, not the production 3930/3248 Controlled V1 training run.

Production full-cache/full-training authorization waits for pilot interpretation.

No sealed panel has been opened.

## VERY IMPORTANT M4 audit

Authority:

`experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

Frozen interpretation:

- Graph V1 identity path was identity-contracting and is falsified;
- Q representation survives;
- QF V1 did not test the intended joint A∪B quotient;
- scalar threshold/admission consumer was falsified;
- CORR is causally supported;
- typed signed joint tiny closure passed;
- V2-C did not prove family-disjoint generalization;
- UNKNOWN existed historically; the novelty is not UNKNOWN itself;
- current change is **problem-pruned, evidence-richer**.

Mandatory downstream question remains open:

> Can richer observable/set-valued evidence produce a RigAnything-like 2.5D equivalent substrate sufficient for Geppetto and functional rigging?

The M4 audit freezes E0–E5 gates for that question. Representation ceiling PASS is only one prerequisite, not downstream product closure.

## Deferred work — preserved, not forgotten

- Stage-B7 publication/rerender of 56 repaired assets;
- source-appearance B pass;
- stylized/product C pass;
- camera jitter / unknown-camera frontend;
- artist-specific cross-view residual modeling.

Do not reopen these by default before the controlled learner result is understood.

## Research operating rule

```text
apparatus/data
  ↓
representation/target
  ↓
learner/optimizer
  ↓
evidence consumer
  ↓
only then genuine information limit
```

Do not infer impossibility from learner failure. Do not infer product sufficiency from exact-oracle/representation PASS. Test downstream consequence.
