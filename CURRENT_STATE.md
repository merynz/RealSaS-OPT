# RealSaS-OPT — Current State

**Date:** 2026-08-23  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `IRIS_CONTROLLED_V1_FAST_PREP_READY__OPTIMIZER_0__SEALED_CLOSED`

## Read this first

This file is the **single continuation authority**. In a new chat, if the user says only “devam et” or “GitHub'a bak ve devam et”, continue from the exact next action in this file. Do not ask the user to reconstruct history.

Detailed handoff: `experiments/iris_controlled_v1/CONTINUATION_HANDOFF_V1.md`.

## Current scientific correction

**Observable evidence was not pruned; the problem was pruned.**

The old M4 pressure was effectively:

`static observations -> privileged exact hidden mechanical-owner identity`

The current IRIS problem is:

`8 controlled neutral views -> persistent observable surface geometry + uncertainty`

The current evidence contract is richer and explicit:

- common-frame P;
- geometric N;
- visibility/support;
- uncertainty/risk;
- coarse and fine persistence evidence;
- reciprocal/cycle consistency;
- set-valued ambiguity;
- provenance.

IRIS is not authorized to use joint IDs, owner IDs, parent graph, skin weights, pose-B mechanics or GFDR fields as target authority.

## Canonical IRIS system boundary

```text
8 ordered RGBA views + known controlled yaw
        ↓
DETERMINISTIC FRONTEND
  order / alpha / resolution / camera geometry
        ↓
NEURAL CORE
  Shared multiscale encoder
  → within-view axial reasoning
  → row-constrained multi-view fusion
  → dense decoder
  → P / N / U / Z_coarse / Z_fine
        ↓
DETERMINISTIC EVIDENCE LAYER
  geometry-constrained candidate domain
  → coarse top-k
  → reciprocal / cycle
  → local fine rerank
  → set-valued ambiguity
  → support + robust P/N fusion
  → reprojection falsification
  → provenance
        ↓
QUALIFIED OBSERVABLE IRIS EVIDENCE
```

Architecture authority: `experiments/iris_controlled_v1/ARCHITECTURE_AND_BOUNDARIES_V1.md`.

## Immediate controlled corpus

Canonical selected assets: **3993**.

Immediate exclusions, all preserved:

- 56 Stage-B6 repair-pending assets — repaired geometry frozen with SHA, but current A renders still bind old geometry;
- 6 active non-Basis shape-key assets — rest-state qualification deferred;
- 1 all-8 blank asset — observation-ineligible.

Therefore Controlled V1 uses **3930 assets**.

Frozen original split membership is preserved; no random resplit:

- FIT 2935;
- TUNE 313;
- CAL 246 — SEALED;
- DEV 270 — SEALED;
- EXTERNAL_HOLDOUT 166 — SEALED.

Open preparation/training count is FIT+TUNE = **3248**.

Source mix remains 3702 Objaverse / 191 Quaternius / 37 KayKit.

## Frozen byte authority

Canonical base execution package is in Google Drive:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1`

Drive folder ID: `15Du2plm2vHYe4Mmm-p1-L6emkxYucN8k`.

Frozen base `PACKAGE_MANIFEST_V1.json` SHA-256:

`e49a67b2ef808fe4f7cc9e414e024d30ab0fddc0ea55099bfa33ef78dbc6f098`

Base package seal:

- 23 listed files;
- optimizer steps `0`;
- sealed panels opened `false`.

The fast-prep patch is an additive pre-optimizer engineering amendment. It does not rewrite the base package.

Fast patch authorities:

- `experiments/iris_controlled_v1/FAST_PREP_ADDENDUM_V1.md`
- `prepare_iris_controlled_v1_fast.py` SHA-256 `8ce6e0a6cdbd25890d25703aee3a41d4b290d80bdb81c05987dc11630a515ec7`
- `launch_iris_controlled_v1_fast.py` SHA-256 `1f35904429912b626dab815ed6af871d3e67fd61729963d5e2973017e421c690`

Both fast scripts are also stored in the Drive package directory. The fast launcher verifies the original frozen base package before work starts.

## Training contract — frozen before optimizer

Authority: `experiments/iris_controlled_v1/IRIS_CONTROLLED_V1_PREREG.md` plus the pre-optimizer engineering addendum `FAST_PREP_ADDENDUM_V1.md`.

Core settings remain:

- 24 epochs;
- batch 1;
- AdamW lr 5e-5;
- weight decay 1e-4;
- grad clip 2.0;
- seed 20260823;
- neural input 256;
- FIT trains;
- TUNE selects;
- CAL/DEV/EXTERNAL remain closed.

Loss after warmup:

`L = P + .25 N + .05 U + .10 Zc + .05 Zf + .20 P_consistency`.

Epochs 0–3 use P/N/U only. Persistence terms enter at epoch 4.

Fast-prep freezes the 512→256 derived RGBA preprocessing as PIL RGBA bilinear before optimizer step 1. Geometry/persistence target authority is unchanged.

## Completed no-optimizer preflight

A real canonical FIT asset was executed through:

`master geometry + 8 raster authorities -> exact physical tracks -> model -> full loss -> backward`.

Result:

- 128 cross-view physical tracks;
- trainable parameters: 5,657,863;
- finite full loss;
- nonzero gradient;
- PASS.

Representation apparatus smoke on the same real asset:

- 390 cross-view pairs;
- exact P top1/top8 = 1.0 / 1.0;
- exact P+N top1/top8 = 1.0 / 1.0.

These are apparatus witnesses, not confirmatory generalization results.

## Operational incident — serial prepare deprecated

On the first GPU Colab launch, the original serial preparation path reached only `20/3248` assets after about 16 minutes: roughly **48 seconds per asset**, projecting to >40 hours before the representation ceiling.

Root cause is engineering/I/O design, not corpus corruption or an IRIS scientific failure:

- thousands of Drive small-file opens were serialized;
- raster/image files were re-read for per-source SHA after already being consumed;
- sixteen 512×512 RGBA images were recompressed per asset;
- all 3248 open assets were prepared **before** a ceiling that only requires 64.

No optimizer step occurred. No sealed panel was opened.

The partial original cache under `cache/IRIS_CONTROLLED_V1` is **DO-NOT-DELETE lineage** but is not the current preparation path.

The original `launch_iris_controlled_v1.py --mode all` is **operationally deprecated for fresh runs**. Its scientific code remains historical authority.

## FAST preparation policy

The replacement path:

1. constructs the deterministic 64 open assets needed by the representation ceiling first;
2. writes derived caches to Colab local SSD `/content/IRIS_CONTROLLED_V1_FAST_CACHE`;
3. uses bounded parallel Drive reads;
4. uses a vectorized correspondence implementation;
5. runs an actual-source parity check against the original correspondence function before preparation and requires zero visibility/row mismatch and <=1e-6 error difference;
6. uses parent corpus/package SHA authority instead of redundantly hashing every source file again;
7. SHA-seals every derived cache artifact;
8. expands to the remaining open assets only after the 64-asset ceiling PASSes.

The 64 cache artifacts are reused during full expansion.

## NEXT EXECUTABLE STEP

**Stop any still-running original serial preparation with Ctrl+C.** Its partial outputs are safe and preserved.

Keep the current NVIDIA Colab runtime mounted. From the Drive package directory run:

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python launch_iris_controlled_v1_fast.py --mode all --workers 8
```

Expected order:

```text
verify frozen base package
→ build deterministic 64-asset local ceiling cache
→ real-source old-vs-vectorized correspondence parity PASS
→ representation ceiling on those 64
→ only if PASS: expand local cache to all FIT+TUNE = 3248
→ only if full cache PASS: start optimizer
```

The fast builder prints `rate=... asset/s` and `ETA_min=...` every 10 completed assets, so throughput is immediately observable.

If only the ceiling is desired first:

```bash
python launch_iris_controlled_v1_fast.py --mode ceiling --workers 8
```

After ceiling PASS, continue:

```bash
python launch_iris_controlled_v1_fast.py --mode prepare-full --workers 8
python launch_iris_controlled_v1_fast.py --mode train
```

Training resume:

```bash
python launch_iris_controlled_v1_fast.py --mode train --resume
```

## Sealed policy

Do **not** inspect CAL, DEV or EXTERNAL_HOLDOUT during training/model selection. TUNE performance is not product generalization.

After the open run, first classify the result. A separate explicit recorded authorization is required before sealed evaluation.

## VERY IMPORTANT M4 audit

Authority:

`experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

Frozen interpretation:

- Graph V1 identity path was genuinely identity-contracting and is falsified;
- Q representation survived;
- QF V1 did not test the intended joint A∪B quotient;
- scalar pair threshold/admission formulation was falsified;
- CORR is causally supported;
- typed signed joint tiny closure passed;
- V2-C did not establish family-disjoint generalization;
- UNKNOWN existed historically; the novelty is not UNKNOWN itself;
- current change is **problem-pruned, evidence-richer**.

Most important unresolved product question:

> Can this richer observable/set-valued evidence actually produce a RigAnything-like 2.5D equivalent substrate sufficient for Geppetto and downstream functional rigging?

The audit freezes E0–E5 gates for that question. Prior S0/B2 evidence is supportive but not complete qualification.

## Deferred corpus work — preserved, not forgotten

Stage-B7 is **DEFERRED**, not cancelled.

Do not reopen the 56-asset repair/rerender thread unless:

1. a measured Controlled V1 result depends on those assets; or
2. a final production retrain is being prepared.

Also deferred:

- source-appearance B pass;
- stylized/product C pass;
- camera jitter / unknown-camera robustness;
- artist-specific cross-view residual modeling.

These return after the strongest controlled case is measured.

## Historical baseline preservation

The older 238-family G1 prereg/baseline remains historical and untouched. It is **not** the current training authority. Controlled V1 is a new explicit line with 3930 clean assets and a new preregistration.

No historical M4/S0/G1 evidence should be deleted merely because interpretation changed.

## Research operating rule

For every PASS/FAIL:

```text
apparatus/data?
  ↓
representation/target?
  ↓
learner/optimizer?
  ↓
evidence consumer?
  ↓
only then genuine information limit
```

Do not infer impossibility from learner failure. Do not infer product sufficiency from an exact-oracle PASS. Test the actual downstream consequence.
