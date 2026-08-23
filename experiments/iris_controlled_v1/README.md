# IRIS Controlled V1 — Training Package

Status: **TRAINING-PREPARED / OPTIMIZER NOT YET STARTED**  
Date: 2026-08-23

This package is the executable continuation line for the 3,930-asset controlled IRIS corpus.
It does **not** rewrite the historical 238-family G1 baseline; that lineage remains frozen for provenance.

## Controlled corpus

- canonical selected: 3,993
- immediate controlled usable: **3,930**
- deferred repair: 56 (SHA-frozen, not deleted)
- active non-Basis shape-key quarantine: 6
- all-8 blank observation quarantine: 1

Frozen split membership is preserved from the master corpus; no random resplit:

| split | count | access |
|---|---:|---|
| FIT | 2935 | OPEN |
| TUNE | 313 | OPEN |
| CAL | 246 | SEALED |
| DEV | 270 | SEALED |
| EXTERNAL_HOLDOUT | 166 | SEALED |

Source composition of the usable set: Objaverse 3702, Quaternius 191, KayKit 37.

## IRIS system boundary

```text
8 ordered RGBA views + known controlled yaw
  -> shared multiscale image encoder
  -> within-view axial reasoning
  -> row/geometry-constrained cross-view fusion
  -> dense decoder
  -> learned P / geometric N / predictive U / coarse+fine persistence Z
  -> deterministic top-k / reciprocal / cycle / local rerank
  -> set-valued ambiguity retention
  -> deterministic support, geometry fusion, reprojection, provenance
```

The **problem was pruned, not the observable evidence**. Evidence is richer than the earlier identity-centric M4 formulation: common-frame P, geometric N, observation support, predictive risk, persistence evidence, set-valued ambiguity and provenance are retained. What is removed from IRIS is authored hidden mechanical identity: joint IDs, owner IDs, parents, skinning and GFDR mechanics are not frontend truth.

## Exact truth apparatus

Training truth is built from the legal observation authority:

`pixel -> triangle_id + nvdiffrast barycentric_uv -> continuous canonical surface locus P`.

Geometric N is reconstructed deterministically from canonical geometry. Cross-view tracks are accepted only when the same physical P is found in another view within a bounded raster-neighborhood + surface-error check. No teacher bone/owner identity is used to create IRIS correspondence truth.

## Preflight already passed

A real FIT asset (`asset_00027381105293114a49dc90`) was exercised end-to-end:

- 8/8 real raster/camera authorities loaded;
- 128 exact cross-view surface tracks built;
- support: 122 tracks visible in 4 views, 6 in 5 views;
- neural core parameters: 5,657,863;
- P/N/U/Z forward shapes valid;
- geometry + correspondence loss finite;
- backward produced nonzero gradient.

A representation-ceiling smoke on the same real asset produced exact P and exact P+N top1/top8 = 1.0 across 390 view-pairs. The mandatory pre-optimizer gate runs on 64 open assets after cache preparation; the one-asset result is apparatus validation, not the confirmatory corpus result.

## Training contract

Default controlled baseline:

- model input: 256×256 derived on the fly from frozen 512 observations; normalized geometry coordinates are resolution-independent;
- 8 ordered views;
- 24 epochs;
- batch 1;
- AdamW, lr 5e-5, weight decay 1e-4;
- grad clip 2.0;
- correspondence warm-up: first 4 epochs geometry-only, then Z + P-consistency terms;
- FIT trains; TUNE selects;
- CAL/DEV/EXTERNAL remain sealed until explicit authorization.

The 512 observation cache is preserved so a later 512 qualification/fine-tune does not require rebuilding truth.

## Colab execution

The package is designed to live at:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1`

Then:

```bash
python launch_iris_controlled_v1.py --mode prepare
python launch_iris_controlled_v1.py --mode ceiling
python launch_iris_controlled_v1.py --mode train
```

Or, after package/Drive verification:

```bash
python launch_iris_controlled_v1.py --mode all
```

`--mode all` still opens **only FIT/TUNE**. Sealed evaluation is a separate explicit action.

## Authority order

1. Drive `IRIS_CONTROLLED_V1_CORPUS_FREEZE.json`
2. Drive `IRIS_CONTROLLED_V1_SPLIT_FREEZE.json`
3. Drive `IRIS_CONTROLLED_V1_DEFERRED_REPAIR_SHA_REGISTRY.json`
4. `IRIS_CONTROLLED_V1_PREREG.md`
5. `AUTHORITY_POINTERS_V1.json`
6. `PACKAGE_MANIFEST_V1.json`
7. executable source files

Nothing excluded is deleted. Re-entry requires a new versioned corpus freeze.
