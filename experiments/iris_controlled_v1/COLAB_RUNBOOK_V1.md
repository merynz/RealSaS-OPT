# IRIS Controlled V1 — Colab Runbook

## Runtime

Use an NVIDIA GPU runtime for training. Cache preparation and representation ceiling are CPU-capable, but `--mode all` proceeds into CUDA training.

## Expected Drive locations

- corpus root: `/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3`
- package: `.../reports/iris_controlled_v1`
- cache: `.../cache/IRIS_CONTROLLED_V1`
- run outputs: `.../runs/IRIS_CONTROLLED_V1`

## Safe sequence

```bash
cd /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_controlled_v1
python launch_iris_controlled_v1.py --mode prepare
python launch_iris_controlled_v1.py --mode ceiling
python launch_iris_controlled_v1.py --mode train
```

Resume an interrupted training run:

```bash
python launch_iris_controlled_v1.py --mode train --resume
```

Do not pass sealed split names to preparation. The standard launcher never opens them.

## What `prepare` does

- SHA-checks the package;
- verifies canonical selection SHA;
- reconstructs the exact 3,930-record seed from the split freeze;
- caches FIT+TUNE images and exact P/N samples;
- builds physical-locus multi-view tracks from raster authority;
- writes per-cache SHA/provenance manifest and seal.

## What `ceiling` does

Runs the quick exact P/P+N information ceiling on a deterministic 64-asset open subset. No optimizer.

## What `train` does

Requires the ceiling PASS, then trains only FIT and selects on TUNE. It writes `RUN_CONTRACT.json`, `last.pt`, `best.pt`, and `history.json`.
