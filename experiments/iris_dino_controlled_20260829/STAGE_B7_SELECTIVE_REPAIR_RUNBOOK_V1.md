# Stage-B7 Selective Repair Publication — Colab Runbook V1

**Status:** `STAGING_ONLY__FAIL_CLOSED__NO_DINO_OPTIMIZER_STEPS`

## Purpose

Re-render only the exact nine repaired members of the recovered historical train-512 population from frozen Stage-B6 `normalized.npz`, while preserving the old per-asset palette and production native-1024 render contract.

The first action is an unchanged historical sentinel re-render. If sentinel parity is not exact, the script exits before rendering any repair asset.

## Required runtime

Use a CUDA Colab runtime. This stage requires nvdiffrast but performs no learning and no optimizer step.

Mount Drive, then install the pinned preflight rasterizer revision:

```bash
pip install -q 'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
```

The revision is not accepted merely because installation succeeds; the historical sentinel must reproduce exactly.

## Canonical Drive paths

Corpus:

```text
/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3
```

Frozen Stage-B6 repair assets:

```text
/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/post_corpus_audit/B6_REPAIR_STAGING_V1
```

Stage-B7 output is separate staging only:

```text
/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_REPAIR_STAGING_V1
```

## Runner source

Use the branch-sealed file:

```text
experiments/iris_dino_controlled_20260829/stage_b7_selective_repair_renderer_v1.py
```

The canonical preflight run MUST use the exact Git blob committed on `dino-zero-step-preflight-20260829`; do not paste-edit the renderer in Colab.

## Canonical command

From a checked-out RealSaS-OPT branch containing the sealed runner:

```bash
python experiments/iris_dino_controlled_20260829/stage_b7_selective_repair_renderer_v1.py \
  --corpus-root /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3 \
  --repair-assets-root /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/post_corpus_audit/B6_REPAIR_STAGING_V1 \
  --staging-root /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_REPAIR_STAGING_V1
```

Default sentinel:

```text
asset_d7d4192f9dac8b146a17bc41
```

It is an unchanged historical production asset, not one of the nine repairs.

## Sentinel hard gate

For all eight sentinel views require:

- visible pixel index array exact;
- triangle-id array exact;
- max absolute barycentric difference <= `5e-5`;
- native `cel_clean.png` byte-domain pixel array exact;
- native `ink_cel.png` pixel array exact;
- `cel_clean_512.png` exact;
- `ink_cel_512.png` exact.

If any view fails, status is:

```text
FAIL_SENTINEL_PARITY__NO_REPAIR_ASSET_RENDERED
```

and the nine-asset staging route remains unopened. Do not loosen the gate after seeing the result; version the renderer contract instead.

## Nine repair assets

The runner is hard-coded to exactly:

```text
asset_1d6d3b17fe506463b8840c51
asset_4970fb8cc7c69df071970dc0
asset_66c8c63c7a97e830b89d0ba8
asset_674fb6e5571ca86dd0e5c858
asset_681fb76277c10d2307f7509f
asset_88fe0971a994597f3866473b
asset_a873bb17b9a66e6e980845a5
asset_ce3577373d7cc71b7025cc74
asset_f56aff9ecf13f7ddf0afcd55
```

For each it verifies Stage-B6 `repair_record.json -> normalized_sha256` before rendering.

## Output

On PASS, staging contains:

```text
SENTINEL_PARITY.json
STAGE_B7_RESULT.json
staged_assets/<asset>/primary_geometry.npz
staged_assets/<asset>/RENDER_COMPLETE.json
staged_assets/<asset>/STAGE_B6_REPAIR_RECORD.json
staged_assets/<asset>/renders/V0..V7/{camera.json,raster_authority.npz,cel_clean.png,ink_cel.png,cel_clean_512.png,ink_cel_512.png}
```

Expected terminal status:

```text
PASS_STAGING_ONLY__MASTER_UNCHANGED
```

with:

- sentinel_pass = true
- staged_asset_count = 9
- pass_asset_count = 9
- master_mutated = false.

## Publication firewall

This runbook intentionally does **not** replace anything under `master/assets`.

Only after the staging artifact is reviewed and the 9/9 result is sealed may a separate atomic promotion step be authored. After promotion, the full exact train-512 availability audit must read 512/512 before DINO feature extraction is opened.
