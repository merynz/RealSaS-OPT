# MapAnything Apache -> RealSaS Orthographic IRIS

This branch is a clean, isolated integration of the **Apache-labelled MapAnything checkpoint** into the canonical RealSaS single-pose geometry problem. It does not replace or rewrite the historical G1/P-V5 baseline.

## What is reused vs changed

**Reused intact at runtime from pinned upstream:** DINO image encoder, MapAnything multi-view information sharing, DPT dense geometry head, confidence and ambiguity outputs. The upstream repository is cloned at an exact commit and installed editable; we do not copy DUSt3R/MASt3R product code into RealSaS.

**RealSaS specialization:** native 1024 RGBA detail path; raster + deterministic known-view rotation product forward; exact known 8-view orthographic camera; deterministic `P = O + dF`; deterministic normals from P; geometry risk; existing geometry-only target firewall. Upstream predicted cameras are ignored as authority.

## Pinned upstream

- repository: `facebookresearch/map-anything`
- code commit: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`
- model: `facebook/map-anything-apache`
- model revision: `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`
- published model-card license: Apache-2.0

See `UPSTREAM_PROVENANCE.json` for the shipping-license caveat and review boundary.

## Setup

```bash
bash scripts/bootstrap_upstream.sh
```

Do **not** install `map-anything[all]`: that optional bundle pulls external research models we do not need and may carry incompatible licenses. The base package is sufficient.

## Manifest

Training/evaluation uses an explicit JSON or JSONL manifest. One row:

```json
{"family_id": 16634, "split": "FIT", "pose_a_dir": "/.../PoseA", "target_npz": "/.../G1_TARGET.npz"}
```

Only `P_A, N_A, XY_A, V_A, direct_obs_A, family_id` are copied from a target container. Pose B, rig, owner, skinning, mechanics and deformation fields never cross the learner target boundary.

## 1. GPU preflight — zero optimizer steps

Run on the exact A100 environment that will train:

```bash
python -u preflight.py \
  --pose-a-dir /path/to/one/authorized/PoseA \
  --target-npz /path/to/its/G1_TARGET.npz \
  --expected-family-id 16634 \
  --stage geometry \
  --output /path/to/PREFLIGHT_GEOMETRY.json
```

Then separately test `--stage full`. The default training schedule contains full encoder unfreeze, so the JSON supplied to `train.py` **must be a PASS from `--stage full`**. A 40 GB A100 may or may not fit full encoder backward for the current ~1B upstream checkpoint. OOM is a fail-closed memory result, not permission to silently change the model. If full does not fit, we make a reviewed config commit for a geometry-only schedule or move to a larger GPU; the training script will not silently downgrade.

## 2. Train

```bash
MANIFEST=/path/to/manifest.jsonl \
PREFLIGHT_JSON=/path/to/PREFLIGHT_FULL.json \
OUT=/path/to/run \
bash scripts/train_a100.sh
```

Default schedule is 12 epochs: 1 adapter warmup, 3 geometry-path epochs, 8 full-image-path epochs. If full preflight does not fit, change the schedule in one reviewed config commit rather than editing code ad hoc during the run.

## 3. Evaluate

```bash
python -u evaluate.py \
  --manifest /path/to/manifest.jsonl \
  --split TUNE \
  --checkpoint /path/to/run/FINAL_CHECKPOINT.pt \
  --output /path/to/EVAL.json
```

For the preregistered matched decision rule, additionally pass a P-V5 metrics JSON with the same family rows via `--baseline-json`.

## Why this is not a toy trial

- The pretrained MapAnything geometry stack is actually in the differentiable training graph; `MapAnything.infer()` is never used for training because upstream marks it inference-only.
- Native 1024 pixels are consumed by a separate detail hierarchy rather than throwing away the product-resolution signal.
- The final geometry is not a generic perspective pointmap: every predicted point is compiled onto the exact RealSaS known orthographic ray.
- Common-frame P, exact forward depth, signed local geometry, uncertainty and same-surface multi-view consistency are all supervised together.
- The optimizer schedule ultimately opens the complete image-only geometry path of the pretrained backbone, subject to an explicit GPU-memory preflight.
- All run artifacts retain upstream model/code revisions and optimizer-step provenance.

Read `PREREG_V1.md` before authorizing optimizer steps.
