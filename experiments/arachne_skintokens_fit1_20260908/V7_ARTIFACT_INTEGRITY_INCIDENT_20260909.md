# V7 artifact integrity incident — 2026-09-09

Scope: `RealSaS.Arachne.SkinFieldCodec.v7` Mage A0 FIT1 run only.

## Observed facts

- Sealed result JSON records the completed V7 run at optimizer step 192 with run fingerprint `18ed583141112c65352d903a97452691cfac5f64383a34ad2a7f0c57ff4abcc6`.
- The only persisted Drive progress artifact has the same run fingerprint, config hash, cache/binding hashes, and parameter count, but its embedded progress `step` is 16.
- Re-decoding that persisted model state reproduces the step-16 V7 metrics (`raw row-L1 mean ~= 1.835025`, p95 ~= `1.902659`), not the sealed step-192 metrics (`mean ~= 1.464178`, p95 ~= `1.841071`). Therefore the Drive `.pt` is genuinely a stale step-16 model, not merely stale wrapper metadata.
- The original V7 notebook's `save_progress()` serializes model + AdamW state + scheduler + Python/NumPy/Torch/CUDA RNG state. The progress payload is about 3.3 GB.
- The runner attempted to overwrite the same Drive-mounted progress path every 16 optimizer steps using local `torch.save(tmp)` followed by `os.replace(tmp, path)`. The small final JSON persisted at step 192, while the cloud-visible multi-GB progress artifact remained at step 16.

## Interpretation

This is an **artifact persistence failure**, not a scientific-training failure and not evidence that V7 stopped at step 16. The scientific result remains the sealed step-192 JSON because it was produced from the in-memory step-192 model. However, the exact final model state was not durably persisted by the original run.

Most plausible mechanism: repeated multi-gigabyte overwrite/rename semantics on the Google Drive FUSE mount did not reliably commit later versions to cloud storage. Treat this as infrastructure behavior until independently reproduced; do not assume `os.replace()` on a mounted Drive path gives durable atomic cloud replacement for multi-GB checkpoints.

## Recovery contract

The step-16 checkpoint is sufficient for exact runner-style recovery because it contains:
- model state,
- AdamW state,
- cosine scheduler state,
- Python RNG,
- NumPy RNG,
- Torch CPU RNG,
- CUDA RNG state.

Recovery must use the original V7 `load_progress()` contract, resume steps 17..192, and then fail closed unless the replayed final model reproduces the sealed result metrics within preregistered tolerances. Only after that proof may row/objective autopsy outputs be treated as belonging to the original V7 final state.

## Persistence policy change

Do not repeatedly overwrite a multi-GB full optimizer checkpoint on Google Drive for short FIT1 runs. For recovery/autopsy:
1. leave the original result/progress artifacts untouched;
2. replay from the stale but valid resume state;
3. after sealed-result agreement, persist **one** model-only FP32 final artifact;
4. stage it on local Colab storage first, copy to Drive once, and verify SHA-256 after the Drive copy;
5. store diagnostic JSON/CSV separately.

Future production runners should separate lightweight frequent resume state from durable final model artifacts, or use a storage method whose multi-GB replacement semantics are explicitly verified.
