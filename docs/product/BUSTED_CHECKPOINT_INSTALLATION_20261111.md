# BUSTED checkpoint installation boundary (2026-11-11)

Product API: `GET /api/checkpoints/status`.

Configure `BUSTED_CHECKPOINT_ROOT` and `BUSTED_CHECKPOINT_MANIFEST`. The manifest must be inside the root. No checkpoint loading or inference occurs in the API server.

```json
{
  "schema": "RealSaS.BustedCheckpointInstallation.v1",
  "models": {
    "IRIS": {"path": "iris.pt", "sha256": "<64 lowercase hex>", "architecture_id": "RealSaS.IRIS.<exact architecture>"},
    "TESSA": {"path": "tessa.pt", "sha256": "<64 lowercase hex>", "architecture_id": "RealSaS.TESSA.<exact architecture>"},
    "AXIS": {"path": "axis.pt", "sha256": "<64 lowercase hex>", "architecture_id": "RealSaS.AXIS.<exact architecture>"},
    "MIRA": {"path": "mira.pt", "sha256": "<64 lowercase hex>", "architecture_id": "RealSaS.MIRA.<exact architecture>"}
  }
}
```

The architecture field is a claimed family identity, **not** independently verified model architecture. Actual checkpoint loaders and Platform ModelRelease approval must verify the payload and architecture independently. SHA-256 proves exact bytes, **not** that a checkpoint is trained, qualified, safe, or usable.

The inspection is intentionally not an executable installer. `checkpoints_present=true` never sets `compile_available=true`. Runtime model promotion requires qualified releases, model->canonical evidence binding, Mechanics/Graphics/Motion/Proof and sealed ProductRevision; no checkpoint presence shortcut is allowed.

No original source, weights, CAS object, Drive corpus or legacy model is modified by this addition.
