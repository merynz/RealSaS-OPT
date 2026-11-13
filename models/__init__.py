"""RealSaS product-facing model package.

Only `models.rss_vnext` is the current product ModelRelease namespace.
Historical Geppetto/Arachne/SkinFieldCodec import identities remain accessible
to old checkpoints and compiler compatibility code without exposing three
separate model directories in the GitHub models/ root.

This is an import-compatibility search path, not a second model registry.
"""
from pathlib import Path as _Path

_HISTORICAL_MODELS = (
    _Path(__file__).resolve().parent.parent / "historical" / "model_sources"
)
if _HISTORICAL_MODELS.is_dir():
    __path__.append(str(_HISTORICAL_MODELS))
