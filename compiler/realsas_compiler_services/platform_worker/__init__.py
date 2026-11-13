"""Temporal Engine worker for the RealSaS scientific compiler."""

# Original EngineRelease import IDs continue to resolve after physical migration.
from pathlib import Path as _Path
_RSS_VNEXT_WORKERS = _Path(__file__).resolve().parents[2] / "rss_vnext" / "workers"
if _RSS_VNEXT_WORKERS.is_dir():
    __path__.append(str(_RSS_VNEXT_WORKERS))
