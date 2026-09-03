"""Compatibility facade for the canonical substrate local-geometry implementation.

Canonical source: realsas_compiler_core.substrate.local_geometry
"""
from .substrate import local_geometry as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
del _impl, _name
