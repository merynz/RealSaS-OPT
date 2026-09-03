"""Compatibility facade for canonical mesh.mesh_binding implementation."""
from .mesh import mesh_binding as _impl
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)
__all__ = [name for name in dir(_impl) if not name.startswith("_")]
del _impl, _name
