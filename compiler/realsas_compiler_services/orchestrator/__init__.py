"""RealSaS orchestration package.

The command module is intentionally lazy-loaded so `python -m ...orchestrator.mainline`
executes exactly once instead of importing mainline during package initialization.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["execute", "status_text", "validate_ledger", "validate_plan"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(
            "compiler.realsas_compiler_services.orchestrator.mainline"
        )
        return getattr(module, name)
    raise AttributeError(name)
