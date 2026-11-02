"""Canonical TESSA model namespace.

TESSA = Topological Evidence for Surface Structure Approximation.

IRIS provides observation-grounded surface evidence; GSA carries the shared
mechanical surface substrate; TESSA emits learned topology/mesh proposals.
TESSA proposals are never canonical by themselves. Compiler qualification,
repair, adoption, and product authority remain mandatory.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .v1 import *  # noqa: F401,F403


def __getattr__(name):
    # Static truth/materialization courts must not initialize the neural runtime.
    # Preserve the existing V1 public namespace, including star-import __all__.
    implementation = import_module('.v1', __name__)
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(dir(import_module('.v1', __name__))))
