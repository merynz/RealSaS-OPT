from __future__ import annotations

"""Canonical Geppetto R6 training entrypoint.

This module intentionally contains no family-specific optimizer schedule. It exposes
only the generic source-complete train step; real fitting remains gated elsewhere.
"""

from .geppetto_train_v2 import geppetto_train_step_v2

__all__ = ["geppetto_train_step_v2"]
