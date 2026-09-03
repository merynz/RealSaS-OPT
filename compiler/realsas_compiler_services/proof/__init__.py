"""Proof-side diagnostic services promoted behind the current compiler authority."""

from .failure_signatures import derive_failure_signatures, no_owner_attribution

__all__ = ["derive_failure_signatures", "no_owner_attribution"]
