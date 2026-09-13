from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import Json, load_proof


def proof_file_present(root: Path) -> bool:
    return any(
        path.is_file()
        for path in (
            root / "proof" / "product_proof_bundle_ir.json",
            root / "product_proof_bundle_ir.json",
        )
    )


def load_optional_proof(root: Path, product: Mapping[str, Any]) -> Json | None:
    """Load and validate proof when present; absence means static inspection only.

    This helper does not fabricate an ABSTAIN proof. A present proof still flows
    through the strict lineage/schema checks in common.load_proof. Runtime/export
    callers remain responsible for requiring an actual current PASS proof.
    """
    if not proof_file_present(root):
        return None
    return load_proof(root, product)


def proof_status(proof: Mapping[str, Any] | None) -> str:
    if proof is None:
        return "UNAVAILABLE"
    return str(proof.get("overall_status") or "UNKNOWN")
