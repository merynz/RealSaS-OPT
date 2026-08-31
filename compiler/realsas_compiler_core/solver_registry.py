from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable, Any


class SolverNotPromotedError(RuntimeError):
    """Raised when code asks current main to execute a historical-only solver."""


@dataclass(frozen=True)
class SolverAuthorityRecord:
    solver_id: str
    historical_impl: str
    current_main_status: str
    numerical_reference: str
    promotion_rule: str
    module_name: str
    callable_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# This registry is authority/provenance metadata, not an executable side door.
# The historical modules named below are deliberately outside the current GitHub
# execution closure. They may become executable only after a separately reviewed
# typed restoration/promotion gate adds the exact implementation bytes to current
# main and changes current_main_status accordingly.
SOLVER_AUTHORITY: dict[str, SolverAuthorityRecord] = {
    "CDT": SolverAuthorityRecord(
        "CDT",
        "v0.5/realsas_mesh.cdt_production",
        "HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL",
        "v97_43 exact-predicate CDT + quality seal",
        "promote only through provenance + typed parity + topology/residual + downstream non-inferiority gate",
        "realsas_mesh.cdt_production",
        "triangulate_production_cdt",
    ),
    "BBW": SolverAuthorityRecord(
        "BBW",
        "v0.5/realsas_weight.production_bbw",
        "HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL",
        "v97_43 bound-active QP + coupled simplex/ADMM/KKT + sparse LDLT/Schur + dual-feasibility audit",
        "normal learned path may not silently synthesize semantic skin; promote only as bounded qualifier or separately typed proposal/fallback after causal need",
        "realsas_weight.production_bbw",
        "solve_bbw_kkt_weights",
    ),
    "ARAP": SolverAuthorityRecord(
        "ARAP",
        "v0.5/realsas_deformation.production_arap",
        "HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL",
        "v97_43 sparse ARAP runtime numerics",
        "promote only against exact CanonicalPuppetGraph.v2 lineage after residual/deformation parity",
        "realsas_deformation.production_arap",
        "solve_production_arap",
    ),
    "XPBD": SolverAuthorityRecord(
        "XPBD",
        "v0.5/realsas_deformation.secondary_xpbd",
        "HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL",
        "v97_43 implicit/graph XPBD + contact binding",
        "promote only against exact product state after constraint/contact residual parity",
        "realsas_deformation.secondary_xpbd",
        "solve_xpbd",
    ),
}

# Backward-compatible name retained deliberately. Empty means exactly what it says:
# a clean checkout of current main exposes no promoted heavy numerical solver.
EXECUTABLE_SOLVERS: dict[str, Callable[..., Any]] = {}


def resolve_promoted_solver(solver_id: str) -> Callable[..., Any]:
    """Resolve a solver only after its authority record is explicitly promoted.

    Merely having historical source provenance is never enough to import or run it.
    This prevents solver_registry from turning Drive-only authority into an implicit
    second execution path while still preserving a narrow future promotion seam.
    """
    try:
        record = SOLVER_AUTHORITY[solver_id]
    except KeyError as exc:
        raise KeyError(f"unknown solver_id:{solver_id}") from exc

    if record.current_main_status != "CANONICAL_MAINLINE_EXECUTABLE":
        raise SolverNotPromotedError(
            f"{solver_id} is {record.current_main_status}; exact typed restoration/promotion is required before execution"
        )

    module = import_module(record.module_name)
    solver = getattr(module, record.callable_name)
    if not callable(solver):
        raise TypeError(f"promoted solver target is not callable:{record.module_name}.{record.callable_name}")
    return solver
