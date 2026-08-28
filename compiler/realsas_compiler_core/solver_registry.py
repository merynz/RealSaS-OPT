from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Any
from realsas_mesh.cdt_production import triangulate_production_cdt
from realsas_weight.production_bbw import solve_bbw_kkt_weights
from realsas_deformation.production_arap import solve_production_arap
from realsas_deformation.secondary_xpbd import solve_xpbd

@dataclass(frozen=True)
class SolverAuthorityRecord:
    solver_id:str
    executable_impl:str
    executable_status:str
    numerical_reference:str
    promotion_rule:str
    def to_dict(self): return asdict(self)

SOLVER_AUTHORITY={
 "CDT":SolverAuthorityRecord("CDT","v0.5/realsas_mesh.cdt_production","EXECUTABLE_BASELINE","v97_43 exact-predicate CDT + quality seal","v0.5 remains executable until parity/source-diff proves current replacement >= late-May authority"),
 "BBW":SolverAuthorityRecord("BBW","v0.5/realsas_weight.production_bbw","EXECUTABLE_BASELINE_NOT_FINAL_NUMERICAL_AUTHORITY","v97_43 bound-active QP + coupled simplex/ADMM/KKT + sparse LDLT/Schur + dual-feasibility audit","do not claim final production BBW from the simpler Python port until parity is demonstrated"),
 "ARAP":SolverAuthorityRecord("ARAP","v0.5/realsas_deformation.production_arap","EXECUTABLE_BASELINE","v97_43 sparse ARAP runtime numerics","promote only after residual/deformation parity"),
 "XPBD":SolverAuthorityRecord("XPBD","v0.5/realsas_deformation.secondary_xpbd","EXECUTABLE_BASELINE","v97_43 implicit/graph XPBD + contact binding","promote only after constraint/contact residual parity"),
}

EXECUTABLE_SOLVERS={"CDT":triangulate_production_cdt,"BBW":solve_bbw_kkt_weights,"ARAP":solve_production_arap,"XPBD":solve_xpbd}
