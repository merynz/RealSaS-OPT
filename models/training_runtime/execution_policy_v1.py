"""Choose among real measured executions of the same sealed learning recipe."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MeasuredExecution:
    name: str
    recipe_sha256: str
    corpus_sha256: str
    model_sha256: str
    hardware_identity: str
    real_corpus: bool
    update_equivalence_pass: bool
    covered_cost_strata: tuple[str, ...]
    update_p95_seconds: float
    evaluation_p95_seconds: float
    checkpoint_p95_seconds: float
    cold_setup_seconds: float
    peak_reserved_bytes: int
    device_memory_bytes: int
    updates: int
    evaluations: int
    checkpoints: int

    @property
    def total_seconds(self):
        return (self.cold_setup_seconds+self.updates*self.update_p95_seconds
                + self.evaluations*self.evaluation_p95_seconds+self.checkpoints*self.checkpoint_p95_seconds)


def select_execution(candidates, *, required_cost_strata, max_total_seconds, memory_fraction=.85):
    if not candidates or not math.isfinite(max_total_seconds) or max_total_seconds <= 0 or not 0 < memory_fraction < 1:
        raise ValueError("EXECUTION_BUDGET_INVALID")
    identities = {(c.recipe_sha256, c.corpus_sha256, c.model_sha256, c.hardware_identity,
                   c.updates, c.evaluations, c.checkpoints) for c in candidates}
    if len(identities) != 1:
        raise ValueError("EXECUTION_CANDIDATES_CHANGE_SCIENCE_OR_HARDWARE")
    accepted, rejected = [], {}
    for c in candidates:
        reasons = []
        timings = (c.update_p95_seconds, c.evaluation_p95_seconds, c.checkpoint_p95_seconds, c.cold_setup_seconds)
        if (not c.name or not all(math.isfinite(t) and t >= 0 for t in timings)
                or c.update_p95_seconds <= 0 or min(c.updates, c.checkpoints, c.evaluations) <= 0
                or min(c.peak_reserved_bytes, c.device_memory_bytes) <= 0):
            reasons.append("INVALID_MEASUREMENT")
        if not c.real_corpus:
            reasons.append("REAL_CORPUS_REQUIRED")
        if not c.update_equivalence_pass:
            reasons.append("UPDATE_EQUIVALENCE_REQUIRED")
        if not set(required_cost_strata).issubset(c.covered_cost_strata):
            reasons.append("COST_STRATA_MISSING")
        if c.peak_reserved_bytes > c.device_memory_bytes*memory_fraction:
            reasons.append("MEMORY_HEADROOM_MISSING")
        if c.total_seconds > max_total_seconds:
            reasons.append("TOTAL_WALL_BUDGET_EXCEEDED")
        if reasons:
            rejected[c.name] = reasons
        else:
            accepted.append(c)
    if not accepted:
        raise ValueError(f"NO_QUALIFIED_EXECUTION:{rejected}")
    return min(accepted, key=lambda c: (c.total_seconds, c.name)), rejected
