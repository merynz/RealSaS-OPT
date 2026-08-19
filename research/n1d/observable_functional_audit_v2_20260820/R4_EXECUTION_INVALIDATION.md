# Observable Functional Audit V2 — R4 Execution Invalidation

**Status:** `INVALID_EXECUTION_COMPATIBILITY__NO_SCIENTIFIC_RESULT`

R4 passed source/input/prereg/observable-state freeze and reached the authorized `TRUTH_OPEN` boundary, but the evaluator process failed during module import before `main()` began and before any evaluator sidecar was parsed.

Failure:

- Python 3.13 `dataclasses` required the dynamically loaded `realsas_gfdr_v2` module to be present in `sys.modules` during `exec_module`.
- R4 `evaluation_phase._load()` created the module object but did not register it in `sys.modules` before execution.
- Import failed at the first `@dataclass` declaration with `AttributeError: 'NoneType' object has no attribute '__dict__'`.

Guards:

- `RESULT_R4.json`: ABSENT.
- sidecar semantic parsing: NOT REACHED.
- scientific thresholds/decision tree: NOT EVALUATED.
- no scientific inference may be drawn from this failed execution.

R4 source remains immutable historical evidence. The compatibility fix must be a new source revision. R5 may preserve the already-frozen scientific question, thresholds and decision tree because no result or sidecar semantic value was observed; nevertheless R5 requires its own source hash, tests, persistence gate and preregistration before execution.
