# Observable Functional Audit V2 — R6.1 build-recipe correction

Status: `PRE_FREEZE_BUILD_RECIPE_ONLY__NO_TRUTH_OPEN__NO_SCIENTIFIC_CHANGE`

The persisted R6 `BUILD_R6_SOURCE.py` correctly pins the R4 evaluator byte SHA-256 but its `LOAD_ANCHOR` assumed a three-line `spec.loader` block. The pinned R4 evaluator stores the same operation on one line:

`mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod`

Therefore the R6 builder failed closed before producing an authority source tree. No evaluator sidecar semantics were opened and no result was produced.

R6.1 changes only the textual patch anchor/replacement so it matches the already-pinned R4 evaluator bytes. The intended Python 3.13 compatibility change is unchanged: register the dynamically created module in `sys.modules[name]` immediately before `spec.loader.exec_module(mod)`.

Scientific contract, population policy, metrics, thresholds, truth-use rules, observable producer, checkpoint, GFDR-V2 source, and decision tree are unchanged. R6.1 remains a clean replication revision and must use untouched `e01`, never the stale local e00 file named `INPUT_MANIFEST_R6.json`.

Pre-freeze local checks after deterministic rebuild:
- base evaluator SHA guard: PASS (`3d39b0afc0213986dad02dc8965df3189d205076219a4016c4e04d9bd753fabc`)
- static truth-firewall test: PASS
- Python 3.13 evaluator import smoke: PASS
- decorator replay on frozen e00 apparatus only (non-scientific smoke): PASS
- resulting evaluator SHA-256: `a986dba818ab63cbd6beac133c5e713698489ca037d5eeba71329e6b216828e3`

No R6.1 scientific run is authorized until the exact e01 input-byte manifest and deterministic R6.1 source bundle are persisted and read back.