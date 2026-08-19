# N1D Observable Functional Audit V2 — Preregistration R5

**Experiment ID:** `N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820`  
**Source revision:** `R5`  
**Source bundle SHA-256:** `a14cb7f2ac425689b526415862050f09d7cb66a560bb17c1555fe82092f5cd40`  
**Input manifest SHA-256:** `3a795e02d8a4379bb976676488604d706fa4fa863c3c0cd27787103781fa34f`  
**Base scientific prereg SHA-256:** `b692311dd5a27ad22a13fa7b178b535fc41ca67532d07089ce071aaafefad29f` (R4)  
**Observable-state bundle SHA-256:** `a4f7f9ce1b4f6d7858be4932a055e7509391d3516e8ad396b50b4332147e443f`  
**Observable-state manifest SHA-256:** `ba487f0ec8af931abc76fd5fee5c36c4850dbb0ae2adb4bb07fa1d49087074d1`

## Binding rule

R5 incorporates the complete scientific contract, population, witness definition, metric definitions, frozen thresholds, aggregate decision tree, invalidation guards, and impossibility-claim prohibition of R4 preregistration **by content-addressed reference, without modification**.

There are no scientific threshold, geometry, GFDR, truth-use, candidate, population, or decision-tree changes in R5.

## Why R5 exists

R4 execution failed during evaluator module import on Python 3.13 before `main()` began and before any evaluator sidecar was parsed. `RESULT_R4.json` was never created. The failure was dynamic-import compatibility only: the GFDR module had to be registered in `sys.modules` before `exec_module` so that Python 3.13 `dataclasses` could resolve its module namespace.

R5 changes only:

1. `evaluation_phase._load()` registers the dynamically loaded module in `sys.modules` before executing it;
2. `tests/test_imports.py` is added and must pass before evaluator execution.

The R5 source manifest states `scientific_change_from_R4 = NONE__PYTHON_3_13_DYNAMIC_IMPORT_COMPATIBILITY_ONLY`.

## Observable-state reuse

The R4 observable state is reused rather than recomputed because all producer bytes and all observable inputs are identical in R5:

- `observable_phase.py` SHA-256 R4 = R5 = `6eea236647e04a5566e93733f35c927639cc6407d4f62db80793edfa7e97dadf`;
- `prepare_runtime.py` SHA-256 R4 = R5 = `a194d60123e314ee5a5281cee37be35b60cfcc318054d135f636ca5326b88783`;
- `mechanics_metrics.py` SHA-256 R4 = R5 = `0164d0f6519c38ba6b9fd0f271f6794e13973418a00cbb8b4b721f4b1b670511`;
- observable raster-set SHA-256 = `efad989dd8a1929e2a9f49e1e94d7e16eda3aea75ca455a1faa052722e349e2c`;
- checkpoint SHA-256 = `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`.

The reused state already passed 8/8 `truth_access=NONE`, 8/8 bit-exact `V_B` replay, and 8/8 `N_B` replay max-abs `<=1e-7` before any sidecar semantics were opened.

## R5 preflight at freeze

- source R5 deterministic ZIP: PASS;
- R5 source mirrors Drive + Library roundtrip SHA: PASS;
- immutable input manifest: PASS;
- 136/136 corpus files SHA verified: PASS;
- R4 observable-state reuse by byte identity: PASS;
- static tests: PASS;
- evaluator import smoke: PASS;
- decorator replay: PASS;
- `RESULT_R5.json`: ABSENT;
- evaluator sidecar semantic values observed before this freeze: NO.

## Scientific contract inherited unchanged from R4

The primary experiment remains: within the actual frozen raster-only N1D Hybrid V11 front door, compare baseline geometry to an evaluator-selected **existing feasible H candidate**, regenerate counterfactual `N_B` with the frozen normal head and `V_B` with the Pose-B raster visual hull, then compare downstream GFDR-V2 mechanics.

Teacher truth may select/evaluate the counterfactual after truth open. It may not create candidate XYZ, supply counterfactual normals/visibility, enter the forward mechanics path, train, retune, or open sealed21/external10.

Primary equivalence continues to exclude coordinate-tautological `F_delta` and `R_rel_B`; the exact frozen block thresholds and decision tree are those in base prereg SHA `b692311d...ad29f`.

No outcome of this audit by itself authorizes an information-theoretic impossibility claim.

## R5 invalidation guards

In addition to every R4 guard, R5 is invalid if:

- the R5 import-compatibility diff contains any semantic change beyond module registration and the import-smoke test;
- the reused observable-state hashes differ from the frozen R4 state hashes;
- any R4 threshold/decision rule is overridden rather than inherited exactly;
- evaluator execution begins before this R5 prereg is persisted in GitHub, Drive, and Library.
