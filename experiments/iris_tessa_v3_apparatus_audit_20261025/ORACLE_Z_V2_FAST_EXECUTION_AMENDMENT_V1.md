# Oracle-Z Representation Court V2 — Fast Execution Amendment V1

**Date:** 2026-10-26  
**Status:** `SEALED_BEFORE_FIRST_V2_SCIENTIFIC_EXECUTION`  
**Applies to:** `ORACLE_Z_REPRESENTATION_PREREG_V2.md`

## Reason

The research-operations policy has been simplified for synthetic / analytic / bounded micro-learning architecture courts. These courts do not create product/model authority and should optimize iteration speed while preserving scientific reproducibility.

The original V2 preregistration required Platform `Experiment -> Attempt -> Artifact/CAS` execution. That operational requirement is superseded for this court only.

## What changes

Scientific Oracle-Z V2 may execute as one sealed GitHub Actions workflow on the audited research branch / exact commit SHA. The canonical scientific evidence is:

1. the immutable preregistration plus this amendment;
2. exact probe source at the executed SHA;
3. GitHub workflow run identity and exact code SHA;
4. uploaded JSON report artifact;
5. the resulting interpretation recorded in `RESEARCH_DECISION_LEDGER_V2.md`.

A workflow green check means execution completed; it is **not** the scientific verdict. The verdict comes only from the preregistered validity/selection rules applied to the emitted report.

## What does not change

No scientific constant, treatment arm, seed, truth family, optimization budget, metric, validity gate, threshold, interpretation rule or non-claim from `ORACLE_Z_REPRESENTATION_PREREG_V2.md` changes.

In particular the fixed calibration/comparison seeds, learning-rate candidates, step counts, carrier budgets, train/eval pools, surface/frequency ladders, firewall, and all selection thresholds remain sealed.

## Operational boundary

Fast GitHub scientific execution is authorized for architecture-search probes that are synthetic/analytic/bounded and do not depend on real training datasets or checkpoint/product authority.

Platform `Experiment -> Attempt -> Artifact/CAS` remains the preferred authority once work crosses into real dataset/checkpoint/training/expensive-compute or product/model-authority experiments. PR #218 is intentionally deferred for that phase.

## Legacy compatibility note

`oracle_z_representation_probe_v2.py` was sealed while Platform execution was expected and therefore internally requests four `REALSAS_*` provenance fields and labels its raw mode `PLATFORM_SCIENTIFIC`. The GitHub wrapper may supply run-scoped compatibility provenance to this unchanged scientific core, then relabel only execution metadata in the outer report. It must not alter scientific arrays, constants, metrics, diagnostics or thresholds.
