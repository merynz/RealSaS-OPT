# Fast Probe Execution Policy V2

**Date:** 2026-10-26  
**Status:** ACTIVE RESEARCH OPERATIONS POLICY  
**Authority:** execution policy only; no scientific or product authority.

## Rule

Synthetic, analytic and bounded micro-learning architecture courts may run as long as their preregistered workload requires.

Runtime is **not** a scientific selection criterion and is not a reason to reduce seeds, controls, held-out cases, optimization steps, or other preregistered validity requirements.

## Timeout policy

For newly authored fast-probe GitHub Actions jobs:

- use `timeout-minutes: 360` as a hung/runaway safety guard;
- do not use 30-minute or similarly tight cutoffs for scientific work;
- a shorter timeout is allowed only when the probe itself has a justified deterministic upper bound far below that timeout;
- if a run is healthy but long, let it complete;
- optimize implementation, batching, vectorization, caching and deterministic reuse before reducing scientific workload.

A timeout/cancellation is an **execution failure**, never a negative scientific result.

## Existing in-flight jobs

Do not rewrite an active workflow merely to change its timeout if doing so would trigger a duplicate run and discard useful compute. Let the in-flight attempt finish; if it hits its old guard, rerun the same sealed court under this policy without changing scientific constants.

## Reporting

Every result records exact workflow/run identity, scientific constants, report artifact, and whether any timeout or infrastructure failure occurred.
