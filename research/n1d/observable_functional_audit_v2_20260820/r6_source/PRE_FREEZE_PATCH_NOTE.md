# R6 pre-freeze source note

`R6_EVALUATOR_IMPORT_COMPAT.patch` was an initial hand-written draft and is **not authoritative**; its second hunk was malformed before any R6 source freeze, preregistration, truth-open, or execution.

R6 source authority is exclusively:

1. verified R4 base source commit `a36133e6d97d7f848ce7b8cddcb43b6067e23596`, root `research/n1d/observable_functional_audit_v2_20260820/r3`;
2. `BUILD_R6_SOURCE.py`, which verifies the R4 evaluator SHA and performs two exact anchor replacements plus adds the import-smoke test;
3. the deterministic R6 source ZIP and its three-store hash manifest produced from that build.

The non-authoritative draft patch must not be used by any runner or manifest.
