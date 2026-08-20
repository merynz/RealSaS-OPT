# R6.3 Execution-Isolation Note

R6.3 is an execution-only supersession of R6.2 for N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820. It changes no frozen scientific source, checkpoint, corpus, population, objective, metric, threshold, witness rule, or decision semantics.

Observed blocker: the frozen R6.2 baseline process can enter extreme runtime pathology when multiple model contexts occur sequentially in one OS process. Family 15290/e01 reproduced this: direct frozen V8 completed in ~27.7 s, frozen gate model context in ~1.7 s, and frozen seed_predict in a fresh process completed normally, while the combined R6.2 baseline process exceeded 10 minutes without an artifact.

R6.3 therefore enforces four explicit OS-process stages per family:

1. frozen V8 generation;
2. frozen gate computation;
3. frozen final-baseline selection / frozen seed_predict if and only if the unchanged frozen gate switches;
4. frozen observable-state generation using the exact final baseline bytes.

Every stage rechecks frozen source-member SHA-256 values, the frozen checkpoint SHA, raster hashes, and truth_access=NONE. Sidecar semantics remain closed.

Admission parity before freeze:
- 09908/e01 V8 route: baseline NPZ/JSON and full observable-state NPZ/JSON byte-exact versus R6.2.
- 12772/e01 seed route: final baseline NPZ/JSON byte-exact versus R6.2.
- 13203/e01 seed route: final baseline NPZ/JSON byte-exact versus R6.2.

R6.2 partial panel outputs are parity/diagnostic evidence only after R6.3 is frozen. Canonical Phase-1 requires all eight families to be regenerated under R6.3 before any evaluator-side sidecar semantic access.
