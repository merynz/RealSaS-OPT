# Observable Functional Audit V2 — R5 Canonicality Invalidation

**Status:** `DIAGNOSTIC_ONLY__SOURCE_AUTHORITY_MISMATCH`

R5 produced a machine result, but final provenance audit found that the source package identity recorded in `SOURCE_MANIFEST_R5.json` does not match the source package/executable bytes present in the run environment and persistent source mirror.

The discrepancy was discovered before state promotion and before the R5 result was declared canonical.

Consequences:

- R5 result is retained as diagnostic evidence only.
- R5 must not be used as a canonical decision or as an information-theoretic impossibility claim.
- The source manifest must not be retroactively edited to make R5 canonical.
- No R5 metric threshold, population rule, truth-use rule, or aggregate decision threshold may be changed based on the R5 outcome.

The correct recovery is a new replication revision using exact source bytes frozen before execution on an audit panel whose evaluator semantics have not been opened by this experiment. R6 therefore inherits the R4/R5 scientific contract unchanged and uses an untouched e01 replication panel.
