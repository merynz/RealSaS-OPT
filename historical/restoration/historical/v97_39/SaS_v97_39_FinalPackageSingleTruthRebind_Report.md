# SaS v97.39 — Final Package Single-Truth Rebind

## Goal

Reduce the repeated split-path/split-truth problem before asking the agent for another compile/test pass. The target is not R6 KKT quality yet; this patch makes R10 measure the final `SaSRealCompiledCharacterPackageAsset` instead of only its temporary proof-time package.

## What changed

- `SaSRealProofHarness` now exposes `RefreshCanonicalGraphFromPackage(report, package)`.
- R10 canonical graph stamping is centralized through `ApplyCanonicalGraphToReport(...)`.
- Duplicate, unused R1/R8 normalization methods inside R10 were removed; R10 now consumes only `SaSRealCompilerEvidenceNormalizer` for R1/R8 evidence rules.
- `SaSUnifiedRealCharacterBuildPipeline` now builds the final package, refreshes R10 from that exact package, rewrites proof artifacts after final package sync, rebuilds package canonical graph, then saves the package.
- R10 proof artifacts are no longer written from the pre-package proof path inside `RunR10`; unified builds write them after final package rebind.

## Why this matters

Previous patches made R1/R8 evidence recognition more consistent, but R10 and the final package could still carry different canonical graph snapshots. That is the same old class of bug: asset A says evidence exists, artifact B says it is missing. v97.39 makes the report and final package share the same canonical graph timing.

## What this does not do

- Does not fake R1 measured source evidence.
- Does not fake R8 XPBD runtime proof.
- Does not relax R6 or R10 gates.
- Does not touch R6 KKT/post-projection quality.
- Does not claim PlayMode runtime consumption proof.

## Expected result

If R1/R8 still report missing after v97.39, the likely cause is real upstream missing evidence or another serialization path outside the unified build, not R10 temporary-package drift.

## Sandbox checks

- Zip integrity: passed.
- Conflict marker scan: clean.
- Brace/paren/bracket balance checked for modified C# files.
- Unity compile/test: not run in this sandbox.
