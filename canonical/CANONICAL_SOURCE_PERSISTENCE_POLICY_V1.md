# RealSaS-OPT — Canonical Source Persistence Policy V1

**Date:** 2026-08-18  
**Status:** BINDING FOR NEW CANONICAL EXPERIMENTS

## Motivation

A canonical experiment must never depend on source code that exists only in a transient runtime such as `/mnt/data`, `/content`, `/tmp`, a notebook kernel, or an unpersisted scratch workspace.

The incident that motivated this policy is the post-N1D global-foreground treatment whose source bytes were hashed and used for preregistered canonical experiments, but the exact source file was not persisted alongside the reports/results. The SHA-256 survived; the source bytes did not.

## Three-store persistence gate

Before any experiment may be called **canonical**, the exact executable source bundle used for prediction/evaluation must exist in all three persistent stores:

1. **GitHub** — executable source, tests, prereg, compact manifests/results/reports.
2. **Google Drive** — mirrored source bundle plus heavy checkpoints/corpora/caches/proof packs.
3. **ChatGPT Library** — mirrored source bundle and canonical manifest for retrieval/audit.

Transient local paths are cache only and have **zero provenance authority**.

## Required canonical source manifest

Every canonical prereg must reference a source manifest containing at minimum:

- source bundle filename;
- source bundle SHA-256;
- GitHub repository + commit SHA + repository path;
- Google Drive file/folder ID + path;
- Library file ID + path;
- checkpoint SHA-256 and persistent location(s), when applicable;
- evaluator/GFDR source SHA-256 and repository path;
- test command and test result summary;
- creation timestamp;
- experiment/prereg identifier.

If any required persistent locator is missing, the prereg state is:

`SOURCE_PERSISTENCE_GATE = FAIL`

and prediction/evaluation is not authorized.

## Ordering rule

The persistence gate runs **before** prereg freeze and before any unopened evaluation truth may be accessed:

```text
source finalized
 -> tests PASS
 -> build immutable source bundle
 -> SHA-256
 -> persist GitHub
 -> persist Drive
 -> persist Library
 -> verify all three copies / locators
 -> freeze source manifest
 -> freeze prereg
 -> execute experiment
```

No canonical result may retroactively repair a missing source-persistence gate.

## Source changes

Any executable source change, including a one-line hotfix, creates a new source authority:

- new source bundle;
- new SHA-256;
- new GitHub commit;
- new Drive mirror;
- new Library mirror;
- new manifest;
- new prereg lineage as required.

Notebook state, monkey patches, shell edits inside a runtime, and modified files under transient working directories are forbidden as unrecorded canonical authority.

## Notebook rule

Notebooks are orchestration surfaces, not source authority. Canonical algorithmic code must live in versioned source modules. A notebook may import/run those modules but must not contain the only authoritative implementation of an experiment route.

## Recovery rule

If exact source bytes cannot be recovered, an old result may remain historical evidence, but the route must be marked `SOURCE_BYTES_UNRECOVERED`. Any reconstruction must pass a preregistered behavioral-parity recovery gate before it can replace the missing source, and the reconstructed source must then pass this three-store persistence gate before further canonical use.

## Storage authority summary

```text
GitHub  = code/version/provenance authority
Drive   = heavy depot + durable source mirror
Library = retrieval/audit mirror
local runtime = disposable cache only
```

This policy is intended to make loss of canonical executable source impossible under normal workflow.