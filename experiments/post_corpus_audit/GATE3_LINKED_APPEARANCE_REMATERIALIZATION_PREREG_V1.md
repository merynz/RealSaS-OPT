# Gate 3 — Linked Appearance Re-materialization Prereg V1

**Date:** 2026-08-23  
**Status:** `FROZEN_BEFORE_LINKED_APPEARANCE_RECOVERY_PROBE`

## Why this gate exists

Stage-A found 3350 selected assets without a locally preserved raw-source path. This is expected under the V4.3 production policy `preserve_reproducible_linked_raw=False`: reproducible linked originals are not persisted after canonical extraction, while immutable/retrievable identifiers, source SHA-256, byte size, license and provenance are retained.

Therefore `no local raw path` is not equivalent to `appearance lost`. The scientific question is whether the selected linked originals can still be re-materialized byte-exactly enough to support an appearance-preserving B observation pass.

## Frozen question

For the selected non-local linked-original population, can the original source object be re-materialized from the preserved immutable identifier/provider semantics and verified against the preserved source SHA-256/size, without changing canonical geometry authority?

## Method

1. Reuse the V4.3 provider-specific materialization semantics; do not invent a new generic downloader.
2. GitHub-linked objects must use exact commit/path retrieval and byte SHA verification.
3. Non-GitHub Objaverse-XL providers must use the same provider-specific identifier semantics as the production builder.
4. Start with a deterministic source/provider-stratified sample; expand only if failures are material or source-class specific.
5. Re-materialized raw files are scratch inputs for appearance inspection/rendering. They do not replace canonical `primary_geometry.npz` authority unless Gate 2 independently proves the old extracted geometry invalid.

## Outputs

For every probed asset:

- provider/source class;
- immutable identifier;
- expected SHA-256 and byte size;
- retrieval success/failure;
- byte SHA match;
- recovered container type;
- material/image dependency summary;
- whether appearance-preserving B rendering is feasible from the recovered object.

## Decision

- High byte-exact recovery by source class -> B-pass may rematerialize on demand; no need to persist all raw files permanently.
- Source-class-specific failure -> bounded provider repair or reduced B coverage with explicit stratification.
- Widespread irrecoverability -> current corpus remains valid as geometry-control A, but is not sufficient as the sole appearance-domain training corpus; add separate appearance-authority sources rather than corrupting A.

No IRIS optimizer step is authorized by this prereg.
