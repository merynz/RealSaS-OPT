# Truth Foundry V9 — evidence audit trace (2026-11-09)

**Scope: local ChatGPT container receipts, NOT a GitHub-hosted raw-source re-execution.** The previous V6/V7/V8/V9 data engineering steps ran in the conversation's isolated `/mnt/data` runtime, not in GitHub Actions. This branch makes the exact V9 evidence manifest, builders and independent verifier visible in GitHub; CI validates the receipt contract and source code syntax only.

## Actual outputs

- 16,397 candidate rows × 264 preserved truth facets.
- V9 independently recounted 2,466 Master full imported-scene *derivative* receipts, 14,351 mesh object records, 370 historical Core family snapshots (5,180 cells).
- Three additional structural facets × 2,466 Master records = **7,398 newly observed cells** in V9; 4,321,410 unchanged.
- 1,527 Master records still lack a full imported-scene receipt in the available snapshot.
- V5 Core observation qualification and prior partial family diagnostics are **different courts**; their conflicts are retained rather than reconciled into PASS.
- **No source original-byte SHA verification, legal-right clearance, human semantic review, 2D art approval, global SDF PASS, lineage-safe split, or model training admission is minted by V9.**

## Storage and reproducibility

Full V9 CSV/SQLite source-bound outputs are preserved in the connected ChatGPT Library:
`/RealSaS Truth Foundry V1/20261109_Raw_Courts_V9/`.
The V6–V8 raw and derived courts are in `/RealSaS Truth Foundry V1/20261109_Raw_Courts_V6/`.
The persistent private source/provider archive is maintained in Google Drive, including the independently retrieved original forensic matrix V3, Core370 partial family snapshots, PartVerse part evidence and UniML canonical GLB sources.

Immutable local V9 output artifact SHA256s are in `INDEPENDENT_READBACK_COURT_V9.json`. The complete V9 readback verifier in this directory references its original `/mnt/data` paths and is preserved as an **unmodified research snapshot**, not an immediately runnable hosted workflow. The separate `check_readback_receipt_v9.py` runs in GitHub CI without pretending to access private source files.

## Next required hosted work

1. Parameterize absolute paths and retrieve private, SHA-pinned input receipts through the same read-only rclone secret already used by existing Truth Foundry hosted workflows. Do not upload secret/source content to public artifacts.
2. Run independent V9 builder and verifier on a trusted push, publish job summary and the actual per-source SHA/court totals.
3. Extend full original-source byte and rights courts to the 1,527 Master missing receipt cases and other source groups.
4. Run separate Geometry UDF/global-SDF-validity, generic Semantic, 2D Graphics, mechanical deformation and TESSA topology teachers; no admission without source-/target-specific independent courts.

**Never interpret the basic V9 trace CI green check as a full source/corpus truth PASS.**
