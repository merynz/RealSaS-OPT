# Repository map · RSS.vNext

This map is **human navigation**; it is not a competing architecture configuration. For executable roles, always inspect `platform/internal/capability/contracts/product_capabilities_v1.json` at the checked-out commit. The checked-out catalog is the source of truth. This integrated source tree selects nine RSS.vNext product roles; ProductRevision PASS remains fail-closed.

## Active product-facing homes

| Path | Responsibility |
| --- | --- |
| `models/rss_vnext/` | Single unified product model and physical IRIS/TESSA/AXIS/MIRA experts; stable original Python import aliases |
| `models/rss_vnext/experts/` | Geometry, topology, articulation and skin source modules under one unified release |
| `compiler/rss_vnext/` | Unified RSS.vNext-specific courts and workers (single physical home; 4 modules subsequently updated from the product integration) |
| `compiler/realsas_compiler_core/`, `compiler/realsas_compiler_services/` | Shared canonical primitives, mechanics, graphics, motion, proof and compatibility machinery |
| `runtime/` | Interactive puppet, skin deformation, native rendering |
| `product/` | BUSTED Awaken/Studio |
| `platform/` | Go control plane, releases, CAS, workflow ownership and ProductRevision |
| `tests/`, `tools/` | Tests, engineering and source-derived audits |

## Retained history and scientific apparatus

| Path | Status |
| --- | --- |
| `historical/model_sources/arachne/`, `historical/model_sources/geppetto/`, `historical/model_sources/skin_field_codec/` | Byte-preserved old model lineage with original `models.*` Python import aliases. Do not claim ProductRelease authority. |
| `canonical/` | Legacy mixed namespace: historical experiment/forensic records **and** machine policies with path-bound consumers |
| `historical/` | Historic documentation and restoration records |
| `experiments/` (path-bound scientific apparatus); legacy notebooks, `research/`, `prereg/`, `prospective/` now below `historical/` | Research evidence, notebooks, proposals; no automatic product authority |
| `docs/` | Guidance and reference; dated narratives can be stale |
| `.github/workflows/` | Live GitHub Actions (~222), active RSS.vNext courts and still-referenced operators |
| `historical/ci_workflows/` | Disabled, byte-preserved 2026 scientific workflow definitions and original SHA manifests |

The inherited `canonical/` directory name should not be used as an assertion of present authority. However `canonical/CI_EXECUTION_AUTHORITY_V1.json` and `canonical/PLATFORM_CI_OPERATIONS_AUTHORITY_V1.json` are still referenced by active tests, workflows and operators. Preserve their exact paths until atomic migration.

## Machine-visible housekeeping

`python tools/repository_clarity_audit_v1.py --json` reports live catalog roles, static legacy imports, workflow triggers and CI budget classification. The **230-file workflow budget is a ceiling**, not a recommended count.

Move or rename a historical file only after cataloging path consumers, hashes, dynamic loaders, workflows and regression tests. Git history preserves commits but does **not** repair hard-coded runtime or scientific provenance paths.

Root legacy collection relocation: [`historical/ROOT_COLLECTION_RELOCATION_V1.json`](historical/ROOT_COLLECTION_RELOCATION_V1.json) identifies the exact original Git blobs of former `prereg/`, `prospective/`, `restoration/`, `research/` and root `RESTORATION_STATE.md`. These names no longer exist at top level. Machine-readable historical preregistration discovery is retained.
