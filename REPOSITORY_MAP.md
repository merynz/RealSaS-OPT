# Repository map · RSS.vNext

This map is **human navigation**; it is not a competing architecture configuration. For executable roles, always inspect `platform/internal/capability/contracts/product_capabilities_v1.json` at the checked-out commit. The nine-role RSS.vNext candidate is in PR #485; `main` has not yet adopted it.

## Active product-facing homes

| Path | Responsibility |
| --- | --- |
| `models/rss_vnext/` | One shared model inference; specialist evidence fanout |
| `models/iris/`, `models/tessa/`, `models/axis/`, `models/mira/` | Geometry, topology, articulation and skin expert implementations |
| `compiler/` | Evidence adjudication, canonical IR, Mechanics, Graphics, Motion, Proof |
| `runtime/` | Interactive puppet, skin deformation, native rendering |
| `product/` | BUSTED Awaken/Studio |
| `platform/` | Go control plane, releases, CAS, workflow ownership and ProductRevision |
| `tests/`, `tools/` | Tests, engineering and source-derived audits |

## Retained history and scientific apparatus

| Path | Status |
| --- | --- |
| `historical/model_sources/arachne/`, `historical/model_sources/geppetto/`, `historical/model_sources/skin_field_codec/` | Byte-preserved old model lineage with original `models.*` Python import aliases. Do not claim ProductRelease authority. |
| `canonical/` | Legacy mixed namespace: historical experiment/forensic records **and** machine policies with path-bound consumers |
| `historical/`, `restoration/` | Historic documentation and restoration records |
| `experiments/`, `research/`, `notebooks/`, `prereg/`, `prospective/` | Research evidence, notebooks, proposals; no automatic product authority |
| `docs/` | Guidance and reference; dated narratives can be stale |
| `.github/workflows/` | GitHub Actions; not a mirror of the active product DAG |

The inherited `canonical/` directory name should not be used as an assertion of present authority. However `canonical/CI_EXECUTION_AUTHORITY_V1.json` and `canonical/PLATFORM_CI_OPERATIONS_AUTHORITY_V1.json` are still referenced by active tests, workflows and operators. Preserve their exact paths until atomic migration.

## Machine-visible housekeeping

`python tools/repository_clarity_audit_v1.py --json` reports live catalog roles, static legacy imports, workflow triggers and CI budget classification. The **430-file workflow budget is a ceiling**, not a recommended count.

Move or rename a historical file only after cataloging path consumers, hashes, dynamic loaders, workflows and regression tests. Git history preserves commits but does **not** repair hard-coded runtime or scientific provenance paths.
