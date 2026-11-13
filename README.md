# BUSTED · RealSaS-OPT

**We don't generate. We compile.** The product is an **editable puppet**: source artwork → unified RSS.vNext evidence → qualified geometry, articulation, skin, mechanics, 2D presentation, motion → Proof → durable Studio puppet.

> **Execution status (13 November 2026):** This source tree selects a single nine-role RSS.vNext product DAG (Input, Shape, Mesh, Rig, Skin, Mechanics, Graphics, Motion, Proof). Independent ProductProof remains fail-closed while source appearance, contact, renderer parity and persistent edit/reopen witnesses are missing. A successful source test is not a ProductRevision.

## Product architecture — start here

| Area | Product-facing source |
| --- | --- |
| Unified inference and specialist evidence | [`models/rss_vnext/`](models/rss_vnext/) and [IRIS/TESSA/AXIS/MIRA](models/README.md) |
| Canonical compilation and independent qualification | [`compiler/rss_vnext/`](compiler/rss_vnext/) |
| Deform, render and edit consumers | [`runtime/`](runtime/) |
| Awaken and Studio | [`product/`](product/) |
| Releases, capability DAG, artifacts and ProductRevision | [`platform/`](platform/) |
| Regressions and reproducible operators | [`tests/`](tests/) · [`tools/`](tools/) |

**Machine source of truth:** [active product capability catalog](platform/internal/capability/contracts/product_capabilities_v1.json) **at the checked-out SHA**. [System responsibilities](SYSTEM_INDEX.md) and [repository map](REPOSITORY_MAP.md) explain how to read it. [Agent operating contract](AGENTS.md) and [execution lanes](docs/platform/EXECUTION_LANES.md) govern operations.

## Research, compatibility and archive

[`canonical/`](canonical/README.md) is an **inherited name**, not a claim that every old Knight/FIT artifact is active. It mixes sealed historical research with a few referenced machine policies. [`historical/`](historical/README.md) contains archival preregistrations, prospective drafts, restoration files and source-evidence receipts. [`experiments/`](experiments/) and [`docs/`](docs/) retain their existing path-bound scientific/operator content. Older `models.geppetto` and `models.arachne` import identities (now stored under `historical/model_sources/`) are not independent RSS.vNext product releases, but still have compatibility imports; do not relocate without dependency checks.

The repository contains many workflows. **Only 26** are explicitly named in the current RSS.vNext PR fast/full CI authority; other workflows require trigger, caller and coverage audit before consolidation. Run `python tools/repository_clarity_audit_v1.py --json` for a read-only live inventory.

Canonical development continues on `main`. PRs are review/integration surfaces, not alternative scientific authorities.

**Proprietary / all rights reserved.** See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Root layout:** Current product code lives in `models/`, `compiler/`, `runtime/`, `product/`, `platform/`; verification lives in `tests/`, `tools/`. Historic proposals, preregistrations and restoration records have been moved out of root to [`historical/`](historical/README.md) with SHA receipts. `canonical/` is **not yet** fully migrated: some CI and lineage policies are path-bound and must be migrated separately.
