# RealSaS-OPT

RealSaS compiles qualified multi-view 2D artwork into an editable 2D/2.5D puppet with mechanical structure, source-preserving appearance and deterministic runtime consumption.

**Proprietary / all rights reserved.** See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Start here

| Need | Entry |
|---|---|
| Current status and next work | [CURRENT_STATE.md](CURRENT_STATE.md) |
| Diverse Fit8 and first-render admission | [Fit8 admission](docs/corpus/FIT8_ADMISSION.md) |
| Generic product and system evolution | [Product transition](docs/platform/PRODUCTIZATION.md) |
| Operator commands and component replacement | [Execution lanes](docs/platform/EXECUTION_LANES.md) |
| Agent operating rules | [AGENTS.md](AGENTS.md) |
| Live architecture and typed dataflow | [Live map](docs/platform/DYNAMIC_ARCHITECTURE.md) — `python tools/realsas_architecture.py serve` |
| Source ownership | [SYSTEM_INDEX.md](SYSTEM_INDEX.md) |
| Directory map | [REPOSITORY_MAP.md](REPOSITORY_MAP.md) |
| Current contracts and evidence | [Canonical index](canonical/README.md) |
| Earlier records | [Historical index](historical/README.md) |

Canonical source-code continuation is **main only**. Short review PRs carry code changes; research state belongs to Attempts and sealed EngineReleases.

## Current product architecture — RealSaS V2

The current Compiler compatibility plan is a **43-stage dependency DAG**. Public model order is IRIS → TESSA → AXIS → MIRA → Compiler qualification → qualified runtime consumption.

The latest Knight FIT1 mechanics line preserves the exact Stage19 carrier through AXIS V5.4.1, MIRA V5.5 and Compiler qualification of `(M,G,W_M)`. Stage28/32 also seal their checkpoints; Stage35 emits both qualified M and its exact skin binding. Standalone stages29/33/36 have been folded into these producers. Teacher-relative mechanics is PASS; absolute G3 is a separate open gap because the teacher also exceeds its thresholds. Product, unseen and generalization claims remain open.

Geometry, Mechanics and Appearance are co-equal product authorities. Presentation provides editable addressing. Runtime consumes qualified packages and does not re-solve scientific authority.

## Development and product execution

| Layer | Owner |
|---|---|
| Go Platform | Registry, immutable DAG/release snapshots, Attempts, ProductRevisions, dependency scheduling and durable workflows |
| Python Engine | Model inference/training, Compiler semantics, stage execution and proof |
| C++ Runtime | Qualified package consumption, playback and rendering |

Development selects a model or Compiler target. Changed components and affected consumers execute; independent qualified artifacts retain their exact identities. DAG nodes and edges can evolve in a new release without mutating prior releases.

Product compile and product render have separate commands and authority checks. Full pretrained input-to-output product execution and UI integration remain later milestones. Existing FIT receipts do not prove that service.

Use [Execution lanes](docs/platform/EXECUTION_LANES.md) for prerequisites, commands and current import/portability limits. Execution uses the self-hosted `realsas-wsl-1660ti` runner or local resources.

## Current evidence and next work

The mechanics audit is [PLATFORM_MAIN_AUDIT_20261008.json](canonical/PLATFORM_MAIN_AUDIT_20261008.json). Verify the live main ref for current code; its audited baseline SHA is historical evidence.

Current render baseline: one M for geometry and recorded motion, source-wins RGB plus existing CAA. Reproduce it with `python tools/demo/render_knight_m_caa_v1.py --output <directory>`. See [current state](CURRENT_STATE.md). Grip and overlap/draw order are next; product qualification remains open. The previous visual pipeline is retained on `archive/visual-v-before-m-caa-20261011`.

Measured legacy costs include Stage21 CAA compile ~321.7 s, Stage24 qualification ~131.5 s and render ~463.8 s. A sub-two-minute result is a target requiring new measurement.

Historical records remain at their sealed paths where code, workflows or evidence indexes refer to them. [Documentation](docs/README.md) separates present operation from provenance.
