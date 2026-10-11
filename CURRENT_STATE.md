# RealSaS-OPT — Current State

**Date:** 2026-10-11
**Canonical code line:** `main` only
**Architecture:** RealSaS V2, 43-stage Compiler dependency DAG; versioned release DAGs
**Implementation readiness:** `canonical/V2_IMPLEMENTATION_READINESS.json`

## Immediate user-authorized priority

The canonical development/render baseline is **one M for geometry, topology and
motion; source-wins RGB plus the already compiled CAA on M-face/barycentric
addresses**. The user approved the sealed eight-view RUN replay. Independent
per-view visual triangulation and its production consumers are retired from main.
Earlier source remains on `archive/visual-v-before-m-caa-20261011`.

Use `python tools/demo/render_knight_m_caa_v1.py --output <directory>` for the
sealed Knight regression. Immutable inputs and 21 exact RUN poses are pinned in
`tests/fixtures/single_m_caa_knight_v1/baseline.json`. All eight views are freshly
rendered. This executes no inference, retarget or skinning; it is not a fresh
source-to-puppet compile or a Go ProductRevision.

The live architecture entry is `python tools/realsas_architecture.py serve` then
`http://127.0.0.1:8765`. It refreshes from source, shows typed producer/consumer
schema reads and compares successful snapshots. Static JSON/HTML and CLI impact
remain available. See `docs/platform/DYNAMIC_ARCHITECTURE.md`. Unknown reads,
shared closures and ownership debt remain visible; the map cannot authorize reuse.

The current plan contracts three pure transport/seal boundaries: Stage29 into28,
Stage33 into32, and Stage36 into35. Stage28/32 produce checkpoint seals alongside
their qualified skeleton/skin; Stage35 produces qualified M and its exact skin
binding. Existing checkpoint byte checks, binding algorithms and qualification
gates still run. IDs stay stable; ordinals are contiguous. Actual implementation
modules replace star-import proxies in the plan. Previous source is retained at
`archive/architecture-before-simplify-20261012` (main94da501).
Historical EngineReleases and Attempt ledgers retain their own graphs and identities.
The implementation ledger remains disabled, subject NONE and 0/43 PENDING.

Next focus is **grip/weapon relation and semantic overlap/draw order**. Raw depth
and depth ties are known limitations. Native exact pixel/owner parity for diagonal
witnesses and linear-light shipping composition remain open qualification gates.
A green engineering regression must not close those scientific/product gates.

## Current product architecture

Public model order is IRIS → TESSA → AXIS → MIRA → Compiler → Runtime.
Geometry, Mechanics and Appearance remain co-equal product authorities.
`SurfaceAddressing` is the stable canonical mesh-domain/addressing contract;
Complete Appearance Authority owns source-preserving appearance. Presentation
provides editable addressing; Runtime consumes sealed qualified packages.

Stage18/19 construct and statically qualify the exact mechanical carrier M.
AXIS V5.4.1 owns hard required parents and deterministic 256-bin XYZ; Compiler
validates/materializes them without parent reselection. MIRA V5.5 predicts W_M on
that same M. Stage34 derives provisional frames; Stage35 validates final frames
against W_M and proves the exact (M,G,W_M) carrier. No semantic skin transfer runs.
Inactive coincident subtrees may inherit parent frames; active coincident subtrees
fail closed and require explicit AXIS orientation/tail evidence.

Stage23 seals source-wins plus CAA material. Stage25 measures that same renderer;
Stage42 verifies exact M rest/topology and transports Stage41 poses; Stage43
packages that material. CPU/native consumers sample M. Existing uniform/adaptive
CAA samples and abstention codes remain. Legacy single-M atlas packages are
explicitly readable; new canonical compiles require source-wins material.

## Scientific and platform boundaries

Active experiment: `PLATFORM_EXECUTION_LANES`.
State: `CARRIER_NATIVE_FIT1_SCOPED_PASS__PLATFORM_LANES_IMPLEMENTED`.
Knight teacher-relative AXIS/MIRA FIT1 mechanics is PASS in the sealed scope;
absolute G3 remains separately open because the teacher also exceeds its thresholds.
FIT8/FITK/unseen/generalization, full pretrained source-to-puppet execution,
product qualification and sub-two-minute latency are not established.
The shared IRIS geometry/RGB/semantic head proposal remains unapproved.

Go owns Registry, Attempts, immutable release DAGs, ProductRevisions and durable
workflows. Python owns scientific stage semantics; C++ consumes qualified packages.
New scientific execution requires `agent-context` → exact deployed-main
`agent-enter` → explicit released-DAG target execution → terminal inspection →
exact-scope `agent-exit`. Research state belongs to Attempts, source to main.
See `docs/platform/EXECUTION_LANES.md` and `docs/platform/AGENT_HANDOFF.md`.

Platform host/reuse/agent-session engineering evidence is indexed in
`canonical/PLATFORM_MAIN_AUDIT_20261008.json`; its SHAs are historical witnesses,
not the latest main ref. Verified partial-output recovery and finalize-only exits
remain available; inspect original Go context before retrying an interrupted run.
Exact-byte evidence import does not mint scientific qualification. The Knight
input inventory is `canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json`.

Old source-owned visual boundary reports, zero-CAA-consumption findings and failed
per-view renders describe the retired pipeline. They are provenance, not a blocker
or acceptance claim for the approved M+CAA regression. Historical evidence remains
at its sealed paths and archive commits; no ledger is edited to manufacture PASS.

## Current continuation

Keep the live map and simplified plan coherent on main. Verify exact-head tests,
typed bindings and the fresh eight-view approved regression after source changes.
Then address grip/weapon relation and semantic draw order with independently
measured behavior. No model head training, refit or new scientific Attempt is
implied by this architecture cleanup.

Read this file, `AGENTS.md`, `docs/platform/EXECUTION_LANES.md`, then
`canonical/CANONICAL_HANDOFF_20261007.md` and the current plan. Older recovery or
runtime-audit documents must not override this continuation.
