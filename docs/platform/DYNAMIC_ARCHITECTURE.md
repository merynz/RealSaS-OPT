# Source-derived architecture and change impact

`canonical/MAINLINE_EXECUTION_PLAN_V2.json` supplies the executable canonical
DAG. The Go architecture registry supplies module ownership; the same source
import closure used to prepare EngineReleases supplies stage implementation
dependencies. `tools/realsas_architecture.py` reads them on each invocation.
Generated JSON/HTML is a view, never a second registry or execution plan.

Two execution lanes share these modules: development selects a model/Compiler
target with `research-start`/`research-run`; product construction uses
`product-compile`, then `product-render` consumes an exact qualified revision.
Development intentionally reuses sealed FIT outputs/checkpoints instead of
refitting or requesting fresh inference on every iteration. The current
compatibility DAG consumes external FIT receipts for some model steps. Full pretrained input-to-output inference remains an explicit executor
gap; two command lanes do not prove that end-to-end service exists.
Python and the Go toolchain declared by `platform/go.mod` are required for discovery.

```bash
python tools/realsas_architecture.py serve --port 8765
# Open http://127.0.0.1:8765; refreshes after source edits.
python tools/realsas_architecture.py --json-out /tmp/architecture.json map --html-out /tmp/architecture.html
python tools/realsas_architecture.py --json-out /tmp/impact.json impact --base origin/main
python tools/realsas_architecture.py impact --path compiler/realsas_compiler_services/orchestrator/adapters/appearance_v2.py
python tools/realsas_architecture.py impact --manifest-key motion
```

The loopback viewer rechecks source bytes every five seconds and regenerates in
an isolated process when they change. Same-size edits, added/deleted code and
HEAD changes invalidate the view. Generation failures retain the last successful
map with an explicit STALE label; they never show a failed scan as current.
No models run, no files are served outside the fixed viewer/API routes and no
snapshots are committed. Restart the viewer after editing its HTTP server code.

The standalone HTML needs no service, network or JavaScript dependencies. Select
a stage to inspect its upstream producers, manifest input roles, direct consumers,
owner, adapter, policy and source closure. Search a source path to highlight its
consumers and descendants. JSON is the machine-readable counterpart. Every
view includes the checkout SHA, dirty status, plan hash and a byte fingerprint
covering stage source, critical governance, Platform and Runtime code.
The live viewer also compares successive successful maps: changed source consumers, descendants, stage additions/removals and schema/DAG changes. `diff --before /tmp/architecture.json` compares a saved map with the current checkout. Regenerate saved files after edits; the fingerprint is the generated view's identity, not
an assertion that an old file remains live.

The `canonical-architecture-map` CI job regenerates and checks the map on every
main push and PR, including documentation changes, and publishes JSON/HTML as
commit-bound workflow artifacts. It does not commit generated snapshots to main.
Discovery shares adapter closure scans within a single invocation and loads no
model runtime, performs no inference and acquires no agent execution session.

| Observation | Meaning and limit |
|---|---|
| Producer → consumer edge | Declared released-plan dependency, bound as `RealSaS.StageResultManifest`; not proof of inner payload sufficiency or pixel use |
| Typed payload read | Adapter and reachable helper AST request a named schema from a producer, with source line and direct/ancestor/unbound binding; conditional reads are included and unresolved dynamic reads stay visible. Requested outputs do not prove producer emission or runtime use |
| Manifest role | Stage's sealed manifest read set; changing a role reaches its declared consumers and descendants |
| Source edit impact | Import-closure consumers plus DAG descendants; source-only inspection, not a reuse receipt |
| Governance edit | Global closure validation must be rerun; it need not change a per-adapter artifact identity |
| Native render source/header edit | Local C++ include closure reaches Stage44/45 and descendants for inspection; reuse still requires the exact sealed native binary SHA |
| Other native/control/unmapped edit | Explicitly unclassified when no observed consumer maps it; inspect before reuse |
| Shared helper or dynamic import | Real breadth/debt is visible; it is not hidden to claim modular isolation |
| Stage PASS | Separate scoped qualification from a real execution; the map never produces it |

Before execution, Go still compares the baseline and candidate **sealed release
graphs**, stage implementation/policy/semantic input identities and qualified
Registry artifacts. A plan edit is conservatively shown as all-stage inspection;
the graph-aware resolver decides the precise added/removed/changed scope. Nodes
outside displayed impact are not automatically reusable. Static discovery does
not prove that all indirect/dynamic imports or artifact subfields are consumed.

## Agent path

| Work | Entry and exit |
|---|---|
| Read-only architecture/source review | Read `CURRENT_STATE.md`, generate map/impact; no execution session needed |
| Source fix | Canonical main; short review PR allowed, merge before scientific execution; retain exact current-head CI evidence |
| Research execution | Go `agent-context` → exact deployed main `agent-enter` → explicit target/released DAG `research-run` → terminal inspection → exact-scope `agent-exit` |
| Failed or interrupted run | Inspect original command and Go context; retain diagnostics and partial outputs, never silently create another Attempt |
| Main advanced during run | Exit original scope; any new execution must enter an explicit scope on current deployed main |

See [agent handoff](AGENT_HANDOFF.md), [execution lanes](EXECUTION_LANES.md)
and [remaining modularity debt](MODULARITY_AUDIT_20261010.md).

## Current simplification

The current plan has 43 nodes, retaining stable stage IDs and contiguous ordinals.
Stage28 now emits its qualified skeleton plus AXIS checkpoint seal; Stage32 emits
qualified carrier skin plus MIRA checkpoint seal; Stage35 emits qualified M plus
its exact mesh-skin binding. Standalone 29/33/36 were pure seal/identity operations.
Their existing byte checks and binding algorithms still run. Qualification gates
and scientific scope are unchanged. The plan binds actual implementation modules
instead of star-import dispatch proxies. Shared helper breadth remains visible.

Old sealed EngineReleases retain their own DAG snapshots and are not assigned the
current graph. Historical Attempt ledgers are immutable. The disabled, subject-NONE
implementation ledger is synchronized to this current plan. Prior source is kept
on `archive/architecture-before-simplify-20261012` at main 94da501.
