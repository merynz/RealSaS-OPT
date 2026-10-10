# Modularity audit — 2026-10-10

Baseline: exact `main@cc89978e0378086086a788b0c67911cd7058ad35`
(PR70). The architecture is not yet fully isolated at every output boundary.
The DAG resolver is modular; several shipped producer and import boundaries are
still broader than the intended component replacement boundary.

## Verified implementation

`tools/realsas_architecture.py audit` found 46 stages, 104 source dependency
files, no missing critical files or module paths, 95 shared dependency files and
93 ownership-debt entries. These counts describe source ownership/import sharing,
not 93 independent functional failures. There were 261 tracked files outside the
current stage closure, including historical diagnostics. The audit is recomputed
from checkout; these numbers are an observation of this baseline, not a new
cached machine authority.

Go compares per-node implementation, policy, parameters, graph and input
read-sets, then invalidates real consumers/descendants. Node addition/removal
uses old and new graph closures. Navigation/ordinal changes alone do not affect
node semantics. Controlled continuation pins exact unchanged qualified artifacts
and fails before Engine execution when reuse is missing or different.

Generic model/head/evaluation tests cover independent appearance-head changes,
geometry/tree/skin changes, evaluation-only changes and shared trunk propagation.
M/G/W are not privileged architectural constants. Knight preservation belongs to
its matched experiment. New unseen subjects bootstrap new Attempts; evaluations
version their actual corpus, policy and metrics.

## Concrete boundary debt and closure gates

| Boundary | Evidence and consequence | Required correction and acceptance |
|---|---|---|
| Agent entry/exit | Public research execution previously accepted a run without a durable scope/handoff | The accompanying Go change requires a session and mandatory parent intervention; deployment smoke must pass before calling the live path guarded |
| CAA field -> visual texture | `appearance_v2.bake_complete_appearance_asset_stage` binds the canonical completion NPZ but writes source-direct PNGs; `render_knight_sealed_v6.py` reports that the field is not consumed by material transport | Add qualified surface-to-part/view material transport with support/provenance; prove completed samples reach visible pixels without overwriting source or inventing hidden-layer qualification |
| Stage18 geometry/visual output | One StageResultManifest carries mechanical candidate and visual substrate | Publish separately versioned output producers/read-sets; matched replay must preserve exact mechanical artifact identity for a visual-only change |
| Appearance -> motion | Current CAA -> Stage38 puppet -> Stage39/40/41 dependency renews motion evidence after appearance changes | Separate mechanical motion witness from presentation/material sealing with equivalent acceptance proofs; do not just remove dependency edges |
| Adapter/import sharing | Stage20–25 share `appearance_v2.py`; other adapters import it transitively. A file change legitimately widens implementation impact | Split producer implementations and shared pure contracts into real independently imported modules; closure comparisons and actual execution must agree. Never omit hashes to force a small scope |
| Legacy execution surfaces | Many historical workflow/direct scripts still address authority-root files outside the normal Go path | Classify/archive diagnostic entrypoints and route new scientific/product state through Go; filesystem access still requires operational enforcement |
| Visual acceptance | Connected palette/domain implementation is now on main, but amodal material, contact/order and full native parity remain open | Qualify matched dynamic witnesses and native end-to-end playback. A rendered GIF or green engineering CI is not product acceptance |

## Simplification order

Finish the already sealed render and retain its frames. Deploy the Go handoff
guard, then close CAA material consumption before adding more rendering variants.
Separate actual geometry/visual and mechanics/motion/material producer boundaries
one at a time with exact reuse receipts. Archive historical entrypoints only
after their evidence has a canonical address and replay purpose.

Keep one code line and one state owner. Use existing Attempts, Registry, released
DAGs and workflow journals; no new branch ledger, alternate scheduler, universal
M/G/W lock or per-diagnostic authority mechanism.
