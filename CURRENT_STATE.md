# RealSaS-OPT — Current State

**Date:** 2026-10-11
**Canonical code line:** `main` only  
**Architecture:** RealSaS V2, 46-stage Compiler compatibility DAG; versioned release DAGs
**Implementation readiness:** `canonical/V2_IMPLEMENTATION_READINESS.json`

## Immediate user-authorized priority

The canonical development/render baseline is **one M for geometry, topology and
motion; source-wins RGB plus the already compiled CAA on M-face/barycentric
addresses**. The user explicitly approved this cutover after the sealed eight-view
RUN replay. Independent per-view visual triangulation, visual rebinding and their
production consumers are retired from main. The complete previous main is
accessible at `archive/visual-v-before-m-caa-20261011`; it is historical provenance,
not an alternate execution line.

Use `tools/demo/render_knight_m_caa_v1.py --output <directory>` for the sealed
Knight regression. Its immutable inputs and 21 exact RUN poses live in
`tests/fixtures/single_m_caa_knight_v1/baseline.json`. All eight views are freshly
rendered; the command checks 42 original raw RGBA goldens and the full approved
8-view GIF. It executes no inference, retarget or skinning. The old demo-retarget
trial is not the approved motion baseline.

Stage18 produces M and its addressing domain. Stage23 seals the full source-wins
plus CAA material; Stage25 measures that same renderer, Stage42 checks the exact
M rest/topology and transports Stage41 poses, Stage43 packages the same sealed
material, and the CPU/native consumers sample that M. Uniform and adaptive CAA
lattices retain their existing samples and abstention codes. No independent V
mesh is constructed. Legacy single-M atlas packages remain explicitly readable;
new canonical compiles require the source-wins material and cannot silently fall
back to that older appearance behavior.

Next focus is **grip/weapon relation and semantic overlap/draw order**. Raw depth
and depth ties remain observable limitations; this baseline is not a completed
ProductRevision. Native exact pixel/owner parity in diagonal depth ties and
linear-light shipping composition remain open, separate qualification gates.
No green engineering run may close those gates. The repository implementation
ledger remains disabled, subject NONE and 0/46 PENDING.

The shared IRIS geometry/RGB/semantic head proposal remains unapproved. This
cutover changes geometry/appearance transport, not CAA completion, model heads,
M/G/W or the sealed Knight mechanics witness. Any new scientific execution still
requires the existing Go Attempt/context contract.

## Active platform work

The architecture CLI now generates the canonical producer/consumer JSON and
interactive HTML view plus conservative source/manifest impact. Its source
fingerprint rejects edits during discovery; CI regenerates the view without
committing a second authority. See `docs/platform/DYNAMIC_ARCHITECTURE.md`.
Failed authored-oracle terminal states can export verified partial outputs and
exit the original scope; explicit finalize-only recovery can do this after main
advances, without new execution. Lost-reply retries preserve their exact exit
body. This source repair is not evidence that every historic session is closed;
inspect Go context before recovery. See `docs/platform/AGENT_HANDOFF.md`.

Active experiment: `PLATFORM_EXECUTION_LANES`. State:
`CARRIER_NATIVE_FIT1_SCOPED_PASS__PLATFORM_LANES_IMPLEMENTED`.

The audited mechanics baseline is `949422678d34cf69008cffd6b4d7647c8c066b8a`.
AXIS V5.4.1 and MIRA V5.5 Drive result hashes match its promotion receipt.
Knight teacher-relative FIT1 mechanics is PASS; absolute G3 is a separate open
red gap because the teacher also exceeds those thresholds. Neither cancels the
other. The old frozen renderer is not the latest carrier-native execution proof.

See `docs/platform/EXECUTION_LANES.md` for commands, cache identity, host
requirements and remaining inference/portability limits. Go now exposes separate
research target execution, product compile and qualified product render commands.
Implementation validation and merge evidence are recorded in PR #59. Verify the
live `main` ref rather than treating the baseline audit SHA as the latest code.
Audit receipt: `canonical/PLATFORM_MAIN_AUDIT_20261008.json`.

Exact-byte external evidence import and subject-input sealing are implemented.
The released DAG is carried into Python execution (including smaller research
networks). Import does not mint scientific or reuse qualification.
The latest Knight file index is
`canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json`; bounded self-hosted preflight
reports file hashes and default service availability. A real latest-carrier
Knight Attempt and its fresh render remain pending; engineering tests and a
host inventory must not be reported as that execution proof.

Persistent local developer deployment is specified in
`docs/platform/DEVELOPER_HOST.md`: dedicated PostgreSQL, persistent local Temporal,
Go API/control worker and Python Engine, supervised by the operator's systemd
user manager. `tools/platform_deployment_smoke.py` exercises real control-to-Engine
execution and exact independent artifact reuse after a DAG node removal. Host
installation and this smoke must succeed before declaring the deployment live;
they are engineering evidence, never Knight scientific or product qualification.

The first real host smoke exposed an Engine heartbeat call from a worker thread
and a failed Temporal workflow whose Attempt stayed OPEN. Heartbeat now belongs
to the async event loop; Go reconciles only verified terminal unsuccessful
workflows and retains their original diagnostics. A real completed execution
plus exact reuse receipt has now passed on `main@7be6ef368fcc6b432a26f87dbfac737a7cd88374`
in self-hosted run `37784700012`: first Attempt executed Stage02; second Attempt
removed independent Stage01 and reused the exact Stage02 artifact without
executing it. Both Attempts took 3.282 seconds, with no ProductRevision.
The six persistent developer services are active. This is engineering proof;
it does not establish Knight execution, full inference, rendering or product
latency. Exact identities are recorded in `canonical/PLATFORM_MAIN_AUDIT_20261008.json`.

Latest Knight external hydration is sealed: stable Drive IDs and authenticated
size/SHA256 verification cover 12/12 inventory files. Real run `37802717565`
proved Drive -> CAS -> offline 12/12 CAS_HIT. These are transport identities,
not scientific qualification or stage reuse authority.

PR #65 is merged at `1888d91252fdce23843772f3ec0d0e3ddd91b9ae`.
The user has authorized visual mesh/appearance/runtime completion on main,
preserving the exact mechanics and motion witness. Platform preparation is merged. PR #66's presentation candidate is parked diagnostic history;
its source and evidence are retained, not promoted wholesale. Run `37933851760`
failed both matched V6 Attempts; the later runner changes 37/42/43/45, not only 42.
A fast relation replay's green workflow tolerated dynamic exposure failure and
must not be interpreted as visual acceptance.

Controlled component replacement now requires `research-start.intervention` for
every continuation with a parent Attempt. The
contract requires exact direct changes, a same-subject/baseline-release parent, frozen
unchanged stages, pinned SubjectInput roles and exact qualified baseline artifact
reuse before Engine work. Intended input changes declare their roles and only
invalidate their released consumers/descendants.
First bootstrap Attempts retain ordinary construction behavior. Public research
execution now requires a Go-owned agent entry scope and exact handoff; see
`docs/platform/AGENT_HANDOFF.md`. See `docs/platform/CAUSAL_INTERVENTIONS.md` for the receipt,
limits and operations. The geometry/RGB identity split is a future design, not
implemented IRIS functionality. Current product dependencies can still renew
motion proof artifacts after appearance changes without refitting M/G/W.

The canonical Go guard is live: deployment run `38045971674` on
`main@46e26b3acc32cdd4fc7270b12b4e772162890081` passed real entry -> API -> workflow/reuse
-> handoff and exact independent artifact reuse. Both attempts reused the qualified
cache; no new Engine stage execution is claimed for this smoke. Main deployment now
tracks every main push; narrow control-only path filters must not leave CAA/model
source changes in an undeployed checkout. Service refresh is not model inference
or a change to independent scientific artifact identity.

M/G/W are fixed only for the present Knight witness. Future model/head/evaluation
changes use ordinary DAG invalidation and independent artifact identity. The
current boundary debt and closure gates are recorded in
`docs/platform/MODULARITY_AUDIT_20261010.md`; the architecture is not yet fully
isolated at every output/implementation boundary.

The retired visual ownership/material/rebind boundary map is historical
provenance on `archive/visual-v-before-m-caa-20261011`. Current geometry and
appearance share M; use the canonical replay and generated main architecture map. The subsequent 2026-10-10 instruction
also authorizes IRIS appearance research from genuine layered 2D references/prior
outputs. Knight's 3D source is mechanical truth, not visual truth. The initial
source inspection and unexecuted research protocol are recorded in
`docs/research/IRIS_APPEARANCE_RESEARCH_V1_20261010.md`. This does not establish
visual acceptance, a trained head, a released reference consumer, or a mechanics refit.
Main has no server-side required-check protection; the
connected GitHub app lacks administration capability for that setting.

Release discovery now scans each distinct adapter module once per snapshot,
without a cross-release process cache. Local 46-stage comparison measured
7.413 seconds before and 2.166 seconds after, with the entire returned closure
equal. This is release preparation performance, not inference/render latency.

Optimization applies to the scientific/execution system itself, not only agent
workflow speed: measure cold end-to-end construction separately from warm
component replacement/render. Optimize inference, Compiler, CAA and Runtime hot
paths while preserving qualification and numerical contracts. Product latency
target is below120seconds, ideally60; it is not yet a measured product claim.

## Current product architecture

Geometry, Mechanics and Appearance remain co-equal product authorities. `SurfaceAddressing` is the stable canonical mesh-domain/addressing authority and Complete Appearance Authority owns source-preserving visual appearance. Presentation remains first-class editable addressing authority.

Canonical public model naming is `IRIS -> TESSA -> AXIS -> MIRA`. Legacy `GEPPETTO` / `ARACHNE` / `ATLAS` identifiers may remain where required for schema, checkpoint, artifact or historical compatibility.

### Mechanics — latest developed FIT1 line

- Stage19 owns the exact frozen mechanical carrier `M`.
- AXIS V5.4.1 owns a hard causal required tree plus deterministic 256-bin XYZ; residual diffusion is not runtime position authority.
- Compiler validates/materializes AXIS required parents; it does not reselect the tree on this path.
- MIRA V5.5 predicts `W_M` directly on the exact Stage19 carrier basis.
- Stage32 qualifies carrier-native `W_M`; Stage34 derives a subject-free pre-bind provisional frame/envelope; Stage35 rederives and qualifies those frames against final `W_M` before proving the exact `(M,G,W_M)` carrier; Stage36 is identity re-key/seal only, not semantic skin transfer.
- Generic coincident-frame qualification is subject-free: zero-influence subtrees may inherit parent frame; active coincident subtrees fail closed and require explicit AXIS orientation/tail evidence.

Knight FIT1 mechanics are closed relative to the sealed source/teacher motion envelope. Absolute G3 is intentionally not promoted to a universal PASS because the sealed teacher itself lies outside the frozen absolute thresholds. No FIT8/FITK/unseen/product/generalization claim is made.

## Canonical mesh domain

`compiler/realsas_compiler_core/surface_addressing_v1.py` owns `SurfaceAddressing`; Stage18/19 own construction and static qualification of the mechanical carrier. `QualifiedMeshIR` remains the qualified product mesh contract. The current mechanics chain preserves a single carrier basis across MIRA prediction, dynamic proof and runtime binding.

## Complete Appearance Authority

Appearance / CAA remains source-preserving and proof-bound under `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`. The mechanical mesh is not allowed to silently become a second appearance authority.

## Fresh canonical Knight regression render

The user's acceptance target now requires every visual producer, sealed output,
consumer and independent relation measurement. See
`docs/presentation/VISUAL_RELATION_CLOSURE_20261010.md`. The current source-owned
path lacks required visual-contact, equipment-grip, hidden-support/addressing and
semantic-order contracts. A new bound boundary report distinguishes canonical
CAA field validation from actual visual material transport (zero samples on the
present path). Its propagation through Stage42/RSS/Stage45 is engineering
enforcement, not closure of those missing relationships. The read-only main host
preflight must retain its exact data counts; an incomplete boundary blocks another
full render. Do not convert that blocked result or a diagnostic GIF into acceptance.

Current carrier/CAA replay `38044123895` ran on exact
`main@cc89978e0378086086a788b0c67911cd7058ad35` (PR70), without source overlay or
model inference. All eight IDLE views completed as 12-frame GIFs; RUN and SLASH
completed only V4 as full GIFs. Other motion/view frames failed
`VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW`. The job failed and preserved valid
frames/GIFs/checkpoint/report in artifact `11666802329`. This is diagnostic render
evidence, not Go ProductRevision or visual/product acceptance. The report also
records that canonical CAA completion is not consumed by visual material transport.

Run `37578120472` executed on exact `main@287eabef757cb2b1e6b532a4c2d0a71eb62cd324` and completed PASS. It rebuilt sealed Stage23, requalified Stage24 and freshly rerendered IDLE/RUN/SLASH with exact frozen reference hashes.

This is artifact-level reproducibility. It reused sealed upstream artifacts; IRIS/TESSA/AXIS/MIRA were not reinferred and it is not a fresh source-to-puppet compile.

## Execution model

```text
Git/main            = code truth
Attempt             = research truth
ProductRevision     = production truth
Workflow Engine     = durable execution truth
Artifact Registry   = immutable typed artifact identity/dependencies
Proof/Qualification = promotion eligibility
```

Long-lived research/promotion/integration branches must not become product or scientific authority. Research Attempts may reuse exact qualified upstream artifacts by dependency identity. Product compile runs the required inference/Compiler construction path unless an exact semantic cache hit is proven. Runtime consumes qualified ProductRevision artifacts and must not silently re-solve mechanics or appearance.

## Platform enforcement gap

The Go control plane is the intended owner of durable product/research state, Artifact Registry metadata, Attempt/ProductRevision lifecycle, promotion, dependency scheduling and workflows. Legacy direct authority-root/GitHub Actions paths remain CI/reproducibility mechanisms and must not become a second authority.

## Performance baseline

Fresh regression telemetry remains the current optimization baseline:

- Stage19 static qualification: ~60.9 s
- Stage21 CAA compile: ~321.7 s
- Stage24 qualification: ~131.5 s
- renderer: ~463.8 s

These measured costs define optimization priorities after repository currentness is sealed.

## Current continuation

1. Complete platform/state cleanup and verify exact-head CI plus deployment on
   main. Keep PR #66 archived and preserve scientific scope boundaries.
2. Inspect producer/consumer wiring and edited source impact through
   `docs/platform/DYNAMIC_ARCHITECTURE.md`. Unknown mappings and broad shared
   closures remain visible; engineering PASS does not close appearance contracts.
3. The user's next requested diagnosis is one exact neck-gap point and one
   missing-foot point in the previously examined GIFs. Pin each original code,
   renderer, artifacts, frame/view and coordinate, then trace source → GSA →
   TESSA/M address → Stage18 → Stage37 → Stage42 → CAA → runtime. At each boundary
   record DEFINED / PRODUCED / SERIALIZED / CONSUMED / BEHAVIORALLY USED.
   A boundary not executed in that render is not silently filled from another
   run. Unknown evidence stays unknown; report the first proven break.

The shared IRIS proposal remains unapproved. No new head training, refit or
scientific execution is implied by cleanup or this read-only lineage diagnosis.
General mechanics refitting and FIT8/LOFO/unseen claims remain outside the current
Knight witness; future explicitly declared scopes may change them.

## Read first

1. `canonical/CANONICAL_HANDOFF_20261007.md`
2. `CURRENT_STATE.md`
3. `AGENTS.md`
4. `SYSTEM_INDEX.md`
5. `canonical/V2_IMPLEMENTATION_READINESS.json`
6. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
7. `canonical/KNIGHT_AXIS541_MIRA55_CARRIER_NATIVE_PROMOTION_20261007.json`
8. `canonical/TRUTH_CORPUS_V1_STATUS_20261004.md`

Older recovery/runtime-audit documents remain provenance only and must not override this continuation authority.
