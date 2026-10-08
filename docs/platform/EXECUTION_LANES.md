# Development and product execution

Go owns Attempts, EngineReleases, artifact dependencies, the transactional
outbox and Temporal workflows. Python executes the selected scientific stage;
C++ consumes the qualified native package. Git `main` owns code, never the
state of a research experiment.

The immediate delivery is the modular developer path used by operators and
agents: replace selected components, rerun their dependent target closure and
preserve exact independent artifacts/checkpoints. Full input-to-output product
execution and its UI integration are a later productization milestone.

| Operation | Command | State and permitted result |
|---|---|---|
| List the current DAG | `realsasctl stages` | Canonical stage identities and dependencies |
| Register a subject | `realsasctl subject --request subject.json` | Idempotent slug; conflicting identity fails closed |
| Copy exact input bytes | `realsasctl artifact-put --file input --sha256 SHA` | Verified immutable CAS bytes, no qualification |
| Register external evidence | `realsasctl artifact-import --request import.json` | Typed external input, never a stage reuse qualification |
| Seal subject inputs | `realsasctl input-seal --request inputs.json` | Exact role/artifact bindings for an Attempt |
| Seal a version snapshot | `realsasctl release --request release.json` | Immutable RESEARCH or PRODUCT EngineRelease |
| Compare changed components | `realsasctl research-start --request attempt.json` | Research Attempt and changed stages plus descendants |
| Run a development target | `realsasctl research-run --request compile.json` | Existing research Attempt; explicit target; no ProductRevision |
| Construct a product | `realsasctl product-compile --request compile.json` | PRODUCT release; exact semantic reuse; Stage46 qualification before revision |
| Render an existing product | `realsasctl product-render --request render.json` | Exact ProductRevision, motion and render settings; no model stage scheduling |
| Inspect progress | `realsasctl attempt --id UUID` | Stored state and timestamped events |

## Component replacement

Prepare the candidate release from the edited checkout and the intended run
manifest, without loading torch or fitting models:

```bash
python tools/platform_release_snapshot.py \
  --run-manifest /absolute/path/run_manifest.json \
  --name candidate-v1 --purpose RESEARCH --created-by operator \
  --out /absolute/path/release.json
cd platform
go run ./cmd/realsasctl release --request /absolute/path/release.json
```

Each stage pins its transitive implementation hash, policy and declared
manifest read set. Checkpoint, receipt and proposal references therefore belong
to the consuming stage's version identity. A changed shared helper legitimately
invalidates every consumer of that helper.

Release preparation shares each adapter module's source closure within one
snapshot. It retains the exact transitive files and per-stage hash schema;
the next snapshot rereads all source so an edit cannot hide behind a process
cache. This reduces repeated discovery work, not inference or render cost.

EngineReleases also seal an immutable DAG snapshot and the Compiler plan hash.
Research can add/remove nodes and rewire dependencies; comparison uses both
the baseline and candidate graphs. Added nodes and changed consumers rerun;
removed nodes remain in the impact record. Unrelated nodes retain their exact
artifact identities even when ordinals move. The resolver, execution input
binding and repair attribution use the Attempt's released graph. Workers reject
an adapter/manifest/graph-node mismatch before executing or hydrating an output.
The executable adapters and matching plan still ship through main; sealing a
graph does not manufacture an executor for a new node.

The Engine receives the exact released graph in both compile-stage and
stage-capability requests. It reconstructs the full execution plan using that
graph plus scientific policy metadata shipped in the canonical Compiler plan;
the complete reconstructed plan hash must equal the release's bound hash.
Thus the Go scheduler cannot seal one DAG while Python executes another.
Research may use `platform_release_snapshot.py --plan research-plan.json` for a
smaller network or a rewired DAG. Product still requires the exact current
canonical plan. Unshipped metadata, unknown/future or duplicate dependencies,
missing graphs and policy/hash drift fail before execution. A research network
may omit product closure; this does not authorize product admission.

## External evidence import boundary

Upload bytes with their expected SHA256, then register the returned CAS object
using `artifact_type`, `schema_version`, `object`, `source_uri`, and `created_by`.
The source URI is provenance only: moving identical bytes does not change input
identity. Import never issues `REUSE_ELIGIBLE` or `DEMO_REUSE_ELIGIBLE`; a consuming
Engine contract must validate and reseal them. The API rejects qualification
fields supplied by an importer. Seal roles with `subject_id`, `bindings` and
`created_by`; each binding contains `role` and `artifact_id`.

External research bytes may remain in Google Drive while Git stores only the
reviewable manifest/provenance contract. Execution resolves the declared remote
locator into a local or notebook CAS and verifies exact byte size and SHA256
before registration. Provider choice (Drive connector, notebook mount, rclone,
or another future fetch adapter) is transport only; content identity and
scientific authority do not depend on the transport or filesystem path.

Migration08 separates semantic identity from storage deduplication: two typed
contracts may share one verified CAS object while retaining separate producer,
dependency and qualification records. Downgrading to the old globally unique
content/storage pair fails if such aliases exist; it never discards them.

For a segmented `manifest:<key>` role, import the exact JSON value of that
manifest section (with pinned file hashes), not one NPZ under a misleading section
role. Engine verifies section equality and referenced file bytes before executing.
Unsegmented raw inputs must be referenced by content hash in the manifest; unknown
bytes or stale host files fail closed. The Go request carries the exact source
artifacts bound to the execution, alongside qualified stage-result inputs.

`canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json` is the narrow latest-input
index. `tools/platform_host_preflight.py` searches only an explicit authority
root with scan limits and verifies exact file size/hash. Its report is host
diagnostics, not scientific PASS or a completed Knight Attempt. Raw notebook
NPZ/JSON evidence still needs exact qualified carrier/model receipt bindings,
matching CAA/cameras/motion and a live Go/Temporal deployment. Embedded file
references are not silently rewritten during import.

Start a research Attempt with `subject_id`, `baseline_engine_release_id`,
`candidate_engine_release_id`, optional `parent_attempt_id`, and `created_by`.
Use the returned Attempt ID as `research_attempt_id` in its compile request.
The request also specifies `subject_id`, `engine_release_id`,
`subject_input_id`, `target_stage_id`, `compiler_run_id`, `run_manifest_path`,
`run_ledger_path`, `pipeline_plan_sha256`, `idempotency_key`, and `requested_by`.
Run manifests must already exist on the Engine host at the bound canonical run
path. The Engine initializes a missing run-local ledger after version checks.

The resolver reconstructs the target's ancestor closure. Unchanged qualified
stage identities reuse Registry artifacts; changed identities and descendants
execute. Engine workers materialize bound CAS output bytes before invoking a
consumer. Changing skin keeps independent CAA, IRIS and AXIS outputs. Changing
the separately sealed `manifest:motion` input keeps all stages before39.
Unsegmented legacy source inputs intentionally invalidate conservatively.

## Product lane

Product requests cannot attach a research Attempt or use a RESEARCH release.
Research demo qualifications are never product cache hits. Research completion
does not create a ProductRevision, including when its target is Stage46.
Product revision sealing still requires the existing scientific qualifications.
Render cache identity includes revision, qualified motion, view and settings.

The 46-stage compatibility DAG currently validates external FIT execution
receipts and their pinned results. It does **not** itself launch a complete
pretrained source-to-puppet inference service. Do not advertise that service or
a two-minute product latency until its executor and measured end-to-end proof
exist. The separated control paths are a prerequisite, not that proof.

## Host and migration requirements

See `DEVELOPER_HOST.md` for the persistent, loopback-only WSL user-service
installation and real API/outbox/Temporal/Engine smoke. This local developer
deployment is not production infrastructure or Knight render readiness.

When an explicit local integration witness is required, run `realsas-migrate
up`, `realsas-api`, the Go control worker and the Python Engine worker with the
same PostgreSQL database, Temporal namespace and CAS root; the operator API
binds to loopback. This is no longer the default CPU-CI lane. Host-independent
current contracts run on standard GitHub-hosted Linux while the repository is
public, and are guarded to skip when the repository is private. Notebook/Colab
is the default research/GPU lane; local WSL/GTX execution remains available for
native-host witnesses or selected inference where it is useful.

Legacy Registry stage manifests without implementation, manifest read-set or
graph-node identity or portable output fields fail closed during hydration; re-import/reseal them
through verified execution. CAS copies preserve payload bytes, including any
embedded file references. Those references still require the persistent source
authority root: complete recursive portability to a different machine remains
open. Do not silently rewrite payloads or substitute a historical package.
Legacy EngineReleases without a DAG snapshot must be explicitly resealed; the
platform never assigns the current graph to an old release during execution.
Downgrading to the legacy 46-stage database constraint is incompatible with
expanded research releases; migration rollback must fail rather than discard
those sealed records. CI tests reversible DDL in a separate empty database.

## Scientific and latency boundaries

Knight AXIS V5.4.1 and MIRA V5.5 FIT1 are PASS within their recorded scopes.
Carrier-native mechanics is PASS relative to the sealed source/teacher motion
envelope. The teacher also exceeds absolute G3 thresholds; that separate red
gap remains open and does not cancel the scoped PASS. Absolute G3, product
qualification and generalization remain separate claims.

Old frozen IDLE/RUN/SLASH recovery and the August vendor consumer/MWB0/MWB1
replays, including the four frozen September R6 training replays, are manual
historical reproducibility. They no longer queue on every current Compiler
edit. Automatic PR checks cancel superseded heads within each workflow and PR;
current mainline contracts remain automatic.
Historical recovery cannot prove the newest carrier-native package rendered. A fresh render
receipt must record exact code SHA, carrier/skeleton/weights identities,
input artifacts, output hashes, execution mode and wall time. Measure cold
compile and warm render separately; the target is below120seconds, ideally60.
