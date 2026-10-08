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

EngineReleases also seal an immutable DAG snapshot and the Compiler plan hash.
Research can add/remove nodes and rewire dependencies; comparison uses both
the baseline and candidate graphs. Added nodes and changed consumers rerun;
removed nodes remain in the impact record. Unrelated nodes retain their exact
artifact identities even when ordinals move. The resolver, execution input
binding and repair attribution use the Attempt's released graph. Workers reject
an adapter/manifest/graph-node mismatch before executing or hydrating an output.
The executable adapters and matching plan still ship through main; sealing a
graph does not manufacture an executor for a new node.

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

Run `realsas-migrate up`, `realsas-api`, the Go control worker and the Python
Engine worker with the same PostgreSQL database, Temporal namespace and CAS
root. The operator API binds to loopback. Persistent Knight evidence remains on
the self-hosted `realsas-wsl-1660ti` host. CI uses only its self-hosted labels.

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
