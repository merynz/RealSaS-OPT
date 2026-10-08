# Platform throughput + hydration + GC plan (2026-10-08)

Canonical base: `main@b2eb7148d9aeb2a797451bbd6063f4f3695ab2e2`

Original preregistration is preserved at `canonical/PLATFORM_THROUGHPUT_HYDRATION_GC_PREREG_20261008.json`. Execution topology was amended after repository visibility was verified as public; the amendment is recorded at `canonical/PLATFORM_THROUGHPUT_HYDRATION_GC_AMENDMENT_20261008.json` rather than rewriting the preregistration retroactively.

This branch closes the platform prerequisites before fresh Knight rendering and the general inference executor:

1. remove the single-WSL-runner queue as the default CPU CI bottleneck;
2. keep setup cost bounded and cache identity deterministic;
3. hydrate exact external Knight inputs by content identity;
4. preserve explicit authority boundaries between imported evidence, qualification and stage cache hits.

## Execution topology

Active-development topology:

- **GitHub-hosted `ubuntu-latest`**: host-independent CPU CI, source/governance gates, Compiler regressions, DAG validation, control-plane tests and product-shell validation;
- **Notebook / Colab**: default research lane for GPU inference, FIT/LOFO/unseen courts, Drive-backed experimental hydration and render iteration;
- **Local WSL / GTX 1660 Ti**: opt-in lane for explicit product/native-host witnesses and selected inference when local execution is faster or scientifically useful.

No additional self-hosted CPU runners are required.

Because standard hosted execution is intended only during active public-repository development, current hosted jobs contain a repository-visibility guard. If the repository is switched to private during a pause or trip, those hosted jobs skip rather than consuming private-repository Actions minutes.

Independent hosted workflows retain separate concurrency groups, so unrelated gates can run concurrently. Pull-request workflows continue to cancel superseded heads.

## Setup and cache contract

### GitHub-hosted lanes

Ephemeral runners use dependency-keyed GitHub caches:

- `actions/setup-python` pip cache keyed by the pinned requirements inputs;
- `actions/setup-go` module/build cache keyed by `platform/go.sum`.

The job still installs from pinned requirement files; cache reuse changes cost, not dependency authority.

### Persistent local lanes

Persistent local execution keeps a bounded per-runner tool cache. Current reusable namespaces are fingerprinted Python environments plus pip, Go module and Go build caches.

GC is a correctness contract, not a cleanup afterthought:

1. remove temporary/incomplete bootstrap entries first;
2. remove complete fingerprinted venvs older than the configured max age;
3. under quota/disk pressure, evict complete venvs by least-recent use;
4. if still required, evict reconstructable cache namespaces atomically;
5. never delete a venv whose live-job lock is held;
6. fail closed if quota/free-disk safety cannot be recovered.

Arbitrary files are never pruned from inside a reusable environment or module tree.

## PostgreSQL execution

Hosted Platform-Go uses an ephemeral PostgreSQL service owned by that job. Main and migration databases are isolated from the workstation and disappear with the runner.

Persistent local PostgreSQL remains available for local integration/product witnesses only. CI must never migrate or reset the canonical developer database.

## External artifact hydration

External evidence is addressed by manifest identity, not by assumed host path.

Resolution contract:

1. check verified local SHA-addressed CAS;
2. resolve the declared provider locator (initial provider: Google Drive);
3. download to a temporary path on miss;
4. verify byte size and SHA-256;
5. atomically install under the SHA-256 content address;
6. return the verified local path to the consumer.

Hydration never mints a stage PASS and never converts imported bytes into a stage cache hit. Compiler qualification remains mandatory.

The Knight inventory pins twelve Google Drive file IDs. `canonical/PLATFORM_KNIGHT_DRIVE_LOCATOR_PROOF_20261008.json` records authenticated raw-byte proof that all **12/12** resolve to exact inventoried size + SHA-256 bytes.

A one-time real WSL proof also completed successfully in Actions run `37802717565`: online Drive hydration verified 12/12 inputs, then an offline second pass returned 12/12 `CAS_HIT`. The recurring host hydration workflow was removed after that evidence was produced; notebook/Colab is the default research hydration lane going forward.

### Artifact-CAS GC boundary

Automatic GC of the external artifact CAS remains deliberately disabled. A general executor may hold a verified object while another process observes disk pressure; deleting it without an explicit lease/pin protocol is unsafe.

The general inference executor therefore owns the remaining lifetime rule:

1. acquire leases for all resolved content hashes;
2. allow GC to inspect only unleased entries;
3. permit age/LRU eviction only for unleased remote-rehydratable objects;
4. never evict active or qualification-pinned objects.

Until that contract exists, automated GC applies to reconstructable tool caches and eligible stale local CI databases, not external scientific evidence bytes.

## Knight closure order

1. hosted-first CPU CI + mixed execution governance;
2. setup-cost reduction and bounded persistent-local GC;
3. exact 12/12 Knight external hydration — **sealed**;
4. scoped research render contract;
5. general model inference executor + artifact leases;
6. fresh current-lineage `IDLE / RUN / SLASH` render.
