# Platform throughput + hydration + GC plan (2026-10-08)

Canonical base: `main@b2eb7148d9aeb2a797451bbd6063f4f3695ab2e2`

This branch closes three platform prerequisites before fresh Knight rendering and the general inference executor:

1. split the single self-hosted queue into bounded execution lanes without GitHub-hosted minutes;
2. persist expensive local tool/dependency caches, with explicit garbage collection and disk-pressure telemetry;
3. hydrate exact external Knight inputs from declared remote authorities into a local content-addressed store, verifying size + SHA-256 before use.

## Execution lanes

Target topology on the same physical workstation:

- `realsas-ci-light-a`: CPU/lightweight contract lane;
- `realsas-ci-light-b`: second CPU/lightweight contract lane;
- the existing legacy `realsas` runner: exclusive GPU/stateful-host lane after cutover.

The two new CPU runners intentionally do **not** receive the legacy `realsas` label during bootstrap. Existing workflows therefore cannot accidentally execute on them before reviewed workflow cutover. GPU/host work remains serialized; CPU lanes may run concurrently. Any self-hosted workflow reachable from `pull_request` must reject fork PRs before checkout because this repository is public.

`tools/platform_runner_lanes_bootstrap.sh` configures the two CPU runner instances once a short-lived GitHub runner registration token is supplied by the operator. Registration credentials are never stored in this repository.

## Tool-cache contract

Persistent **per-runner tool caches** are bounded resources, never unbounded scratch directories. Current namespaces are the fingerprinted Python venv cache plus pip, Go module and Go build caches.

Every GC run reports:

- current bytes and entry counts;
- venv recency/age;
- namespace byte counts;
- configured soft quota and target watermark;
- minimum free-disk safety floor;
- bytes reclaimed and exact eviction reasons;
- dry-run output when requested.

GC policy:

1. remove temporary/incomplete bootstrap entries first;
2. remove complete fingerprinted venvs older than the configured max age;
3. under quota/disk pressure, evict complete venvs by oldest-use first;
4. if still required, evict reconstructable cache namespaces atomically (`go-build`, then `pip`, then `go-mod`);
5. never delete a venv whose lock is held by a live job;
6. fail closed before execution if the configured soft quota/free-disk safety floor cannot be recovered.

Arbitrary files are never pruned from inside a reusable venv or Go module tree. This prevents GC from manufacturing a partially valid environment.

## Persistent PostgreSQL without shared test state

The already managed loopback-only RealSaS PostgreSQL server is reused as transport. CI does **not** migrate/reset the canonical developer database.

Each workflow run creates unique ephemeral databases:

- `realsas_ci_<epoch>_<run>_<attempt>_main`
- `realsas_ci_<epoch>_<run>_<attempt>_migrations`

The job drops its own databases in `if: always()` cleanup. A bounded stale-database GC removes only valid RealSaS CI database names older than the configured age and only when they have no active connections.

## External artifact hydration

External evidence is addressed by manifest identity, not assumed host paths.

Resolution order:

1. verified local external-artifact CAS hit;
2. declared provider locator (initial provider: Google Drive);
3. download to a temporary path;
4. verify byte size and SHA-256;
5. atomically install under the SHA-256 content address;
6. return the verified local path to the executor.

Hydration never mints stage PASS authority and never turns imported external bytes into a stage cache hit. Existing Compiler qualification remains mandatory.

The Knight inventory now pins twelve Google Drive file IDs. `canonical/PLATFORM_KNIGHT_DRIVE_LOCATOR_PROOF_20261008.json` records an authenticated raw-download proof that all **12/12** locators resolve to the exact inventoried size and SHA-256 bytes.

### Artifact-CAS GC boundary

Automatic GC of the external artifact CAS is **deliberately disabled in this branch**. A model/render executor may hold a verified artifact open while a different job observes disk pressure; deleting it without an explicit lease/pin protocol is unsafe. The general executor closure therefore owns the remaining CAS-lifetime rule:

1. executor acquires leases for all resolved content hashes;
2. GC may inspect only unleased entries;
3. unleased remote-rehydratable objects may then use age/LRU eviction;
4. active or qualification-pinned objects are never evicted.

Until that lease contract exists, disk cleanup applies to tool caches and stale CI databases, not automatically to external evidence CAS bytes.

## One-time host bootstrap

After merge-ready code review, the operator performs two credential-bearing steps that cannot be committed:

1. create a short-lived repository self-hosted-runner registration token and run `tools/platform_runner_lanes_bootstrap.sh` with `REALSAS_GITHUB_RUNNER_TOKEN` set;
2. configure an `rclone` Google Drive remote named `realsas-drive` with download/read-only scope. Prefer a project-owned Google OAuth client rather than rclone's shared client credentials.

Then the exact Knight external bytes are hydrated with:

```bash
python3 tools/platform_artifact_store.py \
  --inventory canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json \
  --cas-root "$HOME/.cache/realsas/external-artifacts" \
  --rclone-remote realsas-drive \
  --out /tmp/knight-hydration.json
```

A second offline invocation must prove that all inputs are local verified hits without Drive access:

```bash
python3 tools/platform_artifact_store.py \
  --inventory canonical/PLATFORM_KNIGHT_INPUT_INVENTORY_V1.json \
  --cas-root "$HOME/.cache/realsas/external-artifacts" \
  --offline \
  --out /tmp/knight-hydration-offline.json
```

## Knight closure order

- exact 12/12 Knight external inventory hydration;
- scoped research render contract;
- general model inference executor plus external-artifact lease semantics;
- fresh `IDLE / RUN / SLASH` render from the current canonical lineage.
