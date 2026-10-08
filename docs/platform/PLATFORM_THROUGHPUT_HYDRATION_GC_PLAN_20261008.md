# Platform throughput + hydration + GC plan (2026-10-08)

Canonical base: `main@b2eb7148d9aeb2a797451bbd6063f4f3695ab2e2`

This branch closes three platform prerequisites before fresh Knight rendering and the general inference executor:

1. split the single self-hosted queue into bounded execution lanes without GitHub-hosted minutes;
2. persist expensive local tool/dependency caches, with explicit garbage collection and disk-pressure telemetry;
3. hydrate exact external Knight inputs from declared remote authorities into a local content-addressed store, verifying size + SHA-256 before use.

## Execution lanes

Target topology on the same physical workstation:

- `ci-light-a`: CPU/lightweight contract lane;
- `ci-light-b`: second CPU/lightweight contract lane;
- `gpu-host`: exclusive GPU/stateful host lane.

GPU/host work remains serialized. CPU lanes may run concurrently. Workflows must request capability labels rather than the generic `realsas` label when they need a specific lane.

## Cache contract

Persistent cache roots are bounded resources, never unbounded scratch directories.

Every cache namespace must expose:

- current bytes;
- file/entry count;
- last-access/mtime age distribution;
- configured soft quota;
- configured hard quota;
- bytes reclaimed by the last GC;
- dry-run GC output.

GC policy:

1. remove temporary/incomplete entries first;
2. remove entries older than the configured max age;
3. if still above the soft quota, evict oldest entries until the target watermark is met;
4. never delete entries locked by a live job;
5. fail closed before execution if free disk is below the hard safety floor.

The local content-addressed artifact store follows the same GC contract, but immutable verified artifacts are evicted only by LRU/age and can be rehydrated from their declared authority provider.

## External artifact hydration

External evidence is addressed by manifest identity, not assumed host paths.

Resolution order:

1. verified local CAS hit;
2. declared provider locator (initial provider: Google Drive);
3. download to a temporary path;
4. verify byte size and SHA-256;
5. atomically install under the SHA-256 content address;
6. return the verified local path to the executor.

Hydration never mints stage PASS authority and never turns imported external bytes into a stage cache hit. Existing compiler qualification remains mandatory.

## Knight closure order

- exact 12/12 Knight external inventory hydration;
- scoped research render contract;
- general model inference executor;
- fresh `IDLE / RUN / SLASH` render from the current canonical lineage.
