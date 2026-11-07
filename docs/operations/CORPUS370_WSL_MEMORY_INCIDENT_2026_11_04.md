# Corpus370 WSL / GitHub runner memory incident — 2026-11-04

## Evidence and uncertainty

- Runner identity: `realsas-wsl-1660ti`, systemd service
  `actions.runner.merynz-RealSaS-OPT.realsas-wsl-1660ti.service`.
  Windows host `DESKTOP-KIB671G`, Ubuntu/WSL; NOT a GitHub-hosted runner.
- Before recovery: 7.7 GiB RAM, 2.0 GiB swap, ~676 GiB disk available.
  `~/.cache/pip` ~296 MiB and APT archives ~917 MiB. Disk space is NOT
  currently constraining; deleting caches cannot prevent an OOM.
- User-provided prior-boot kernel journal proves FOUR OOM kills on 2026-11-02
  from `realsas-platform-engine.service` Python children. Largest measured
  anonymous RSS was 7,077,428 KiB (~6.75 GiB). There is no recorded
  2026-11-04 OOM in that previous-boot journal excerpt; DO NOT claim the
  new runner death was definitively OOM.
- GitHub Actions original native temporal `41841934896` failed after
  ~46.5 minutes, well before its 240-minute timeout, with the active step
  left in-progress and no completed artifact. Its cancellation policy was
  `cancel-in-progress: false`. Runner service later auto-restarted after
  Ubuntu was opened. Prior WSL outage/runner death is likely, cause OPEN.
- No giant memory-consuming process was present in the user's process list
  after Ubuntu restart. A `Runner.Listener run --startuptype service` was
  active already; do NOT launch a duplicate `./run.sh`.

## Containment implemented and MEASURED

- `tools/corpus/run_corpus370_wsl_isolated_v1.py`:
  preflight current MemAvailable >= 5 GiB, root free disk >= 12 GiB;
  creates a non-shell `systemd-run --user --scope` for the worker and
  descendants with `MemoryHigh=3584M`, `MemoryMax=4608M`,
  `MemorySwapMax=1024M`, `CPUQuota=180%`. No unbounded fallback.
- GitHub system service does not export `XDG_RUNTIME_DIR` or
  `DBUS_SESSION_BUS_ADDRESS`. The wrapper explicitly binds to
  `/run/user/$(id -u)/bus`, verifies the existing bus socket and owner,
  and fails closed if it is not accessible. This was detected in failed
  first host probe 41858843665 (NO corpus started).
- **Real WSL verified** in
  [host probe 41859184220](https://github.com/merynz/RealSaS-OPT/actions/runs/41859184220):
  memory.high 3758096384 bytes (3584 MiB),
  memory.max 4831838208 bytes (4608 MiB),
  memory.swap.max 1073741824 bytes (1024 MiB),
  exit 0. No Blender, data deletion, source mutation or training.
- `corpus370-v5-native-temporal-all-v4-real.yml`,
  `corpus370-v5-source-locus-geometric-ray-all352-real-v1.yml`, and
  `corpus370-v5-source-palette-intervention-pilot-real-v1.yml`:
  **workflow_dispatch only**, memory-scoped heavy execution. Never start
  another 352/full authored teacher on source code push.
- Native sweep retains exact locked V5 identity and its stable recovery
  path; original bytes/courts are SHA-checked and uploaded to Drive
  checkpoint-by-checkpoint. No truth approval from this incident fix.
- Legacy UNSTARTED, unguarded runs 41851612569, 41842197638,
  41844980208 were cancelled deliberately; no active truth run was killed.

## Platform engine memory warning

The historical OOM victims were NOT Blender but
`realsas-platform-engine.service` Python children. The new Corpus370
scope protects its own worker and runner; it does NOT cap a separate
platform-engine service. Do not run heavy product-engine and corpus
measurements together on this 8 GiB machine.

Before a larger manual corpus run, inspect the actual user service:

```bash
systemctl --user show realsas-platform-engine.service \
  -p MemoryCurrent -p MemoryHigh -p MemoryMax -p MemorySwapMax
```

An OPTIONAL temporary protection, only if the user accepts that heavy
engine work may fail or slow down under the cap:

```bash
systemctl --user set-property --runtime realsas-platform-engine.service \
  MemoryHigh=2048M MemoryMax=3072M MemorySwapMax=1024M
systemctl --user show realsas-platform-engine.service \
  -p MemoryHigh -p MemoryMax -p MemorySwapMax
```

`--runtime` limits do not persist across system manager restarts.
The command is NOT auto-executed in this PR. Applying it is an operator
decision, since it can terminate an oversized product-engine workload.

## Do NOT do

- Do not `rm -rf /home/monster/realsas_authority` or wipe source/teacher
  NPZ, court, roster, receipt, split, lineage or checkpoint directories.
- Do not randomly `drop_caches`, wipe swap, or kill all Python processes.
- Do not mistake old journal OOM entries for proof of a Nov 4 event.
- Do not promote any unmeasured corpus truth or change 2px mask policy.

## Next

Use the small verified host probe as a guard. Start one guarded heavy
workflow MANUALLY only after checking no concurrent engine batch.
Inspect per-subject fail receipts, resume SHA-verified checkpoints,
and classify OOM/worker failures separately from scientific courts.
