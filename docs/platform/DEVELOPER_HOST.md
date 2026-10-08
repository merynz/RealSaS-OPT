# Persistent local developer host

This is the modular **developer** deployment, not production infrastructure or
Knight scientific qualification. PostgreSQL owns Registry/Attempts; Temporal
owns durable workflows; Go owns scheduling; Python runs released adapters.
Temporal CLI's SQLite development server is explicitly not production-ready.

Code ships from exact canonical `main`. Short PRs validate installation contracts
without mutating services; a deployment-file change merged to main starts the
authorized self-hosted install. No GitHub-hosted runner is used. Unrelated model
or Compiler changes do not reinstall services automatically.

## Once on WSL

The operator must have a running **user** systemd manager. If the installer reports
`USER_SYSTEMD_UNAVAILABLE`, run in the normal `monster` WSL terminal:

```bash
sudo loginctl enable-linger "$(id -un)"
systemctl --user show-environment >/dev/null
```

This operator step gives user services a lifecycle independent of the Actions
runner and the interactive shell. The installer never runs sudo, changes runner
tracking, uses nohup, kills another service, or resets a database. If systemd is
unavailable even after this step, report that blocker instead of bypassing it.

## Layout and commands

Deployment root: `/home/monster/realsas_platform`.

| Location | Responsibility |
|---|---|
| `releases/<main-sha>-<installation-id>/` | Immutable source snapshot, Go binaries, Python environment, pinned Temporal CLI |
| `current` | Atomically selected code release |
| `data/postgres/` | Dedicated PostgreSQL cluster; private Unix socket, no TCP listener |
| `data/temporal.sqlite` | Local durable Temporal development history |
| `artifacts/cas/` | Exact immutable input/output bytes |
| `deployment.json`, `receipts/` | Installation identity and engineering smoke results |
| `/home/monster/realsas_authority/runs/` | Exact Compiler manifests, ledgers and persistent source references |

The operator API and Temporal frontend bind to loopback. PostgreSQL trust
authentication is confined to the private owner-only socket directory, not an
open network listener. Services run as the normal user, with restrictive umask.

From a clean checkout at the exact current main SHA, with Go 1.27.1 and Python3.12:

```bash
python tools/platform_deploy.py \
  --root /home/monster/realsas_platform \
  --authority-root /home/monster/realsas_authority \
  --expected-main-sha "$(git rev-parse HEAD)"
/home/monster/realsas_platform/current/bin/realsas-python \
  /home/monster/realsas_platform/current/source/tools/platform_deployment_smoke.py \
  --root /home/monster/realsas_platform
systemctl --user status realsas-platform.target
systemctl --user is-active realsas-platform-{postgres,temporal,migrate,api,control,engine}.service
journalctl --user -u realsas-platform-control.service -n 50 --no-pager
```

Use `current/bin/realsasctl` for the normal research/product commands in
`EXECUTION_LANES.md`. Worker source versions must match the selected release;
install a reviewed new main version before executing a changed implementation.
The deployment environment is a pinned CPU adapter environment, not a substitute
for historical model training environments or a proven complete inference service.
Native Runtime player installation and the latest Knight qualified inputs remain
separate readiness requirements; active services do not imply render readiness.
Shared CPython builds require an explicit base-library path in the service
environment. The installer discovers it from the deployed interpreter and tests
Engine imports with no inherited Actions environment. Readiness requires every
managed unit to be active; `systemctl is-active`'s multi-unit exit code alone is
not an all-services health proof. Failed startup prints bounded managed journals.
Use `current/bin/realsas-python` for operator Python utilities: it supplies the
same base-library path outside Actions and preserves command arguments exactly.

## Scope and upgrade boundary

The smoke uses an actual API request, transactional outbox, Temporal workflow,
Python Engine execution and Registry CAS commit. It rewires a two-node research
network to run only SourceLicense; then removes its independent SourceBytes
sibling and proves exact StageResult artifact reuse. It never mints a
ProductRevision. A second smoke may reuse previously sealed engineering output;
the receipt records whether the first target really executed or was already warm.
This is not model inference, Knight mechanics, appearance or rendering proof.

Upgrades preserve all data and retain the previous code-release location. They
refuse unmanaged roots/units, occupied foreign ports and any OPEN Attempts;
operators must drain or resolve those Attempts explicitly. Package builds and
checks happen before stopping live services. Migrations are forward-only during
installation. Do not blindly roll back code after a schema change: review migration
compatibility first. Automatic rollback or crash-recovery of a partially switched
deployment is not implemented; failed startup reports an explicit operator gap
without deleting data. A partially initialized PostgreSQL directory also requires
inspection; it is never silently replaced.

Source inputs remain honest raw imports. Latest Knight NPZ/result JSON is not a
qualified Stage19/28/32 contract by itself, and old frozen CAA/runtime packages
cannot be relabeled as the latest carrier-native result.
