# Local developer release disk guard and WSL D: migration (2026-11-07)

## Incident (actual user-provided workstation measurements)

- Windows C:\ was completely full (99 MiB available), while WSL ext4 mounted at `/` incorrectly appeared to offer 666 GiB of usable capacity from its maximum virtual allocation.
- `/home/monster/realsas_platform/releases` contained **157 immutable installations / ~200 GiB**.
- One active installation occupies ~1.8 GiB: source ~318 MiB, bin ~229 MiB, venv ~1.2 GiB.
- Persistent `realsas_platform/artifacts` ~4.6 GiB, `data` ~214 MiB, and `realsas_authority` ~54 GiB are separate scientific/product state; **do not delete**.
- `tools/platform_deploy.py` previously created an always-new full source+venv+binary directory for each manual deployment and had no count/disk safety cap.

## Changes

## Prevention at the source, not an arbitrary workflow blockade

* Identical canonical \`code_sha\` already installed and healthy: verified \`NOOP_IDENTICAL_RELEASE\`, no copied source, no venv, no rebuilt Go binaries. If the same-SHA installation is unhealthy, report a repair-needed condition rather than silently multiplying copies. Explicit \`--force-rebuild\` is recovery-only.
* Distinct code revisions with identical *complete recursive requirements* and Python patch version: \`tools/platform_shared_python_env_v1.py\` references one fingerprint-keyed immutable Python environment under \`shared_python_envs\`. The \`release/venv\` symlink points to it. Existing \`platform_ci_bootstrap.ensure_venv\` provides persistent locks, atomic construction and cache receipts. This developer service cache is not a training environment; old releases remain replayable.
* 12-release threshold: **advisory warning, not a mandatory pause**. Legitimate new releases can proceed.
* **Hard** preflight stops only when C: is below 10 GiB or Linux filesystem below 8 GiB. The C: condition is conservative OS safety even if WSL is backed by D:, and must be separately reviewed if that becomes intrusive.
* Unique source commits still have distinct source/bin snapshots, so the count can grow absent periodic approved retention. This PR does **not** silently auto-prune historical releases or interrupt Platform services for cleaning.

`tools/platform_release_retention_v1.py` provides a **read-only-by-default** old-release plan. `--apply` requires a typed confirmation token, inactive `realsas-platform.target` **and every managed service unit**, absence of live process references in `/proc`, no mountpoint within deleted release trees, active/previous deployment marker consistency, and keeping at least five newest release directories. Default retention keeps six newest PLUS active, previous, and top-level release symlink references. `shutil.rmtree` is limited to strict release-name directories under the verified deployment root. It does not traverse or delete `artifacts`, `data`, `receipts`, `realsas_authority` or the working Git repo.

**Human review still required** before deleting historical release snapshots: ensure historical run/replay dependencies and current DB Attempts do not require them. The maintenance tool deliberately refuses to run while the Platform runs.

## Operator sequence (no automatic workstation access from ChatGPT)

1. Confirm actual Windows D: drive filesystem is **NTFS**, free capacity is larger than current WSL VHD byte size with margin, verify WSL supports native `--manage --move`. A verified backup of important research state is strongly recommended. Do not unregister the distro.
2. Check zero in-flight Platform `attempts` before intentionally stopping Platform services or shutting down WSL; if unknown, do not stop mid-experiment.
3. If supported, use PowerShell after `wsl --shutdown`: `wsl --manage <actual-distro-name> --move D:\WSL\<actual-distro-name>`. Confirm distro is registered and all prior state accessible. This moves the Linux VHD; moving random source folders to `/mnt/d` would change Unix filesystem contracts and is **not** the plan.
4. In the moved Ubuntu, run `python3 tools/platform_release_retention_v1.py --retain 6` to inspect candidates. Ensure exact authority links and whether old releases are required by any execution receipts. When in-flight attempts are zero, stop managed Platform only and then explicitly apply `--apply --confirm PRUNE-REALSAS-OLD-RELEASES`. Default never deletes.
5. Verify `readlink -f ~/realsas_platform/current`, source/bin/venv, service health, `df -h / /mnt/c /mnt/d`, and separate `realsas_authority` hashes.

Never run `wsl --unregister` or manually move/delete `ext4.vhdx`. If native `--move` is unsupported, stop and plan a separately verified export/backup/import migration. **This PR changes code only**, not the operator machine; merging an approved guard deployment is required for protection to become active. Existing 157 release directories will warn but will not block a distinct authorized deployment while actual disk capacity is safe.

## Tests

Synthetic tests cover nonblocking count warnings, actual-volume exhaustion, identical-SHA no-op, cross-commit shared immutable Python env, active/previous retention, and explicit-safe delete scopes. GitHub-hosted workflow `platform-release-disk-safety-v1` does not request a self-hosted runner.
