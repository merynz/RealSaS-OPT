# Local developer release disk guard and WSL D: migration (2026-11-07)

## Incident (actual user-provided workstation measurements)

- Windows C:\ was completely full (99 MiB available), while WSL ext4 mounted at `/` incorrectly appeared to offer 666 GiB of usable capacity from its maximum virtual allocation.
- `/home/monster/realsas_platform/releases` contained **157 immutable installations / ~200 GiB**.
- One active installation occupies ~1.8 GiB: source ~318 MiB, bin ~229 MiB, venv ~1.2 GiB.
- Persistent `realsas_platform/artifacts` ~4.6 GiB, `data` ~214 MiB, and `realsas_authority` ~54 GiB are separate scientific/product state; **do not delete**.
- `tools/platform_deploy.py` previously created an always-new full source+venv+binary directory for each manual deployment and had no count/disk safety cap.

## Changes

`tools/platform_release_budget_v1.py` is executed in `tools/platform_deploy.py::preflight`, **before writing a new release**. It stops installation if:

- **12** matching historical release directories already exist (prevents another 157-copy runaway);
- Windows-mounted C:\ is under **10 GiB free**, even if Linux ext4 `df` lies about actual host free capacity;
- Linux deployment filesystem is under **8 GiB free**.

`tools/platform_release_retention_v1.py` provides a **read-only-by-default** old-release plan. `--apply` requires a typed confirmation token, inactive `realsas-platform.target` **and every managed service unit**, absence of live process references in `/proc`, no mountpoint within deleted release trees, active/previous deployment marker consistency, and keeping at least five newest release directories. Default retention keeps six newest PLUS active, previous, and top-level release symlink references. `shutil.rmtree` is limited to strict release-name directories under the verified deployment root. It does not traverse or delete `artifacts`, `data`, `receipts`, `realsas_authority` or the working Git repo.

**Human review still required** before deleting historical release snapshots: ensure historical run/replay dependencies and current DB Attempts do not require them. The maintenance tool deliberately refuses to run while the Platform runs.

## Operator sequence (no automatic workstation access from ChatGPT)

1. Confirm actual Windows D: drive filesystem is **NTFS**, free capacity is larger than current WSL VHD byte size with margin, verify WSL supports native `--manage --move`. A verified backup of important research state is strongly recommended. Do not unregister the distro.
2. Check zero in-flight Platform `attempts` before intentionally stopping Platform services or shutting down WSL; if unknown, do not stop mid-experiment.
3. If supported, use PowerShell after `wsl --shutdown`: `wsl --manage <actual-distro-name> --move D:\WSL\<actual-distro-name>`. Confirm distro is registered and all prior state accessible. This moves the Linux VHD; moving random source folders to `/mnt/d` would change Unix filesystem contracts and is **not** the plan.
4. In the moved Ubuntu, run `python3 tools/platform_release_retention_v1.py --retain 6` to inspect candidates. Ensure exact authority links and whether old releases are required by any execution receipts. When in-flight attempts are zero, stop managed Platform only and then explicitly apply `--apply --confirm PRUNE-REALSAS-OLD-RELEASES`. Default never deletes.
5. Verify `readlink -f ~/realsas_platform/current`, source/bin/venv, service health, `df -h / /mnt/c /mnt/d`, and separate `realsas_authority` hashes.

Never run `wsl --unregister` or manually move/delete `ext4.vhdx`. If native `--move` is unsupported, stop and plan a separately verified export/backup/import migration. **This PR changes code only**, not the operator machine; merging an approved guard deployment is required for protection to become active. Existing 157 release directories will cause the new preflight to block until safely cleaned.

## Tests

Synthetic `tests/platform/test_platform_release_budget_v1.py` and `tests/platform/test_platform_release_retention_v1.py` check count/space cap, zero current/previous loss, fake source/artifact preservation, and fail-closed destructive guard. No workstation data or real deletion in CI. GitHub-hosted workflow `platform-release-disk-safety-v1` does not request a self-hosted runner.
