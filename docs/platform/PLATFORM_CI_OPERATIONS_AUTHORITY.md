# Canonical Platform CI / runner / GC operations

Source of operational authority: `canonical/PLATFORM_CI_OPERATIONS_AUTHORITY_V1.json` on `main`; machine enforcement: `tests/repository/test_platform_ci_operations_authority_v1.py`. Do not replace or fork this contract when starting a new chat, branch, agent, or research apparatus.

This policy reconciles #65 (`1888d91`: hosted CPU, exact hydration and bounded cache GC), `eaa9831` (PR cancellation), #378 (`9f9f536`: exact-main manual WSL installation), and `f293d96`/`4634e56` (cache GC and tests).

- GitHub-hosted public CI executes CPU/source contracts, not physical WSL installation or scientific claims.
- WSL `realsas-wsl-1660ti` runs explicitly authorized native-host deployment / witnesses, from an exact pinned main SHA; jobs require the named self-hosted labels. Never convert a code PR check into physical deployment.
- Colab/Notebook is the default heavy learned research compute lane, with strict immutable source/input/checkpoint receipt verification and Go Platform import.
- Go/Temporal owns scientific workflow execution state. GitHub Actions cannot mint Attempt, ProductRevision, or evidence authority.
- Local persistent cache GC belongs to `tools/platform_cache_gc.py` and tests; it protects active locks, disk floor, SHA256 hydration and atomic environment install. It is not a GitHub Actions run-deletion service.

## Enforced workflow budget and changes

The canonical workflow directory is a controlled enforcement surface, currently hundreds of workflows. The global source budget caps workflow file count at 430 and tests PR cancellation policy. A new workflow or execution lane requires a deliberate policy + test change, not a chat-specific workaround. A particular product integration may impose a stricter subordinate court budget; RSS.vNext's ten fast / sixteen full gate policy is `canonical/CI_EXECUTION_AUTHORITY_V1.json` on PR #448.

Review PR workflow sources before shipping them: path filters must be meaningful; specific CPU gates remain hosted in the public repository; frozen experiments should be `workflow_dispatch` only. Treat workflow PASS as narrowly-scoped code-contract evidence.

## GitHub queued workflow-run incident

On 12 Nov 2026, PR #448 showed 543 old `queued` **run records** from 21 previous SHAs; sampled records had zero **jobs**. Current head ran successfully. Cancellation and force-cancellation were refused and a single controlled DELETE probe returned `403 Could not delete the workflow run`. This is not evidence of a WSL queue or OOM. We cannot claim the backend records were cleaned.

`ci_queue_sentinel_v1.yml` monitors read-only once daily or on explicit dispatch, classifies stale queued records, samples job counts and fails with a clear Support escalation marker. Do not repeat bulk cancel/delete, restart WSL for zero-job records, touch other PRs, or erase successful CI history. GitHub Support must investigate inaccessible zombie records using run IDs and REST error responses. Monitor until actual queue recount reaches zero.

## Operational acceptance

A runner fix requires confirming actual **jobs** assigned to a matching offline or saturated runner (not zero-job workflow runs). Obtain runner status, service health, RAM/swap/disk and OOM events before changes. WSL redeploy remains explicit. A full release/pr merge must validate latest source SHA rather than rely on green checks from old commits.
