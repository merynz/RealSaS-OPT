# RealSaS CI / execution authority (v1)

**Normative machine contract:** `canonical/CI_EXECUTION_AUTHORITY_V1.json`  
**Enforcement:** `tests/repository/test_ci_execution_authority_v1.py` plus `tests/repository/test_self_hosted_execution_policy.py`.

This is not a replacement for the Go Platform, an independent science admission rule, or a new runner arrangement. It reconciles the existing hosted-first, local-Witness and bounded-GC decisions from #65 and #378.

## Execution ownership

- **Code and synthetic CPU PR CI:** GitHub-hosted Linux while repository is public. Its success certifies only tests on an exact SHA.
- **Physical WSL / GTX / deployment:** explicit, main-SHA-bound `workflow_dispatch` and selected local/native witness. Never treat hosted contract PASS as physical deployment.
- **Large learned research:** notebook/Colab, then exact content-hash, source-lineage and receipt verification through Go Platform. Keep GitHub Actions out of scientific state ownership.
- **Durable workflow, artifacts and product:** Go control plane, EngineRelease, Attempt, ProductRevision and CAS, not git branches or Actions history.

## PR load budget

For RSS.vNext's current 26 named checks, preserve ten fast automatic checks on `synchronize`, and run sixteen full courts on PR open/reopen/ready transition or explicit dispatch. The full court is **not removed**. After a new commit on an already-ready PR, the full court's former successes are stale. Re-dispatch the full workflows on the PR branch (or move Draft → Ready to rerun) and use the `ci_pr_full_admission_v1.yml` manual admission check on the exact latest SHA.

Do not interpret a PR-wide old green status as evidence for a new head. A branch may not merge under this operating policy without the full latest-head admission report. Repo ruleset protection, if desired, requires an administrator to configure required status checks: the source-code test alone cannot install GitHub branch protection.

Do not add broad PR synchronization triggers or new automatic checks without amending the named execution-budget policy and test together. Avoid a commit storm: batch logical changes, run narrow tests locally, then push reviewed units.

## Queue incident: 12 November 2026

543 GitHub Actions **workflow runs**, not 543 runnable jobs, remained `queued` from 21 historical PR #448 commits in a thirty-minute interval on 11 November. Every sampled stale run had **0 jobs**. Latest PR #448 SHA `75a3e4d` had 26 successful checks; the zero-job cleanup attempt failed with HTTP 403 `Could not delete the workflow run`. No evidence yet implicates WSL runner availability. Do not restart the runner as a supposed fix for zero-job records.

`ci_queue_sentinel_v1.yml` is a read-only, hosted monitor that detects stale queued records. If the GitHub scheduler refuses cancellation/force-cancellation/deletion, report the exact API responses and open a **GitHub Support** ticket. Do not bulk-delete Actions history, delete unrelated PR runs or repeat hundreds of failing API requests. The repo's GITHUB_TOKEN cannot override a GitHub service error. The failed `rss_vnext_pr448_queue_drain.yml` auto-cleanup workflow was deliberately removed; it is not a repair mechanism and must not reappear. A clean queue requires GitHub to clear these orphan records or a verified supported operation; do not report success until an API recount returns zero.

## GC is a different operation

Local persistent cache GC lives in `tools/platform_cache_gc.py`, enforced by `tests/repository/test_platform_cache_gc.py`. Entries follow temporary/incomplete → age expiry → LRU watermark; preserve active locks, atomic installed environments, disk safety floor, immutable artifact identity and verified hydration. **Never** apply local cache GC to GitHub Actions records or delete CAS authority based on an Actions status.

## Operator checklist

1. Check PR current SHA, queue counts by workflow, statuses and representative job counts.
2. Separate GitHub scheduling anomalies from real queued **jobs** waiting on `[self-hosted, linux, x64, realsas]`.
3. If there are actual self-hosted jobs, inspect online/idle labels, systemd user service, host RAM/swap, disk and OOM logs before a controlled restart.
4. For abnormal zero-job records, use read-only sentinel and GitHub Support escalation; retain current-head checks.
5. Before any merge, obtain fresh full-court PASS receipts and exact-head admission; never treat previous head/legacy experiments as authorization.
