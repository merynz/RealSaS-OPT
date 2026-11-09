# RSS.vNext — Agent operation and live completion ledger

**Design owner:** frozen RSS.vNext architecture ([PR #397](https://github.com/merynz/RealSaS-OPT/pull/397))  
**Live execution owner:** [Master GitHub issue #399](https://github.com/merynz/RealSaS-OPT/issues/399), with 25 independent task issues #400–#424  
**Index:** \`canonical/RSS_VNEXT_WORK_LEDGER_INDEX_V1.json\`  
**Readable live checkboxes:** \`python tools/ledger/rss_vnext_status.py\`

## First-read protocol (any future agent)

1. Read \`AGENTS.md\` and \`CURRENT_STATE.md\` from the *actual checked out, current* source revision.
2. Read frozen design: \`canonical/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_V1_20261109.json\` and \`docs/research/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_20261109.md\`.
3. Read \`canonical/RSS_VNEXT_WORK_LEDGER_INDEX_V1.json\`; refresh the **actual** live GitHub issue states. Never use old model memory, stale Markdown checkboxes, or the index file as operational status.
4. Inspect related PRs/current Actions/Attempt before starting work. In particular, Truth Foundry shards, current Corpus370 campaigns and architecture development may be concurrent.
5. Read the task issue objective, dependencies and **definition of done**, then comment with scope/commit/owner and any conflicting concurrent work.

## Get the live ledger

\`\`\`bash
# May read public repo without token; token avoids rate limits
python tools/ledger/rss_vnext_status.py

# Exact live JSON for agents and automation
GITHUB_TOKEN="$GITHUB_TOKEN" python tools/ledger/rss_vnext_status.py --format json --output /tmp/rss-vnext-ledger.json

# Offline fixtures/tests only; explicitly flagged NOT LIVE
python tools/ledger/rss_vnext_status.py --snapshot /tmp/mock-issues.json --format md
\`\`\`

The tool is **read-only**, uses GitHub Issues REST GET and renders **[x] only when** (a) issue is \`closed\`, (b) \`state_reason=completed\`, and (c) an explicit completion receipt is present with \`## Completion Evidence\`, \`RSS_LEDGER_COMPLETION_V1\`, a source or commit SHA-40/SHA-64 and a relevant GitHub/Drive URL.

A syntactically valid receipt is **not** itself proof of scientific admission; the closing agent must independently verify the actual source/court/receipt under exact lineage. GitHub native issues remain the source of live work state, and compiler/Platform artifacts remain the source of actual experiment/product scientific authority.

If an issue closes without a valid receipt it displays \`CLOSED_UNVERIFIED\` **not [x]**. Closing it as \`not_planned\` also does not mark it done. Reopen on invalidated evidence.

## Task lifecycle

- \`OPEN\`: work not yet fully qualified. A PR or green unit test is partial progress, not completion.
- \`WORKING\`: keep the issue **open**, claim the owner in comments/assignee, link exact branch, code SHA, source provider cohort and current Attempt.
- \`BLOCKED\`: keep issue **open**, describe exact cause, missing source-right court, runner/Drive availability or dependency. Never infer source failure from environment failure.
- \`COMPLETE_WITH_RECEIPT\`: all specified acceptance gates, evidence and independent review are present; add a completion block, then close with reason \`completed\`.
- \`CLOSED_UNVERIFIED\`: closed without required receipt; investigate and reopen or repair actual evidence.
- \`CLOSED_NOT_PLANNED\`: intentionally retired after documented scope change, never counted as completed.

### Required completion comment/body section

\`\`\`markdown
## Completion Evidence
RSS_LEDGER_COMPLETION_V1

- Scope: <exact source/model/product capability; explicitly state exclusions>
- Implemented: <PR, reviewed and merged commit SHA40>
- CI: <GitHub Actions run URL; test name and outcome>
- Source/court: <independent run ID / artifact URL / SHA64, or NOT_APPLICABLE_FOR_CODE_ONLY>
- Caller/consumer: <exact typed reader/writer and integration validation>
- Limitations: <remaining unknown, mask and failures>
- Reviewer: <independent acceptance and how verified>
\`\`\`

When a task has real science admission requirements, \`NOT_APPLICABLE_FOR_CODE_ONLY\` cannot satisfy it. Code-only tasks do not receive corpus/model/product admission from a CI pass.

No agent may mark an issue complete merely by claiming a percentage or publishing a document.

## Why the ledger is per GitHub issue rather than a single mutable JSON task status

Each issue is a single independently claimable and closable work item with a durable timeline. Multiple agents can close different issues without racing on a shared JSON file or force-overwriting a Markdown checklist. \`RSS_VNEXT_WORK_LEDGER_INDEX_V1.json\` is the static catalog of IDs, dependencies and issue numbers, not a live status mirror. To add a new task, create a unique issue, update the versioned index and add a test in a reviewed PR. To subdivide a task, link successors and preserve the original acceptance gate; do not close a parent prematurely.

The master issue #399 intentionally contains a static **index** with \`[ ]\` task links. The dynamic, accurate \`[x]\` view is produced by the CLI (and after workflow deployment by GitHub Actions issue-event summaries), not by editing #399 after each concurrent issue update.

## Source-of-truth and historical separation

**Design freeze ≠ code completion ≠ source rights ≠ training admission ≠ unseen result ≠ product activation.** Read actual code and live registry, not a plan's declared readiness. Models are still reviewed against the older working V2 compatibility path as an implementation baseline.

- **IRIS:** Geometry/Semantic/Graphics shared 3D evidence.
- **TESSA:** animation-ready structural mesh/attachment proposal, differentiable surface bridge.
- **AXIS:** fully editable RigProgram beyond bare skeleton, learned control targets/constraints.
- **MIRA:** exact-mesh SkinTokens/sparse weight expert sharing AXIS causal state.
- **Compiler:** generic verified solver execution, CAA/presentation, bounded repair only.
- **Truth Foundry:** real source identity/rights, observed/derived/unknown, per-head masked targets, lineage-safe split; no privileged teacher output at inference.
- **Platform:** Go Attempt/EngineRelease/Artifact Registry retains actual execution/release state. A GitHub issue is NOT a scientific or product authority.

## Work order and noninterference

Priority is **source qualification and six-head truth** in parallel with architecture contracts and necessary implementation preparation. The architecture design is frozen, but the large dataset, existing V12/V13/V14 source work, Core370 repair, and live Actions may remain unfinished. Do not restart active source runs merely because a ledger item is open.

Existing PR #397 (architecture freeze), #398 (private derivative recovery), #327 (Corpus370) and older independent scientific records are references; their states must be rechecked rather than assumed.
