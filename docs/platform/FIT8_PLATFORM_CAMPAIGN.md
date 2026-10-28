# Fit8 platform-native full-E2E campaign

Fit8 is a platform experiment, not a notebook-owned pipeline. The eight heterogeneous subjects enter the same released 43-stage Compiler DAG at `01_SOURCE_BYTES_SEALED` and target the same product-closure stage as the Knight line. Knight remains a separate regression member and is not a Fit8 training subject.

## Authority boundary

The platform owns Subjects, sealed SubjectInputs, EngineRelease identity, ResearchAttempts, stage scheduling, artifact lineage and PASS/FAIL-producing courts. A100/Colab is only an external GPU worker for preregistered learned stages. Returned checkpoints and receipts are imported as exact bytes and then validated by the consuming stage; import or training loss cannot issue qualification.

The campaign coordinator adds no scientific stage and owns no qualification. It groups ordinary subject-bound ResearchAttempts under one release and one final target, submits their compile commands, and reports whether all cohort members plus the regression set reached that target. `fit8_pass_claimed` remains false in campaign status.

## Required campaign

The canonical member contract is `canonical/FIT8_PLATFORM_CAMPAIGN_V1.json`.

Cohort: Snake, Bat, T-Rex, Horse, Spider, GreenBlob, Shark and George. Regression: Knight.

Every cohort member must independently traverse source/import/rest-pose mechanics, camera contract, exact source observations, normalization, learned stages, canonical M, appearance, mechanics, presentation, motion, runtime and closure. A subject is not admitted by another subject's result.

## Failure semantics

A failure is evidence about the generic system. Do not replace the subject with an easier witness and do not add subject-name production branches. Route the failure to the owning generic module, make the smallest general correction, and rerun the affected closure across the campaign. Knight regression is preserved at the same time.

A change that fixes one subject but breaks another has not closed the generic invariant. Campaign rollup is complete only when all eight cohort Attempts and the Knight regression Attempt complete the same final closure target without hidden failed members.

## Operator surface

### Preferred cohort entrypoint

Do not hand-author per-subject `campaign-start` / `campaign-run` bindings for a sealed cohort. The generic bootstrap executor is the operator entrypoint:

```bash
python tools/platform_cohort_bootstrap_v1.py \
  --descriptor canonical/FIT8_PLATFORM_BOOTSTRAP_V1.json \
  --repo /path/to/canonical/main \
  --platform-root /home/monster/realsas_platform \
  --authority-root /home/monster/realsas_authority \
  --out /home/monster/realsas_authority/campaign_bootstrap/FIT8_BOOTSTRAP_RESULT.json
```

The descriptor is data. The executor contains no Fit8 subject names and no fixed cohort cardinality. A later Fit40/LOFO cohort must reuse the same executable with a different sealed descriptor rather than adding a new subject-specific bootstrap script.

Before campaign mutation the executor verifies the external bundle receipt, exact member bytes, path safety, roster identity, the shared IRIS cohort/policy binding and every subject's observation-binding identity. It then uses the public Platform API to ensure Subjects, import raw manifest-section bytes, seal SubjectInputs, create one subject-agnostic RESEARCH EngineRelease, start the campaign, write campaign-bound run manifests and submit the campaign. Raw hydration/import never mints qualification.

`--no-submit` performs the same fail-closed input/release/campaign preparation and writes the exact run bindings without submitting execution. It is an operator preflight, not scientific PASS.

### GitHub cohort operator

For remote research operation, `.github/workflows/platform_cohort_operator.yml` is the generic self-hosted entrypoint. It accepts a repo-relative bootstrap descriptor and an explicit `submit` boolean. The default is preflight (`submit=false`); campaign execution is never started merely because canonical main was deployed.

The workflow requires `refs/heads/main`, verifies that the self-hosted platform deployment is the exact selected main SHA, rejects absolute/parent-traversing descriptor paths, verifies the cohort storage remote, invokes the same generic bootstrap executor, and preserves the exact bootstrap receipt as a workflow artifact. It contains no subject-name execution branch. Fit40/LOFO must use the same workflow with a different sealed descriptor.

This GitHub surface is only a remote operator for the Platform API. GitHub Actions does not own Attempts, stage qualification, campaign state, or model authority.

### Lower-level campaign API/CLI

The durable coordinator remains available directly when the Subject/Input/Release bindings already exist:

```bash
realsasctl campaign-start --repo /path/to/canonical/main --request campaign_start.json
realsasctl campaign-run   --repo /path/to/canonical/main --request campaign_run.json
realsasctl campaign --id <CAMPAIGN_UUID>
```

`campaign-start` derives the product-pass target from the candidate released DAG and rejects a graph whose product target does not transitively close the full released DAG. It creates one ordinary ResearchAttempt per member.

That is the default whole-system path. A finite source-apparatus campaign may
instead supply `target_stage_id` for an explicitly released research component.
Every member uses that same immutable target and all released source-entry roots
must remain in its ancestor closure. The sealed target is part of campaign
identity; changing it creates a distinct campaign. This supports measurement
completion without leaving Attempts idle at a later training barrier.

Status reports `execution_scope` and `goal_complete` for that bounded operation.
`full_closure_complete` is true only for a completed product-closure target;
component completion cannot issue FIT8 PASS or ProductRevision. The canonical
FIT8 whole-system descriptor keeps its existing final product target.

`campaign-run` requires one sealed SubjectInput and canonical run-manifest/ledger path for every member. It submits the same final target for every member using deterministic campaign-scoped command identities. The normal single-subject `research-run` path continues to require an agent session; the campaign path is the durable multi-subject coordinator and pins the API's deployed canonical code SHA.

## Learned-stage handoff

For shared cohort models, the common checkpoint is a shared immutable artifact but each subject's consuming stage remains subject-bound and must validate its own upstream bindings and downstream courts. IRIS Rev3 uses one shared model/optimizer, one global optimizer update after averaging all eight subject gradients, and one exposure per subject per global update. The Compiler validator rejects the superseded round-robin exposure contract.

The EngineRelease declares required and optional external-evidence roles. Missing required learned-stage evidence produces `WAITING_EXTERNAL_EVIDENCE` without minting a failure or qualification; an authorized optional shared payload may be bound where the experiment requires it without forcing the same payload onto an independent regression member.

The same platform pattern should be used for TESSA, AXIS and MIRA as their cohort training contracts are sealed. Shared learned parameters do not make downstream Compiler qualification shared.
