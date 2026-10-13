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

After migrations and API deployment:

```bash
realsasctl campaign-start --repo /path/to/canonical/main --request fit8_campaign_start.json
realsasctl campaign-run   --repo /path/to/canonical/main --request fit8_campaign_run.json
realsasctl campaign --id <CAMPAIGN_UUID>
```

`campaign-start` derives the product-pass target from the candidate released DAG and rejects a graph whose product target does not transitively close the full released DAG. It creates one ordinary ResearchAttempt per member.

`campaign-run` requires one sealed SubjectInput and canonical run-manifest/ledger path for every member. It submits the same final target for every member using deterministic campaign-scoped command identities. The normal single-subject `research-run` path continues to require an agent session; the campaign path is the durable multi-subject coordinator and pins the API's deployed canonical code SHA.

## Learned-stage handoff

For shared Fit8 models, the common checkpoint is a shared immutable artifact but each subject's consuming stage remains subject-bound and must validate its own upstream bindings and downstream courts. IRIS Rev3 uses one shared model/optimizer, one global optimizer update after averaging all eight subject gradients, and one exposure per subject per global update. The Compiler validator rejects the superseded round-robin exposure contract.

The same platform pattern should be used for TESSA, AXIS and MIRA as their Fit8 training contracts are sealed. Shared learned parameters do not make downstream Compiler qualification shared.
