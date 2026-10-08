# Contributing to RealSaS

RealSaS is a proprietary research and product repository. Access to the repository does not grant a license to use its contents. See `LICENSE`.

## Before changing code or scientific authority

Read, in order:

1. [CURRENT_STATE.md](CURRENT_STATE.md)
2. [AGENTS.md](AGENTS.md)
3. [Execution lanes](docs/platform/EXECUTION_LANES.md)
4. [System ownership](SYSTEM_INDEX.md)
5. [Canonical index](canonical/README.md)

Use a short code PR into main. Start component experiments as Go research
Attempts against immutable baseline/candidate EngineReleases. Select a model or
Compiler target, inspect the affected DAG closure and retain independent
artifacts. DAG changes belong to a new release; do not mutate prior execution
state or accumulate research state on a long-lived branch.

Do not infer authority from branch recency, workflow color, file date, or a detached report.

## Change classes

### Ordinary implementation / maintenance

- keep one obvious semantic owner;
- preserve typed Compiler boundaries;
- add or update regression tests;
- do not import dated experiment code into promoted `models/`, `compiler/`, `product/`, or `runtime/` source;
- use current repository environment profiles under `requirements/` for local/CI checks.

### Scientific experiment

A scientific intervention must have, before the first optimizer step when applicable:

- explicit question/hypothesis;
- frozen inputs and source identity;
- preregistered arms, gates and interpretation rules;
- fail-closed preflight;
- deterministic seed/schedule policy where required;
- result/provenance artifact identities;
- explicit non-claims.

A scientific PASS does **not** promote itself.

### Promotion / refreeze

Promotion requires a separate transaction reconciling:

- promoted source path and byte/source identity;
- scientific closure/result;
- checkpoint/result/evidence hashes;
- architecture and experiment ledgers;
- `CURRENT_STATE.md` and machine continuity state;
- provenance/supersession;
- regression/CI protection.

## Model and architecture naming

RealSaS implementation names must describe RealSaS responsibilities and mechanisms, not external project branding. External project/model names may appear in comparison, audit, bibliography, provenance or research-lineage documents. DINO/DINOv2 is an explicit upstream dependency and is exempt where its identity is technically required.

## CI / runner policy

The workflows designated as **current execution authority** run only on the local self-hosted runner:

`[self-hosted, linux, x64, realsas]`

Known runner: `realsas-wsl-1660ti`. Local execution is also permitted.

Runner policy is checked in `tests/repository/test_self_hosted_execution_policy.py`.
All workflow definitions use self-hosted execution. Frozen historical training
and recovery replays are manual reproducibility, not normal current development.
Automatic PR workflows cancel superseded heads in their workflow/PR group.

Do not add GitHub-hosted execution.

## Local checks

```bash
python -m pip install -r requirements/mainline-ci.txt
python -m pip install -r requirements/torch-cpu.txt
python -m pip install -r requirements/dev.txt
pre-commit run --all-files
python -m pytest -q tests/repository tests/models
```

Run narrower subsystem tests as appropriate. Scientific notebooks additionally require their own preregistered/preflight checks.

## Pull requests

A PR should state:

- what authority or implementation it changes;
- what it explicitly does not claim;
- tests/evidence run;
- any source/checkpoint/result hashes relevant to a scientific promotion;
- whether continuation state must change.

Do not combine unrelated scientific interventions into one causal result.
