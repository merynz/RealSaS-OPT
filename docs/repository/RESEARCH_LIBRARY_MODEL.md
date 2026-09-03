# RealSaS Research Library Model

RealSaS-OPT is a research repository with a continuously upgraded executable mainline. The repository must support aggressive experimentation **without making the current system hard to find**.

## 1. Four semantic zones

### Mainline library

`models/`, `compiler/`, `runtime/`

These directories answer: **"What implementation do we currently use?"**

A current executable mechanism gets one obvious semantic home here. Mainline code may evolve, be replaced, or be demoted when later evidence falsifies it.

### Labs

`experiments/`

This directory answers: **"What are we currently testing, comparing, falsifying or trying to improve?"**

Experiments may be messy internally, dated for provenance, and may compare multiple candidate implementations. They do not become current merely because they are newer.

### Decision/evidence library

`canonical/`

This directory answers: **"Why is the mainline currently shaped this way, and what was actually proven?"**

It stores architecture contracts, preregistrations, closures, source seals, promotion decisions and current authority ledgers. Canonical documents describe authority; executable code still lives in its semantic mainline home.

### Provenance reserve

`historical/`

This directory answers: **"What superseded or external historical authority might still matter for source-diff or restoration?"**

It must not become a second executable mainline.

## 2. One-way promotion model

The normal lifecycle is:

```text
question
  -> experiments/<bounded question>/
  -> evidence + closure
  -> canonical promotion decision
  -> audited implementation enters/replaces mainline semantic home
  -> mainline regression / E2E
  -> repository map + state ledger updated
```

If a later experiment disproves or improves a mainline mechanism:

```text
new experiment evidence
  -> explicit replacement decision
  -> mainline implementation replaced or amended
  -> old behavior retained by git history / canonical provenance
  -> regressions rerun
```

Do not keep two active implementations merely to preserve history. Git history is the default archive. A physical historical copy is justified only when byte authority/source-diff requires it.

## 3. Import direction

Hard repository hygiene target:

```text
experiments  -> may import mainline
models       -> may import stable Compiler proposal/IR contracts
compiler     -> may consume typed model outputs, but must not import dated experiment implementation
runtime      -> consumes proof-gated Compiler export
mainline     -X-> experiments
```

Temporary violations discovered during migration must be recorded and removed by a named promotion task; they must not become permanent architecture.

## 4. Mainline does not mean scientifically final

A mainline implementation means:

> "best currently authorized implementation for the current architecture and gate set"

It does **not** mean eternally canonical, publication-final, globally generalizing, or immune to replacement.

This distinction is essential for RealSaS because IRIS, Geppetto, SkinFieldCodec, Arachne, compiler numerics, proof/repair and runtime integration may all be upgraded as new experiments close.

## 5. Human navigation invariant

A reader who only opens the root index must be able to answer:

1. What are the active learned models?
2. What are the active Compiler layers?
3. What runtime exists?
4. What is currently blocked/passed?
5. Where are experimental alternatives?
6. What historical authorities remain candidates for promotion?

If any of these requires guessing from dates or searching the whole repository, repository organization is considered incomplete.

## 6. Model organization

`models/` is organized by semantic subsystem, not by experiment date:

- `models/iris/`
- `models/geppetto/`
- `models/skin_field_codec/`
- `models/arachne/`

Each model eventually owns its current inference implementation plus model-local training/evaluation/test code. Dated experiment trees remain evidence and candidate development spaces.

## 7. Compiler organization

The Compiler has two categories:

- `compiler/realsas_compiler_core/` — canonical typed authority and deterministic compilation/qualification;
- `compiler/realsas_compiler_services/` — subordinate promoted mechanisms such as proof/repair/export/numerical services.

The current core is still too flat for final library-quality navigation. It will be normalized **after dependency audit**, using compatibility exports during movement so semantics and provenance remain stable. Intended semantic groups are:

```text
contracts / IR
substrate + local geometry
rig qualification
skin qualification
mesh + mesh-skin binding
appearance + completion
motion
product assembly + lineage
proof binding
```

No file is moved only for aesthetics. A move must establish one semantic owner, preserve public contracts, and pass regressions.

## 8. Upgrade dispositions

Every candidate source file considered for mainline receives one of:

- `PROMOTE_CORE`
- `PROMOTE_SERVICE`
- `PROMOTE_MODEL_CORE`
- `PROMOTE_MODEL_TRAINING`
- `PROMOTE_MODEL_EVALUATION`
- `KEEP_EXPERIMENT_ONLY`
- `KEEP_HISTORICAL_ONLY`
- `SUPERSEDED`
- `DROP_DUPLICATE`
- `SOURCE_DIFF_REQUIRED`

A directory is never promoted wholesale merely because one file in it is useful.

## 9. Current restoration constraint

During Compiler/runtime restoration, library normalization and historical promotion are coupled but not conflated:

- restore only audited mechanisms;
- place restored mechanisms in their current semantic owner;
- do not resurrect historical monoliths;
- do not postpone obvious ownership cleanup indefinitely;
- do not perform a global architecture refreeze until restoration + source ownership normalization regressions close.
