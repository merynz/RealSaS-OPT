# RealSaS — Investor Demo Run Ledger V1

Append-only operational record for `demo/investor-single-specimen-e2e`.

## 2026-09-01 / D0

### Branch genesis

- base repository: `merynz/RealSaS-OPT`
- canonical base branch: `main`
- base commit: `bba22f25313b32aa4f8c2fbe418e3f2590198f38`
- demo branch: `demo/investor-single-specimen-e2e`
- branch authority: experimental side branch, not main continuation authority
- specimen selected: `NO`
- optimizer steps: `0`

### User authorization recorded

The user explicitly authorized:

- a separate investor-demo branch;
- generic implementation of IRIS, Geppetto, and Arachne before specimen selection;
- subsequent single-specimen memorization/overfit for all three learned modules;
- teacher truth during fitting/evaluation;
- final image-only inference with no teacher/source-rig/source-weight injection.

The user explicitly rejected specimen-specific Python hacks and required specimen selection only after the generic models are ready.

### Base technical audit

Observed on main:

- current typed Compiler skeleton qualification exists;
- current typed Compiler skin qualification exists;
- typed MWB/product V2 lineage exists;
- Geppetto learned predictor absent;
- Arachne learned predictor absent;
- current IRIS V2 learned predictor incomplete/not yet implemented as a full executable learner;
- older IRIS controlled learner exists as historical scaffold but has an obsolete direct-P/N output boundary;
- native runtime source not present as current executable source; exact source remains historical external authority.

### Governance artifacts created

- `README.md`
- `DEMO_ACCEPTANCE_CONTRACT_V1.md`
- `DEMO_ARCHITECTURE_READINESS_V1.md`
- `DEMO_TASK_BOARD_V1.md`
- this ledger

### Current execution state

`ARCHITECTURE_IMPLEMENTATION_OPEN__SPECIMEN_SELECTION_FORBIDDEN__OPTIMIZER_STEPS_0`

Next action: implement generic branch-local model/adapter contracts beginning with shared conditioning/checkpoint/provenance infrastructure, then IRIS -> Geppetto -> Arachne.
