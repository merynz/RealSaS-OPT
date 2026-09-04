# RealSaS — Pre-FIT hardening reconciliation V2

**Date:** 2026-09-04  
**Repository:** `merynz/RealSaS-OPT`  
**Scientific optimizer steps at reconciliation:** `0`

## Purpose

This record reconciles the current pre-FIT product authority after the 2026-09-04 hardening sequence. It does **not** rewrite the experimental history.

Earlier G1/G2 work, R6/U0/U1 work, quotient experiments, behavioral panels, restoration ledgers, and other dated experiment records remain preserved evidence with the exact claim boundaries under which they were run. Their omission from a later reviewer summary does not turn them into failures or make them disappear. Conversely, historical PASS evidence is not silently promoted into current product authority when the living source has changed.

## Current authority

The current source-selection authority is:

- `canonical/ARCHITECTURE_FREEZE_V2.json`
- living-source fingerprint: `9585aa3af8975fe66af64e161b9289da02b77c012c682bca536c664ef1d271f0`
- pre-FIT selection-apparatus fingerprint: `9ed25dafe62c62c0cee69dc4878c5587366db5e296a657c9649c86bf334fecd1`
- visible canonical graph optimizer SHA-256: `b2fddb64753ca783e298be4f1078c70b9667fa9931c67de738f955976e2587c1`
- vendor execution closure raw SHA-256: `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`
- freeze candidate run: `33862732418`, `42/42 PASS`, violations `[]`

`canonical/ARCHITECTURE_FREEZE_V1.json` and `canonical/BEHAVIORAL_HARDENING_LEDGER_20260903.md` remain historical provenance. Their old continuation ordering and old freeze state are superseded for **current selection authority only**; their underlying scoped experiments are not deleted or reinterpreted.

## Current hardening closures

Before promotion to main, the current branch passed:

- visible hash-pinned canonical graph optimizer execution authority;
- Geppetto deterministic/matching hardening, including permutation-invariant coincident-locus handling and removal of the incorrect float64 tie safety assumption;
- IRIS mode determinism, resource contract and support telemetry regressions;
- bounded Compiler skin and MWB2 mesh-skin repair accounting;
- Arachne V2 checkpoint/lineage contract;
- learned-artifact registry fail-closed authorization;
- typed pre-FIT textured family truth eligibility;
- blinded family selector requiring typed eligibility reports rather than forgeable booleans.

The pre-main wide gates then passed after one stale historical blob pin was rebound to the already-tested bounded `mwb2_skin.py` repair-accounting implementation.

## Exact main baseline already verified

The promoted code baseline `d4da94279e1f8d2bbc3882c428b975e2e19cfab4` passed on exact `main`:

- `current-mainline-self-hosted-ci` run `33863458125`: `SUCCESS`;
- `current-runtime-self-hosted-ci` run `33863458344`: `SUCCESS`.

The runtime gate includes directional binding/bake/projection regressions, synthetic proof/export wiring, native C++ build, ABI smoke, and exact package open/sample/render interlock. These remain **wiring/runtime evidence**, not a learned real-family product PASS.

## What is still not authorized

The architecture seal authorizes pre-FIT family **selection**, not training. At this reconciliation point:

- FIT8 cohort selected: `NO`;
- Family-1 exact observations bound: `NO`;
- IRIS real-family truth bound: `NO`;
- anonymous Geppetto real-family truth bound: `NO`;
- dense Arachne real-family truth bound: `NO`;
- zero-step learned real-family E2E harness: `NOT YET RUN`;
- Family-1 optimizer steps: `0`;
- generalization claim: `NO`;
- product-ready claim: `NO`.

## Next canonical staircase

1. run one final exact-main verification after these metadata reconciliation commits;
2. freeze a **new** exact first-fit base ref without moving the historical first-fit base;
3. enumerate the authoritative Master textured 1024 candidate pool using only frozen pre-FIT criteria;
4. evaluate exact typed eligibility and visual/data-quality audit before any fit metric exists;
5. blind-freeze eight families;
6. bind Family-1 exact observations and all three training-truth authorities;
7. run the real-family zero-step E2E harness;
8. only then permit Family-1 optimizer steps;
9. Family-1 PASS additionally requires visible rendered 8-direction/rest output and at least one compiled motion result;
10. only after that visual/scientific PASS may the remaining seven families be fit; unseen generalization comes later.

This reconciliation therefore closes the **authority ambiguity**, not the real-family learning experiment.
