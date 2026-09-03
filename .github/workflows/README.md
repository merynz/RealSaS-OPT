# CI Workflow Index

`.github/workflows/` is an **enforcement surface**, not a second architecture document.

A workflow proves only the contract named by that workflow and the source paths/tests it executes. A green historical or experiment workflow does **not** by itself authorize product truth, fitting, shipping, or architectural supersession.

## Restoration gates

These are the restoration-specific gates introduced on `restoration/compiler-runtime-promotion-v1-20260903`:

- `native_runtime_source_gate.yml` — verifies the sealed nine-file native runtime subtree byte-for-byte, configures/builds it, then runs ABI smoke CTest.
- `proof_service_promotion_gate.yml` — compiles the current proof boundary and promoted diagnostic service, then runs causal diagnostic/source tests.

Their authority is bounded to the promoted mechanisms they test.

## Current product/source contract families

Workflow names such as the following enforce current typed source contracts or product-boundary invariants:

- `architecture_v4_contract.yml`
- `complete_e2e_source_contract.yml`
- `proof_engine_source_contract.yml`
- `export_runtime_source_contract.yml`
- `motion_source_contract.yml`
- `appearance_source_contract.yml`
- `arachne_shipping_boundary.yml`
- Geppetto / Arachne / IRIS source-contract gates
- MWB typed-seam / directional behavioral gates

Consult `CURRENT_STATE.md` and `canonical/` before interpreting any of these as a scientific authorization gate.

## Research / apparatus workflows

Names tied to a specific experiment, oracle substrate, corpus triage, one-family/heterogeneous probe, capacity test, or dated apparatus are research evidence. Examples include `r6_oracle_substrate_*`, `consumer_*`, `corpus_*`, `mwb*`, and similar narrow gates.

They may be extremely valuable evidence, but they never become canonical ownership merely because they pass.

## Workflow rules

1. New production CI must name the contract it enforces, not a person/family/witness.
2. Product workflows must use path filters narrow enough to explain what triggered them.
3. A workflow that protects historical or experimental apparatus must not be described as a product-completion gate.
4. Branch-specific hard-coding is allowed only for deliberately temporary migration/restoration gates and should be removed or generalized before merge to canonical main.
5. Duplicate workflows with overlapping claims should eventually be consolidated, but deletion happens only after their unique evidence/coverage is catalogued.
6. Green CI is necessary evidence, never sufficient scientific authority by itself.
