# Experiments

`experiments/` contains research, training, diagnostics, causal probes, compatibility apparatus and scientific evidence. **Nothing in this directory owns shipping product truth.**

## How to read this directory

Experiment folders are intentionally allowed to retain historical names and dates because provenance matters. Do not infer authority from recency or naming.

Common families currently present include:

- IRIS perception/reprojection/control experiments;
- Geppetto/Arachne capacity, oracle and teacher-projection apparatus;
- MWB identity/directional seam experiments;
- consumer/export/runtime interlock probes;
- corpus/FIT observation and family-selection apparatus;
- geometry, appearance, motion and deformation behavioral probes;
- exact/synthetic/one-family E2E harnesses.

The active scientific authorization state is defined by `CURRENT_STATE.md` on canonical main and by `RESTORATION_STATE.md` while the restoration branch is open.

## New experiment contract

New experiment directories should use one obvious purpose and include a local `README.md` containing:

1. **Question** — the falsifiable question being tested.
2. **Inputs** — exact source/product/corpus authority and hashes where material.
3. **Forbidden information** — truth/firewall constraints.
4. **Procedure** — executable entry point and deterministic configuration.
5. **Outputs** — reports, metrics and artifacts produced.
6. **Pass/fail rule** — preregistered threshold or invariant when applicable.
7. **Authority** — explicitly state what a PASS does and does not authorize.
8. **Status** — planned, active, closed, superseded, archival, or quarantined.

Preferred new naming shape:

`experiments/<domain>_<gate_or_question>_<YYYYMMDD>/`

Dates are provenance, not supersession.

## Promotion rule

When an experiment proves a mechanism that belongs in production:

`experiment evidence -> canonical decision -> production implementation -> production regression gate`

Do **not** import an experiment package into shipping code as a shortcut. Production code moves into its semantic home (`compiler/realsas_compiler_core/`, `compiler/realsas_compiler_services/`, or `runtime/`) and the experiment remains as provenance/evidence.

## Cleanup policy

Existing experiment folders are preserved until their unique evidence is catalogued. Cleanup means classification first, deletion last. A folder may later be marked `SUPERSEDED`, `ARCHIVAL`, or `QUARANTINED`, but evidence should never silently disappear.
