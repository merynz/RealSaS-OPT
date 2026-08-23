# IRIS Controlled V1 — Historical Only / Do Not Run

**Effective:** 2026-08-24  
**Authority:** `CURRENT_STATE.md` on `audit/iris-architecture-discipline-20260824`

This directory is preserved for provenance, diagnostic comparison and reproducibility of the completed Controlled V1 / M256 lineage.

It is **not an active execution authority** on the audit branch.

Do not run:

- `launch_iris_controlled_v1.py`;
- old `representation_ceiling_v1.py` as the current representation gate;
- old training/evaluation launchers;
- M256/mini launchers as a substitute for V2.

The active candidate implementation is:

`experiments/iris_single_pose_v2/`

The next authorized executable is optimizer-zero only:

`experiments/iris_single_pose_v2/run_representation_authority_v1.py`

No optimizer run is authorized until `CURRENT_STATE.md` is explicitly advanced after the frozen R0-R3 Representation Authority result and its canonical interpretation.

Nothing in this marker invalidates historical V1/M256 measurements; it only prevents those executables from being mistaken for the current scientific contract.
