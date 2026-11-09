# RealSaS Truth Foundry V20 — Actual Normalized NPZ Numeric Court

## Scope

190 distinct, SHA256-locked, physically materialized `normalized.npz` derivative files from the V17 strict Master726 allowlist. Normalized derivatives are **not creator-original files**; original parity, rights clearance, asset/character semantics, front/back raster view, motion/attachment correctness, training and TESSA animation-ready admissions are **all zero**.

## Results

- 190 / 726 = 26.171% of the eligible historical Master derivative transport queue; 190 / 16,397 = 1.159% of the whole candidate registry.
- All 190 NPZ CRC, array shape/index/finite, parent hierarchy and historical `source_structure.json` V/F/J count-parity courts passed. 0 quarantined.
- 2,240,149 actual NPZ vertices; 3,930,804 actual NPZ triangular faces; 127,641,236 actual dense skin weight entries.
- 44 files have zero-sum skin vertices (16,127 total), 25 have nonmanifold edges (324 total), 161 open boundary edges (434,757 total), 22 near-zero area triangles (2,504 total). These are **quality flags**, not automatic training fails or self-intersection findings.
- 1,346 previously unknown facet cells newly observed; 346 weaker historical-structure facet cells upgraded; 4,327,116 state+value cell pairs untouched. Existing V19 preserved.
- Independent source-bytes, SHA256, NPZ CRC and full 16,397 × 264 matrix readback: PASS.

## Evidence levels

`OBSERVED_REHASHED_NORMALIZED_NPZ_GEOMETRY_SKIN_ONLY` denotes NPZ derivative arrays tested *now*. It does not assert original topology, correct labels, anatomical semantic ownership, appearance parity, rigging quality, editability, rights, or model readiness.

## Provenance

Each receipt records source_key, strict recovery receipt digest, Drive NPZ file id, NPZ size, raw current NPZ SHA256, exact counts and conservative quality metrics. V20 delta separately records previous state and previous value SHA256 whenever a cell was upgraded. No original source binaries are put in GitHub or bundled in this report ZIP. The retrieved NPZ bytes are kept only in the private working cache.

## Reproduce

In a trusted environment with requisite private inputs under `REALSAS_TRUTH_ROOT` (default `/mnt/data`):

`python -m tools.corpus.truth_foundry.evidence_v20.build_v20`

`python -m tools.corpus.truth_foundry.evidence_v20.verify_v20`

The source inventory and V19 baseline SHA locks are fixed; missing/changed inputs fail closed. Use only eligible Master726 sources, exclude 135 protected holdouts, 644 unverified paths and 22 rejected recovery exceptions.