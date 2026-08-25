# Representation Authority Execution Freeze — CI96

**Date:** 2026-08-24  
**Status:** `PRE_RESULT_EXECUTION_FROZEN__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Why CI96 exists

CI86 completed the frozen 256-asset representation-only source stage, then failed before cache/audit/study execution because the uploaded execution ZIP omitted `coords.py`, a local dependency imported by `geometry.py`.

No R0-R3 result was opened. The frozen panel, scoring, noise, truth, candidate universe and thresholds are unchanged.

CI96 is an apparatus/provenance correction only.

## Exact authority

- branch: `audit/iris-architecture-discipline-20260824`
- source head: `2f1b8ff199f0c147b9641ff2ec2cd22f56c67cb5`
- workflow: `IRIS V2 Preflight`
- run: `#96`
- run ID: `32677490168`
- conclusion: `SUCCESS`
- artifact ID: `9503132948`
- artifact: `iris-v2-r0-r3-execution-bundle-v3`
- artifact ZIP SHA-256: `488e053bbdcd84eb846b3cc856984f70c69f026220256c2872fa44eea35fb7ee`
- Drive mirror: `reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI96.zip`
- Drive file ID: `1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`
- launcher: `RealSaS_IRIS_V2_R0_R3_Representation_Only_CI96.ipynb`
- launcher SHA-256: `d0cb6260bcd021a1286d2ad16e18f9f184f3122626d8cea215b04fcedccaebcd`

## Bundle dependency closure

The final bundle contains exactly:

- `BUNDLE_INFO.json`
- `SHA256SUMS.txt`
- `build_representation_seed_v1.py`
- `stage_representation_authority_v1.py`
- `prepare_representation_cache_v1.py`
- `audit_representation_stage_cache_v1.py`
- `representation_authority_study_v1.py`
- `compact_representation_handoff_v1.py`
- `geometry.py`
- `coords.py`
- `run_representation_authority_v2.py`

Before upload, CI96:

1. checks exact source-head checkout;
2. compiles committed sources;
3. runs architecture/coordinate/matcher, learner-firewall, representation-data, evaluator and R0-R3 semantic preflights;
4. builds the bundle;
5. scans bundled imports against repository-local modules and fails on missing dependencies;
6. executes the representation-only synthetic stage/cache/audit smoke from inside the isolated bundle directory;
7. verifies internal `SHA256SUMS`;
8. removes test `__pycache__` artifacts;
9. asserts the exact final bundle file set;
10. only then uploads the artifact.

All steps passed in run #96.

## Scientific freeze — unchanged

- panel: 256 OPEN assets, ordered ID digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- SAME-locus tolerance: `0.003` canonical units
- R0: exact P
- R1: exact P+N
- R2 P sigma: `0, .0005, .001, .0025, .005, .01`
- R3: full 6x5 P/N noise grid, N angles `0,5,10,20,40` degrees
- P+N diagnostic coefficient: `0.05`, fixed
- deterministic Philox observation noise
- exact tangent-plane N perturbation
- physical set-containment top1/top4/top8
- reciprocal + 3-view cycle
- family median/p90/p95 tails
- geometry-only nearest-non-equivalent confusability diagnostic
- optimizer steps: `0`
- sealed splits opened: `false`
- training authorized: `false`

R4 remains undefined and may not be formulated after seeing R0-R3 results without a new pre-result formulation preregistration.

## CI86 stage salvage rule

The CI96 launcher may reuse the already-completed CI86 local stage only within the same live Colab runtime and only if it verifies:

- CI86 frozen seed SHA and panel digest;
- identical representation stage-builder SHA;
- all 256 stage markers;
- stage schema/profile and physical firewall;
- `images_staged=false`;
- exact staged file set and file SHA for every asset;
- no RGB/image files.

If any check fails, CI96 must stage fresh. No asset substitution is permitted.

## Next authority

Run the CI96 launcher. A successful run may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

No optimizer or learner training is authorized by this freeze.
