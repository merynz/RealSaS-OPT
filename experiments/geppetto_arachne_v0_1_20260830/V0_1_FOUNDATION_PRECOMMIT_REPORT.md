# Geppetto/Arachne V0.1 — Foundation Slice Precommit Report

**Date:** 2026-08-30  
**Base main:** `1e10b3f5fd140feb237f15ee427ef9d713d87a6b`  
**Branch:** `geppetto-arachne-v0-1-20260830`  
**Status:** `FOUNDATION_AND_G0_1_SCAFFOLD_PREFLIGHT_PASS__FREEZE_CANDIDATE_NOT_SEALED`

## Scope completed

- `SkeletonTeacherProjectionV1`
- 27D typed surface/interior consumer token contract
- optional/versioned geometry uncertainty validity
- deterministic interior sampling
- five-state point→control path evidence
- production-safe point/control geometry with no teacher-tail leakage
- Geppetto G0.1 geometry-aware KNN set encoder scaffold
- autoregressive anonymous-control decoder scaffold
- position uncertainty / existence / root evidence heads
- full directed parent-evidence head
- current `SkeletonProposalIR` proposal adapter boundary

No Arachne neural decoder, training run, architecture seal, or product qualification is claimed by this patch.

## Executed validation

Foundation suite:

```text
PASS test_interior_sampler_never_admits_outside_or_surface
PASS test_path_state_detects_certain_outside_crossing
PASS test_point_control_geometry_root_does_not_invent_axis
PASS test_surface_token_validity_and_optional_uncertainty
PASS test_teacher_projection_preserves_multiple_roots_and_randomized_bfs_is_legal
PASS test_teacher_projection_rejects_cycle
PASS test_teacher_projection_skips_helper_and_preserves_skin_column
FOUNDATION_V0_1_PASS 7/7
```

Geppetto scaffold suite:

```text
PASS test_current_compiler_proposal_adapter_boundary
PASS test_forward_contract_and_diagonal_parent_mask
PASS test_token_permutation_equivariance_at_decoder_output
GEPPETTO_G0_1_SCAFFOLD_PASS 3/3
```

Combined slice smoke:

```text
PASS foundation
PASS geppetto
GEPPETTO_ARACHNE_V0_1_SLICE_PASS
```

All Python sources/tests were also passed through `python -m py_compile` before commit.

## Exact-byte SHA-256

```text
contracts_v0_1.py       30cfe9567b4fee8181fde436fee675d0feca7ad92133f70229286c59c095eb2d
teacher_projection_v1.py e4e4b16cd9a21110835100a83465fb20ca20cc39ed703f23738540b4a1bc0889
substrate_adapter_v1.py  54ae842a5fd08549216485d22b9d4898ff9e0b5f0e65bb1964984b4de8647f14
geppetto_g0_1.py         9d8a686fbc4877acfca57e13c63edff001c098506e6b73dadf04b1dc8ec76762
test_repo_slice_v0_1.py  704e4fb1b12945fcca897b3aae6d08402858626246c192e981f7cf785596d8c9
```

## Git blob equality proof

Local `git hash-object` values were compared to the GitHub branch blob/content SHAs after write:

```text
contracts_v0_1.py        9a25832eb669f055d4abdcb0f69cd78a730e89b1
teacher_projection_v1.py 3f82dd5559ab3620ed9d47005316642037d9d856
substrate_adapter_v1.py  b51747110b0957f2e482959eff06728bb02b0253
geppetto_g0_1.py         f7e9423c48b5b8a8940d912815c65b75e2c1f546
test_repo_slice_v0_1.py  79613d4d2345fcae037c617c428892dd81e36693
```

Result: **5/5 exact byte equality** between tested local executable files and committed branch files.

## Authority / safety boundary

- source `bone_tails` stay teacher-only audit/skin-column provenance;
- source IDs do not become proposal/canonical IDs;
- G0.1 emits anonymous proposals and directed parent evidence only;
- Compiler remains root/tree/canonical-ID authority;
- current IRIS risk head is not assumed to exist;
- Arachne must consume `QualifiedSkeletonIR`, not raw G0 topology;
- V0.1 remains a freeze-candidate, not a sealed architecture.

## Next gate

Before training G0.1, run the TRAIN-authority skeleton projection audit for:
helper-bone chains, multi-root frequency, zero-length deform bones, skin mass on non-deform columns, deform-control count distribution, and the `bone_head` control-location convention. This gate determines the training data contract; it must not be inferred from later model outcomes.
