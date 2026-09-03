# RealSaS — Deterministic Preset Motion / Deformation Behavioral Preregistration — 2026-09-03

**Status:** `FROZEN_BEFORE_BASELINE_RESULT`

## Seam

`Qualified mechanical S/G/W -> build_deterministic_preset_motion -> puppet-local joint track -> verified LBS motion probe`

This is a deterministic producer/consumer capacity gate. It does not claim animation style quality, learned motion, or generalization.

## Why this gate exists

A non-constant motion key track is not sufficient product evidence if the selected canonical joint has no qualified skin influence and therefore cannot move the puppet under the base LBS path.

The authority for this gate is downstream behavior, not joint naming or exact selected joint identity.

## Frozen witnesses

Three generic two-/three-joint mechanical witnesses are fixed before baseline execution:

1. `weighted_child`: a non-root joint carries material qualified skin mass;
2. `root_only_with_decoy_child`: a legal non-root joint exists but carries exactly zero skin mass, while the root carries the surface;
3. `split_children_with_decoy`: multiple legal non-roots exist with heterogeneous qualified skin support including a zero-mass decoy.

All witnesses use anonymous canonical IDs, simple planar surface points, qualified simplex skin rows and no authored semantic joint names.

## Frozen probe

For the single preset joint track:

- use the quarter-duration key, where the generated preset has its positive rotation amplitude;
- convert the key's puppet-local in-plane rotation into a 4x4 LBS transform about the selected canonical joint rest pivot;
- keep all untracked joints identity;
- evaluate the admitted surface points with the qualified W matrix using `apply_verified_lbs_v1`;
- compare against rest positions.

## Frozen PASS criteria

For every witness:

1. motion state is deterministic under exact replay;
2. exactly one generated preset joint track is sufficient for this source gate;
3. track transform space is `PUPPET_LOCAL_2D_2P5D`;
4. authored joint-name, quaternion and rigid-Vec3 authority remain absent;
5. the selected joint has positive total qualified skin mass;
6. quarter-key verified-LBS max displacement is materially nonzero (`> 1e-10` object units);
7. increasing only `amplitude_deg` from `4` to `12` strictly increases RMS surface displacement;
8. changing amplitude changes motion-state hash;
9. no witness-specific joint ID or topology branch is allowed in the producer.

The gate does **not** require a particular joint identity. Any deterministic joint choice is acceptable if it satisfies the behavioral criteria.

## Failure policy

Witnesses, amplitudes, probe construction and thresholds are frozen after this preregistration. A baseline failure may authorize only a generic motion-producer repair. Proof-engine/runtime causal sensitivity remains a later seam and is not granted PASS by this producer gate.

## Refreeze policy

Architecture refreeze remains blocked until this gate passes and is promoted into the architecture-freeze executable prerequisites.
