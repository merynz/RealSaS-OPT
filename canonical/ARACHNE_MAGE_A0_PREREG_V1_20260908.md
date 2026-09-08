# RealSaS — Arachne Mage A0 Codec Ceiling Prereg V1 — 2026-09-08

**Status:** `PREREGISTERED_BEFORE_A0_OUTPUT__A0_OPTIMIZER_AUTHORIZED_AFTER_RUNNER_PREFLIGHT__A1_BLOCKED`

## Scientific question

Can the already-proven shipping/default `SkinFieldCodecV1` represent the real Mage skin field on the exact current product boundary:

`IRIS/GSA RiggingSurfaceIR S (950) + Geppetto/Compiler QualifiedSkeletonIR G (22)`

without allowing uncertain teacher projection rows or Compiler repair to manufacture the PASS?

A0 is a one-family representation/capacity ceiling. It is not Arachne prediction and not generalization.

## Frozen input authority

- exact shipping surface lineage: `67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb`
- exact QualifiedSkeletonIR.v2 lineage: `738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306`
- exact S/G/W binding: `cb41eb7055b8e2646628daecdd0e31dfc079d163d5f5adaaa1a92f1ca1dfb994`
- P0 W content: `c15db7b78d272ac22998071e1fb1cec4c65d7366133ef16fb72824f222c852d9`
- exact conditioning V1 hash: `7b82d57b1967858c6a66de871ecd570a9bdae408b30ff84c3fcbd8e908551047`
- exact conditioning V2 hash: `89973cd1f0ac989d10cdfeafdef89545668fda4998363df3e872961646879939`
- local exact-conditioning cache SHA-256: `12484afc23d5c03cbad8020266ed5b39c3201d979e78f96380e748902152be6e`

Target rows and columns are rebound by exact `surface_id` / `canonical_joint_id`; incidental array order is never authority.

## Frozen confidence policy

P0 confidence inventory:

- HIGH: `920`
- MEDIUM: `7`
- MEDIUM_LOW: `4`
- LOW: `19`

Define the authoritative teacher-supervision mask as all rows except LOW: `931 / 950` rows.

The 19 LOW rows are excluded from **both**:

1. the teacher-weight encoder aggregation used to create A0 joint latents; and
2. teacher reconstruction/deformation loss and teacher-error PASS metrics.

Implementation rule:

- call `encode_teacher_weights(..., surface_mask=teacher_supervision_mask, ...)`;
- call `decode_from_latents(..., surface_mask=full_950_surface_mask, ...)`;
- compute reconstruction and teacher-motion error only with `teacher_supervision_mask`;
- submit the decoded full 950-row field to Compiler and full-field safety checks.

Thus uncertain teacher rows cannot steer the latent or the loss, while the frozen decoder is still exercised on all admitted surface rows.

LOW-row teacher L1/deformation error is telemetry only and can never create or veto A0 scientific PASS. LOW-row confidence remains preserved for later visual/motion investigation.

## Frozen codec

Use the existing shipping/default model bytes and config only:

- architecture: `RealSaS.SkinFieldCodec.ContinuousJointField.v1`
- config hash: `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`
- surface feature dim: `20`
- joint feature dim: `8`
- hidden dim: `192`
- latent dim: `64`
- encoder layers: `3`
- decoder layers: `3`
- dropout: `0`

No SkinTokens-style sigmoid redesign, RigAnything direct pairwise arm, new decoder, extra hidden mesh input, or Geppetto hidden-state reuse is authorized in A0.

## Frozen optimizer discipline

Inherited from the causally closed shipping-codec cooling protocol:

- seed: `20260908`
- AdamW `lr=1e-3`, `weight_decay=1e-4`
- `CosineAnnealingLR(T_max=1536, eta_min=0)`
- max optimizer steps: `1536`
- check at step 1 and every `32` steps
- checkpoint every `128` steps and at terminal PASS/failure
- `3` consecutive FULL PASS checks required
- early stop at the first check completing the 3-check streak
- no post-PASS continuation before decoder freeze

Exact resume checkpoint must include model, optimizer, scheduler, step, trace/streak and Python/NumPy/Torch/CUDA RNG states. Resume may not reset the terminal streak.

## Frozen A0 motion-sensitivity probe

A0 retains the already-used generic codec identity-sensitive 4-pose transform family rather than inventing a Mage-specific semantic animation before the representation ceiling is known.

For lexicographically ordered qualified joint index `ji`, with `u=(ji+1)/J`:

- pose 0: identity;
- pose 1: translation `x=0.10*u`, `y=0.035*(-1 if ji odd else +1)`;
- pose 2: translation `y=0.085*u`, `z=0.030*(ji-(J-1)/2)`;
- pose 3: translation `x=-0.055*(ji-(J-1)/2)`, `z=0.070*u`.

This is a codec sensitivity probe, not final runtime motion proof. Hierarchical limb/compound/visual probes remain mandatory at A1/FIT1 closure.

## FULL PASS criteria

All must hold at the same check.

### Authoritative 931-row teacher band

Raw decoded field:

- row-L1 p95 `<= 0.05`;
- teacher-motion deformation error ratio `<= 0.05`.

Compiler-qualified field:

- row-L1 p95 `<= 0.05`;
- teacher-motion deformation error ratio `<= 0.05`.

### Full 950-row product safety

- decoded weights finite;
- negative weight count `= 0`;
- raw simplex max absolute residual `<= 1e-6`;
- qualified simplex max absolute residual `<= 1e-6`;
- qualified row count `= 950`;
- illegal/missing S/G references `= 0`;
- Compiler total correction L1 `<= 1e-5`;
- Compiler maximum row correction L1 `<= 1e-5`;
- no top-k sparsification in A0 qualification (`max_influences=None`).

Compiler is legality/lineage/simplex authority only and may not rescue semantic field error.

### LOW-row telemetry

Report separately:

- raw and qualified row-L1 mean/p95 against provisional P0 LOW W;
- deformation-error telemetry;
- dominant-joint distribution;
- local GSA-neighbor weight discontinuity telemetry;
- finite/simplex status.

No LOW-row teacher metric contributes to PASS because the teacher mapping is explicitly uncertain there.

## A0 result semantics

`PASS` means the frozen shipping codec can encode/decode the authoritative real-Mage skin field on 931 teacher-authoritative rows while producing a legal full 950-row field at the shipping S/G boundary for 3 consecutive checks.

It does not mean Arachne can infer the latent from observations and does not establish held-out generalization.

At PASS, freeze the **earliest sustained-PASS decoder state** and its hashes. A1 must use that exact decoder; decoder retraining during A1 is forbidden.

## Change control

After optimizer step 1, do not change target W, confidence mask, S/G lineage, conditioning hashes, model config, optimizer family/schedule, thresholds, probe transforms, Compiler repair budget or terminal-stability rule to obtain PASS.

A failure may motivate only a separately preregistered causal diagnostic/arm.

## Next gate

1. implement/run source + exact-cache preflight;
2. execute A0 under this prereg;
3. if and only if A0 PASS: freeze decoder and preregister `ARACHNE_MAGE_A1_FIT1`.

`A1_OPTIMIZER_AUTHORIZED = FALSE` until A0 PASS.
