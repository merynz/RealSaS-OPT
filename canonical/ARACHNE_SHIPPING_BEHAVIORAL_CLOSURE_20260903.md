# RealSaS — Arachne Shipping Behavioral Closure — 2026-09-03

**Status:** `PASS_SHIPPING_A0_A1_CLOSED__CROSS_REGION_REPLAY_PASS__NO_REAL_FAMILY_CLAIM`

## 1. Scope

This closure covers the shipping semantic boundary:

`RiggingSurfaceIR S + QualifiedSkeletonIR G`
`-> default ArachneCandidateV2`
`-> per-joint latent evidence`
`-> frozen shipping SkinFieldCodecV1 decoder`
`-> raw dense W`
`-> SkinProposalIR`
`-> Compiler.qualify_skin`
`-> QualifiedSkinIR W`
`-> verified LBS deformation behavior`.

It closes generic synthetic behavioral capacity and handoff integrity. It is not a real-family generalization claim and does not authorize formal Family-1 selection.

## 2. Historical evidence corrections preserved

Earlier A0 behavioral evidence used a tiny surrogate Codec (`~32 hidden / 8 latent / 2+2 layers`) and an evaluator that failed to rebind teacher rows onto canonical lexicographic `surface_ids` for `N>=10`.

Those historical failures remain real executions, but they are not shipping-product capacity authority. They remain classified as:

- tiny-model failure: `NON_AUTHORITATIVE_FOR_SHIPPING_CAPACITY`;
- branch/sharp V1 row-binding conclusions: `INVALID / CONTAMINATED_BY_ROW_BINDING_BUG`.

No historical run is deleted or rewritten as a PASS.

## 3. Independent objective bug repaired

The historical per-class active-influence CE weighting made exact teacher W non-stationary. The repair changed emphasis to a row-scalar form so the exact teacher simplex is stationary again.

Regression evidence after repair: exact-truth CE logit gradient max approximately `4.43e-17`.

This bug was causal and generic, but it did not by itself explain the historical sharp failure.

## 4. Shipping SkinFieldCodec A0 closure

Shipping/default Codec configuration:

- architecture: `RealSaS.SkinFieldCodec.v1`;
- config hash: `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`;
- hidden dim: `192`;
- latent dim: `64`;
- encoder layers: `3`;
- decoder layers: `3`.

The same preregistered heterogeneous witnesses, seeds and acceptance thresholds were retained.

Boundary:

`teacher W -> shipping Codec -> raw W -> SkinProposalIR -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`.

Constant AdamW LR `1e-3` repeatedly entered the valid sharp region but did not sustain three consecutive PASS checks. This falsified hard representational impossibility while demonstrating that constant LR was not a stable shipping A0 protocol.

Controlled full-panel A/B changed only optimizer schedule:

- constant `1e-3`;
- `CosineAnnealingLR(T_max=1536, eta_min=0)`.

Stable cosine A0 result:

| witness | pass step | final raw/qualified row-L1 p95 | final deformation ratio |
| --- | ---: | ---: | ---: |
| `chain_blend_3` | 544 | ~0.01764 | ~0.00631 |
| `branch_blend_4` | 672 | ~0.02839 | ~0.00795 |
| `sharp_fork_5` | 1056 | ~0.02705 | ~0.00508 |

Compiler total correction remained on the order of `1e-7` to `1e-6`; raw and qualified W were effectively identical for acceptance.

A0 verdicts:

- `SHIPPING_CODEC_REPRESENTATION_BOTTLENECK = FALSIFIED` on the preregistered generic panel;
- `CONSTANT_LR_1E-3_AS_STABLE_SHIPPING_A0_PROTOCOL = FALSIFIED`;
- `GENERIC_COSINE_A0_PROTOCOL = PASS`;
- `COMPILER_RESCUE_EXPLAINS_PASS = FALSIFIED`.

## 5. Shipping Arachne A1 closure

Default shipping Arachne configuration:

- architecture: `RealSaS.ArachneCandidate.SegmentAwareJointField.v2`;
- config hash: `ee24afce200619c06753e39a617528be0fd84695e6358db24d828693ebcb72d1`;
- model dim: `128`;
- surface encoder layers: `2`;
- attention heads: `4`;
- feedforward dim: `384`.

For every witness the shipping Codec was first independently A0-qualified with the stable cosine protocol, then frozen. Only Arachne was optimized in A1.

Authoritative path:

`S + Qualified G`
`-> ArachneCandidateV2`
`-> frozen shipping Codec decoder`
`-> raw W`
`-> model.propose() / SkinProposalIR`
`-> Compiler.qualify_skin`
`-> QualifiedSkinIR`
`-> verified LBS`.

Results:

| witness | A1 pass step | final raw/qualified row-L1 p95 | deformation ratio | latent abs p95 |
| --- | ---: | ---: | ---: | ---: |
| `chain_blend_3` | 128 | ~0.01749 | ~0.00626 | ~0.02781 |
| `branch_blend_4` | 512 | ~0.09542 | ~0.02462 | ~0.16002 |
| `sharp_fork_5` | 1760 | ~0.08293 | ~0.01613 | ~0.16369 |

All three sustained the preregistered A1 requirement for three consecutive checks. Raw and Compiler-qualified W remained effectively identical; Compiler correction stayed far below `1e-5`.

The branch/sharp latent may differ materially from the teacher encoder latent while the final W and deformation behavior remain valid. Therefore teacher latent identity is diagnostic rather than product authority.

## 6. Exact cross-region replay

Workflow: `arachne-shipping-boundary`

Run: `33770002712`

First execution:

- job: `100697514568`;
- region: `westcentralus`;
- result: `3 passed`;
- witness pass steps: `128 / 512 / 1760`.

Exact job rerun with unchanged source/config/protocol:

- job: `100717194442`;
- region: `eastus`;
- result: `3 passed`;
- same witness pass steps and final metrics.

Verdict:

`CROSS_REGION_DETERMINISTIC_REPLAY = PASS`.

## 7. Scientific closure verdict

`SHIPPING_ARACHNE_TO_FROZEN_SHIPPING_CODEC_TO_COMPILER_TO_LBS = PASS / CLOSED`

`HYBRID_ARACHNE_LATENT_CODEC_BOUNDARY_AS_NECESSARY_INFORMATION_BOTTLENECK = FALSIFIED` on this preregistered generic panel.

This does **not** prove:

- real-family fit;
- cross-family generalization;
- observation-substrate information sufficiency;
- predicted IRIS sufficiency.

The next P0 question is upstream information sufficiency:

`U0_REFERENCE_FULL_SURFACE` vs `U1_OBSERVATION_ORACLE_SUBSTRATE`.

No architecture refreeze or formal family selection is authorized until that oracle-substrate gate is resolved.

## 8. CI authority after closure

Active Arachne/Codec CI authority is now:

1. source/generic invariants;
2. exact-truth Codec objective stationarity;
3. stable shipping A0 cosine gate;
4. shipping A1 -> frozen shipping Codec -> Compiler -> verified LBS gate.

Historical/falsified diagnostics remain in the repository for provenance but are not active product PASS authority.
