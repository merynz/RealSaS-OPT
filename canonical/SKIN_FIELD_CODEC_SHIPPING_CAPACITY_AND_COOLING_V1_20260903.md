# RealSaS — SkinFieldCodec shipping capacity and optimizer cooling V1 — 2026-09-03

**Status:** `SHIPPING_CODEC_CAPACITY_PASS__CONSTANT_LR_STABILITY_FALSIFIED__COSINE_GENERIC_PROTOCOL_PASS__ARACHNE_A1_NEXT`

## Question

The previous A0 behavioral panel used a deliberately small surrogate Codec (`32 hidden / 8 latent / 2 encoder / 2 decoder layers`). Its sharp witness failure could not establish product-level SkinFieldCodec incapacity because the shipping/default Codec is materially larger:

- hidden dimension: `192`
- latent dimension: `64`
- encoder layers: `3`
- decoder layers: `3`
- dropout: `0.0`
- config hash: `24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

The required causal question was therefore:

> Can the actual shipping Codec represent the preregistered synthetic skin fields and survive the real downstream Compiler + verified-LBS boundary?

## Frozen witness/acceptance authority

No witness, seed or threshold was changed.

Frozen panel:

- `chain_blend_3`, seed `20260921`
- `branch_blend_4`, seed `20260922`
- `sharp_fork_5`, seed `20260923`

Frozen A0 acceptance:

- row-L1 p95 `<= 0.05`
- deformation ratio `<= 0.05`
- simplex residual `<= 1e-6`
- no negative weights
- Compiler correction L1 `<= 1e-5`
- exact qualified row coverage
- `3` consecutive accepted evaluations
- evaluation every `32` optimizer steps
- maximum `1536` steps

The shipping test additionally evaluates raw decoded W and Compiler-qualified W separately. Compiler qualification is not allowed to hide a bad predictor.

## Shipping boundary

```text
teacher W
 -> shipping SkinFieldCodecV1 (192/64/3/3)
 -> raw decoded W
 -> SkinProposalIR
 -> Compiler.qualify_skin
 -> QualifiedSkinIR
 -> verified LBS
```

## Constant-LR shipping run

Dedicated workflow: `skin-field-codec-shipping-capacity`.

Observed on the current branch:

- `chain_blend_3`: sustained PASS
- `branch_blend_4`: sustained PASS
- `sharp_fork_5`: FAIL to sustain three consecutive accepted evaluations

The sharp witness nevertheless entered the acceptance region repeatedly. In the eastus replay its best raw/qualified row-L1 p95 was approximately `0.021160744`, while the final constant-LR p95 was approximately `0.0909396`.

This directly falsifies the claim that the frozen acceptance region is unreachable by the shipping representation.

Raw and qualified W were nearly identical throughout. Compiler total correction was only on the order of `1e-7` to `1e-6`, far below the `1e-5` repair ceiling. Therefore the result is not produced by Compiler rescue.

### Constant-LR verdict

`CONSTANT_LR_1E-3_AS_STABLE_SHIPPING_A0_PROTOCOL = FALSIFIED`

The failure mode is trajectory instability/oscillation after entering a valid region, not demonstrated representational impossibility.

## Controlled optimizer cooling A/B

A full-panel controlled A/B kept all of the following identical:

- shipping Codec bytes/config;
- witness definitions;
- seeds;
- loss function;
- AdamW optimizer family;
- weight decay;
- data/binding;
- acceptance thresholds;
- evaluation cadence;
- maximum steps.

The single manipulated variable was:

```text
A: constant LR = 1e-3
B: CosineAnnealingLR(T_max=1536, eta_min=0)
```

The cosine lane sustained PASS on all three witnesses.

### Cosine results

| witness | sustained PASS step | final raw row-L1 p95 | final qualified row-L1 p95 | final deformation ratio | Compiler correction L1 |
|---|---:|---:|---:|---:|---:|
| `chain_blend_3` | `544` | `0.01764197` | `0.01764196` | `~0.006313` | `~3.10e-7` |
| `branch_blend_4` | `672` | `0.02839057` | `0.02839056` | `~0.007950` | `~4.79e-7` |
| `sharp_fork_5` | `1056` | `0.02704993` | `0.02704993` | `~0.005085` | `~5.91e-7` |

The sharp lane also reached a best p95 of approximately `0.0192014` before the sustained acceptance sequence.

## Causal conclusions

### 1. Shipping Codec representation bottleneck

`FALSIFIED` for the preregistered generic synthetic panel.

The actual shipping representation can encode/decode all three frozen field classes below the frozen W and deformation ceilings.

This does **not** prove generalization to real families. It closes the narrower architectural-capacity question that motivated the cleanroom audit.

### 2. Hybrid Arachne/Codec split

Still `UNKNOWN` at this gate.

A0 proves that the Codec can represent the field when teacher W is available to its encoder. It does not yet prove that Arachne can infer the required per-joint latent from shipping `S + Qualified G` conditioning.

Therefore the next authoritative seam is:

```text
shipping S/G conditioning
 -> default ArachneCandidateV2
 -> frozen qualified shipping Codec decoder
 -> raw W
 -> SkinProposalIR
 -> Compiler.qualify_skin
 -> verified LBS
```

### 3. Compiler role

Compiler is behaving as intended: legality/simplex/lineage authority with negligible bounded correction. It is not responsible for the successful skin field.

## Training-protocol decision

The cosine schedule is admissible as a **generic shipping A0 optimizer protocol repair** because:

- it was selected by a controlled full-panel causal comparison;
- it changes no witness, seed, family-specific constant or architecture;
- it resolves the same optimizer instability across heterogeneous synthetic semantics;
- the constant-LR historical failure remains preserved as evidence.

This decision is training protocol, not architecture redesign.

## Next gate

`P0 SHIPPING ARACHNE -> FROZEN SHIPPING CODEC -> COMPILER -> VERIFIED LBS`

Use the same frozen witness panel and existing A1 product thresholds. Instantiate the default shipping Arachne config (`128 model dim / 2 surface encoder layers / 4 heads / 384 FF`) and the qualified shipping Codec (`192/64/3/3`). Do not substitute the historical tiny A1 model.

Architecture refreeze and formal family selection remain blocked until this seam and the remaining cleanroom obligations close.
