# Arachne A1 V5 Minimal K4 Direct Simplex — FIT1 Mainline Promotion

**Date:** 2026-09-11  
**Witness:** Mage FIT1  
**Promotion status:** `FIT1_FROZEN_PROMOTED`  
**Architecture:** `RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5`

## Decision

Promote the sealed minimal learned-skinning route:

`RiggingSurfaceIR + QualifiedSkeletonIR -> exact V4 observable-input backbone -> K4×512 Z -> direct row-simplex decoder -> SkinProposalIR -> Compiler skin qualification`

The A0 continuous-field decoder is **not** a runtime dependency of this promoted route.

This is the minimal architecture authorized by the matched 16K causal diagnostic. Both direct readout arms passed; because `Z_DIRECT_SIMPLEX` passed with fourteen consecutive full FIT1 checks while preserving the K4 latent interface, the smaller seam change is preferred over bypassing K4.

## Frozen identities

- V4 backbone checkpoint SHA-256: `95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763`
- V5 source SHA-256: `a66adaebe92e9181873888ef90941ad87e4b3d835d21647e65176df4441a0f9e`
- sealed Z direct decoder SHA-256: `078d5155f798d7b926c19463a31bf01c19973be30787f5fbff7916be87894832`
- promoted decoder delta SHA-256: `13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137`
- V5 closure report SHA-256: `6ba5d63a5e9d5cab0ebc6f374bc31e86ffb2a3ed325b89dc9880b5b858ce83f4`
- V5 closure seal SHA-256: `11fad459db94bc5604585fb63f22a018f538c8d30084d92c0c4e83747c619048`
- composite manifest SHA-256: `a6fec97b84739452e4b0126c4107a54f23a585d2f14c00548e0501d12b4c9a4f`
- decoder parameters: `325,313`
- total V5 parameters: `138,378,466`

## Closure result

- GSA row-L1 p95: `0.04237784981177733` — PASS against frozen `<=0.05`
- deformation-error ratio: `0.019856400787830353` — PASS
- articulated deformation ratio: `0.002780771814286709` — PASS
- dominant accuracy: `0.9957173447537473`
- Compiler total correction L1: `2.9468642839168442e-05`
- holdout p95 diagnostic: `0.15087631421532924`
- A0 continuous-field runtime model loaded: `false`
- optimizer/backward/update in closure: `false / false / false`

Categorical closure verdict:

`PASS__MINIMAL_K4_Z_DIRECT_SIMPLEX_V5_FIT1_CLOSURE`

## Causal basis

The preceding full-horizon matched diagnostic produced:

- Z direct best p95 `0.04237794729012932`, stable streak `14`
- H direct best p95 `0.039081553225318116`, stable streak `15`
- verdict `BOTH_DIRECT_SIMPLEX_PASS__PREFER_MINIMAL_Z_DECODER_SEAM`

Because H and Z crossed the gate at nearly the same training horizon, no material H→Z information bottleneck is supported. The common causal change was removal of the old continuous-field decoding interface in favor of direct per-row joint competition.

## Mainline source home

- `models/arachne/v4/` — exact frozen V4 backbone source dependency
- `models/arachne/v5/` — promoted minimal direct-simplex source and witness

Historical A0/V4 diagnostics and failed experiments remain preserved as scientific lineage. Promotion does not delete or rewrite their evidence.

## Claim boundary

Supported:

> On the controlled real Mage FIT1 witness, the current generic RealSaS rigging-core route reaches qualified learned skin weights under the frozen `0.05` skinning gate using product-available surface+skeleton inputs and no A0 continuous-field runtime model.

Not supported:

- unseen-character or unseen-family generalization;
- FIT8/LOFO closure;
- independent end-to-end `PRODUCT_PASS`;
- animation/appearance acceptance.

The next scientific gate is `V5_FAMILY_DISJOINT_UNSEEN_GENERALIZATION_GATE`, after the FIT1 demo/evidence lock.
