# RealSaS — IRIS privileged-input / truth-leak scope

**Date:** 2026-09-03  
**Status:** `SCOPE_ONLY__REPAIR_NOT_OPENED__LEARNED_IRIS_RESULTS_QUARANTINED`

This note scopes the currently identified IRIS leakage/privileged-input paths without opening a repair stream. It does not alter the active Geppetto behavioral-hardening sequence.

## Epistemic correction

The current IRIS V2 source and its 2026-08-31 preregistration explicitly admit native RGBA. Therefore the alpha path described below is **not an accidental implementation drift from that seal**. Under the stricter scientific firewall now adopted — learned perception must not receive deterministic raster foreground/segmentation authority as an input shortcut — the leak is in the scientific input contract itself and is faithfully implemented by source.

## Confirmed reachable privileged-input paths

### A. Alpha enters a learned native image backbone — CONFIRMED

- `ObservationContractV2` requires `8 x 1024 x 1024 x 4` RGBA and `load_observation_manifest_v2()` returns those four channels.
- `NativeResolutionPyramidV2(in_channels=4)` is a learned convolutional pyramid.
- `NativeResolutionPyramidV2.forward()` and `QDescriptorSamplerV2.sample_native_streamed()` require four-channel images and send them directly through the learned convolution blocks.
- Therefore renderer alpha is directly available to trainable IRIS parameters.

This is the highest-priority confirmed leakage path under the new firewall.

### B. Foreground masks constrain the learned hypothesis domain — CONFIRMED REACHABILITY

- `build_q_domain_v2()` accepts `foreground_masks` and uses them to set `foreground_support` and `candidate_valid`.
- `build_canonical_ray_lattice_v2()` also uses the foreground mask to decide which anchor rays exist at all.
- The Gate-0 real-corpus implementation constructs the image foreground mask from alpha (`alpha >= 128`) when alpha is non-opaque, with an exact-green fallback otherwise.
- Gate-0 additionally checks this image support against `raster_authority.npz` support.

This path does not inject a teacher 3D coordinate into the network, but it gives the learner/search domain deterministic segmentation/silhouette authority. Under the stricter firewall it is privileged observation geometry and must be separately adjudicated/repaired before learned IRIS evidence is reopened.

### C. Absolute view identity is injected into a learned evidence encoder — CONFIRMED

`QEvidenceEncoderV2` constructs a learned input scalar

```python
vi = torch.linspace(-1.0, 1.0, 8)
```

and concatenates it with sampled descriptors, projected image coordinates and validity before trainable layers.

Exact camera geometry itself is legitimate analytic Mode-G authority. However, the stricter firewall distinguishes analytic camera use from feeding absolute view/orbit identity into a learned appearance/evidence module. This path is therefore a confirmed metadata shortcut requiring explicit policy before IRIS results are unquarantined.

### D. Foundation backbone alpha leak — NOT PRESENT IN INSPECTED V2 SOURCE

`FrozenFoundationAdapterV2.forward()` requires `[B,8,3,H,W]` RGB. The frozen foundation path does not directly consume alpha in the inspected source.

The confirmed alpha leak is therefore the **learned native-resolution path**, not the frozen foundation adapter.

## Teacher-truth reachability checked so far

### Primary geometry / barycentrics / raster authority into learned forward — NOT CONFIRMED

The real-corpus Gate-0 code reads:

- `primary_geometry.npz` vertices/faces;
- `raster_authority.npz` pixel ids, triangle ids and barycentric coordinates;
- reconstructed truth rows.

But Gate-0 explicitly records `scientific_optimizer_steps = 0`, and these quantities are used there for containment/resolvability measurement rather than a learned forward pass.

In the inspected `train_v2.py`, `teacher_depth` and `teacher_support` are consumed by the loss, while the model forward receives `images`, frozen foundation features and the Q-domain. No direct `P`, `N`, triangle id or barycentric tensor has yet been found entering `IrisReprojectionV2.forward()`.

**Current status:** direct source-mesh/P/N/barycentric learner leakage is `NOT_PROVEN`; alpha/mask/view-metadata privilege is `CONFIRMED`.

## Artifact impact / quarantine boundary

### Quarantine

Until the privileged-input contract is repaired and refrozen, the following may not be used as clean evidence for an RGB-only / observation-only IRIS learner claim:

1. any learned IRIS Reprojection V2 checkpoint or metric produced with the current four-channel native path;
2. any learned evaluation downstream of such a checkpoint;
3. historical IRIS/G1 learner results whose declared input includes deterministic raster foreground alpha, if they are being cited under the stricter no-alpha scientific claim.

### Not automatically contaminated by this leak

1. IRIS V2 Gate-0 synthetic/real deterministic preflights with zero optimizer steps are not learned-checkpoint evidence. Their hull/foreground measurements remain what they are, but must not be reinterpreted as evidence for an RGB-only learner.
2. The current generic Geppetto behavioral witnesses use synthetic `RiggingSurfaceIR` and do not consume an IRIS learned checkpoint; their behavioral result is not invalidated by this IRIS input leak.
3. Frozen foundation feature caches generated from the inspected RGB-only foundation adapter are not alpha-bearing by that adapter path. Their provenance still requires normal hash/observation checks.

## Still unknown / inventory required before closure

- Exact inventory of Drive/notebook/checkpoint artifacts produced outside the repository with V2/G1 learned IRIS source.
- Whether any downstream real-family artifact currently cited as product evidence embeds a learned IRIS checkpoint from a privileged-input run.
- Whether foreground-mask restriction is to be removed entirely from learned inference or retained only as a non-learned compiler/search-domain authority under an explicitly revised contract.
- Whether absolute `vi` view identity is removed from the learned evidence encoder or replaced by an allowed analytic-equivariant camera relation representation.

## Current action boundary

`SCOPE_COMPLETE_ENOUGH_TO_QUARANTINE__REPAIR_DEFERRED`

Do not open a new IRIS fit or use existing affected learned results as scientific support while Geppetto hardening is active. Repair/refreeze is a later explicit task; this note exists so no intervening decision silently inherits the leak.
