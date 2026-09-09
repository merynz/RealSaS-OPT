# Geppetto Reference-Strength V1 — FIT1 Frozen Mainline

This package is the **current FIT1-frozen Geppetto formulation** promoted after the sealed Mage reference-strength closure.

Scientific source authority:

- frozen optimizer/source commit: `f7be46f0a97df62a793ebf91b22297c894854f39`
- seal commit: `ae0af0cd39dd2468a012ba21890a4fed2da7c4c9`
- closure: `canonical/GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md`
- mainline promotion authority: `canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md`

## Frozen formulation

`RiggingSurfaceIR -> lossless fieldwise tensorization -> direct surface encoder + exact GSA relation message passing + full-surface transformer memory -> prediction-only causal control recurrence with per-step full-surface cross-attention -> coarse 3D locus + conditional residual diffusion -> native STOP/existence/root/salience/support evidence + all-pairs directed parent evidence -> SkeletonProposalIR -> Compiler exact qualification`

The shipping entry point is:

`GeppettoReferenceStrengthNoLearnedSlotV1`

Architecture id:

`RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion.DeterministicViewDirection.v1`

The canonical eight yaw views are represented by a fixed Fourier camera-direction buffer. There is no learned absolute V0..V7 slot embedding in the frozen model.

## Source preservation

`geppetto_reference_strength_candidate_v1.py` and `rigging_surface_tensorization_v1.py` are byte-preserved copies of the sealed FIT1 source blobs:

- candidate blob: `e9b626815c63a96ac1d390580e1aa19f8bf1cfb2`
- tensorization blob: `647fdcb98c305c3f7d3afd05dd61449999721034`

The no-learned-slot entry file is a semantic-home rebind of frozen blob `770909a9c2992ada0f3018badb2a42632a124b3b`; its algorithm is unchanged and its import now points at this package's promoted base candidate.

The byte-preserved base candidate retains one historical import path to the tensorization module. For reproducibility, the exact tensorization source is also preserved at `experiments/geppetto_reference_strength_fullstack_v1/rigging_surface_tensorization_v1.py` on main. This is compatibility residue, not a second semantic implementation.

## Authority boundary

Geppetto emits proposal/evidence only. It does not own canonical joint IDs, final root/tree selection, graph legality, product state, or proof. `compiler/realsas_compiler_core/` remains the authority that qualifies the proposal and mints `QualifiedSkeletonIR`.

## What the FIT1 freeze proves

On the real Mage FIT1 witness, the frozen formulation reached the preregistered terminal gate at optimizer step `14080` and maintained `48/48` consecutive full structural PASS checks across `3072` optimizer steps, with native STOP/count, no teacher feedback during free-running inference, a qualified 22-control mechanical core, and four frozen diffusion evaluation seeds.

It does **not** prove unseen-character/family generalization. It also does not imply Arachne or product closure.
