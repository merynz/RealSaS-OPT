# TESSA Shared-Scene Bottleneck Court V1

Date: 2026-10-30
Status: PREREGISTRATION_SOURCE__PLATFORM_EXECUTION_REQUIRED

## Question
Does reducing the IRIS scene to a sparse/re-encoded handoff before TESSA materially damage topology reconstruction compared with a sibling topology decoder that reads the same shared Dense3D scene Z directly?

## Fixed factors
- Same eight-view observations and cameras.
- Same IRIS observation encoder and Dense3D carrier Z.
- Same topology decoder depth/width/token grammar where possible.
- Same topology teacher and free-running decoded-mesh courts.
- No teacher topology is available at inference.

## Arms
B0 REDUCED_HANDOFF: historical reduced surface/endpoint handoff before a second topology encoder.
B1 DIRECT_SHARED_Z: topology decoder reads bounded typed tokens/queries derived directly from the shared Dense3D Z; no 17-D GSA replacement scene.
B2 DIRECT_SHARED_Z_PLUS_LOCAL_REFS: B1 plus typed surface/query references as addresses/local evidence only; refs cannot replace Z.

## Primary metrics
- free-running topology sequence validity;
- decoded mesh surface fidelity;
- endpoint/local relation retention;
- rare thin/contact/grazing topology strata;
- information-recovery control: prediction of scene evidence deliberately omitted by the reduced handoff;
- parameter/compute matched comparison where feasible.

## Interpretation
Direct shared-Z is selected only if it improves or preserves decoded topology while demonstrating that reduced handoff loses recoverable scene evidence. A larger model cannot win solely by parameter count.

## Non-claims
This court decides the neural information boundary only. It does not decide whether TESSA is required for product topology, whether compact topology is mechanically suitable, or whether any candidate is product-authoritative. Canonical execution requires Platform Experiment/Attempt + EngineRelease + CAS artifacts from reviewed main.
