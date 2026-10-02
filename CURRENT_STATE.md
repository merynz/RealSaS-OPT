# RealSaS-OPT — Current State

**Date:** 2026-10-02  
**Canonical continuation branch:** `main`  
**Active experiment:** `KNIGHT_APPEARANCE_RUNTIME_DATAFLOW_AUDIT`  
**State:** `MAIN_CANONICALIZED__FROZEN_WITNESS_VERIFIED__RUNTIME_DATAFLOW_AUDIT_ACTIVE`  
**Certification rule:** the current `main` head itself must have completed green CI; generated-view bot commits are followed by a human/tool certification commit.

## Canonical lineage

The current main combines the latest preserved Knight R&D lineage with the Oct-1 recovered engine/platform implementation.

- Machine lineage record: `canonical/CURRENT_RND_LINEAGE_V1.json`
- Current producer/consumer graph: `canonical/APPEARANCE_RUNTIME_DATAFLOW_GRAPH_V1_20261002.json`
- Human audit: `docs/audit/APPEARANCE_RUNTIME_DATAFLOW_AUDIT_20261002.md`
- Current 46-stage V2 scientific execution plan: `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
- The Go platform/control plane does **not** hard-code a 46-stage cardinality; 46-stage refers to the current V2 scientific plan.

Historical/recovery readiness remains preserved at `canonical/V2_IMPLEMENTATION_READINESS.json`, but its former VF11 continuation state is superseded by the current Knight appearance/runtime audit.

## Accepted Knight witness

The corrected-weight/topology harmonic-first Knight mechanical-CAA witness was recovered byte-for-byte:

- IDLE `4c64437ba1f2ecaca8302a52c598f3ba7750a923fd50b90f7cafa3f8473386c3`
- RUN `64ad50eeb9ea55ce202ddc46728ffea9074419dbf855938840f69d06661806e3`
- SLASH `adc8e8cd0e2272a836e50f7a73585208074703ea5f3c9a549583d7a659fa2b85`

This is an R&D/demo witness, not product visual authority.

## Current product authorities

- **Geometry** owns canonical renderable surface, topology and stable `SurfaceAddressing`.
- `QualifiedMeshIR` is the qualified mechanical/product mesh authority after the V2 mesh gates.
- **Mechanics** owns skeleton, skin, deformation, contacts and motion.
- **Appearance** owns the **Complete Appearance Authority**: source-preserving art, provenance, completion and appearance qualification.
- **Presentation** owns role-free slots/attachments/grouping; first-witness Runtime V2 intentionally excludes authored runtime order/visibility/clipping execution.
- Physical visibility remains **posed canonical XYZ + camera depth** unless a separately qualified presentation authority explicitly supersedes it.

## Proven runtime data-flow findings

1. The historical Knight path contained a real Stage37 → Stage42 authority discontinuity: Stage37 could declare source-owned visual authority while historical Stage42 reopened the mechanical mesh + mechanical CAA lane.
2. The recovered engine implements the typed source-owned path:
   `Stage18 VisualMeshSetIR → Stage37 QualifiedVisualPresentationSetIR → Stage38 CompletePuppetStateIR → Stage42 SourceOwnedVisualRuntimeProjectionIR → Stage43 RSS → Stage44 native source-owned renderer → Stage45 source-owned visual integrity`.
3. The recovered source-owned path still has two unclosed contracts:
   - visual-triangle deformation-domain coherence is not guaranteed because mechanical affine binding is solved per visual vertex;
   - canonical physical depth/overlap ownership is not transported to native source-owned playback, where face-array order currently acts as an implicit overlap authority.
4. Stage45 currently proves native/reference parity, direct-source provenance, non-empty frames, flips and catastrophic stretch, but does not yet prove overlap-owner/physical-visibility correctness.

## Immediate execution

Close, in this order:

A. **Visual triangle deformation-domain coherence**  
B. **Canonical depth → visual overlap ownership**  
C. **Stage45 proof of A + B**

Do not select RGB generation, a new texture architecture, or another appearance representation until these existing contracts are measured and either closed or falsified.
