# Mechanics runtime research — Spine as a design reference, not an equivalent engine

Date: 2026-11-06
Scope: isolated PR #353 / research/mechanics-generic-closure-v1

## Independence and architectural boundaries

Spine is an established **2D skeletal animation runtime**. BUSTED/RealSaS is a compiler-driven editable puppet with a 3D mechanical carrier, 2D presentation/visibility, source evidence and a separately trained upstream IRIS/TESSA/AXIS/MIRA model stack. These systems are not equivalent and must not share an unqualified mechanical truth contract.

The following public documentation is used for *ideas and correctness tests*, not copied source code:

- Spine physics constraints: https://esotericsoftware.com/spine-physics-constraints
- Spine constraint order: https://esotericsoftware.com/spine-constraints#Order
- Spine runtime skeleton transform update: https://esotericsoftware.com/spine-runtime-skeletons
- Spine transform constraints: https://esotericsoftware.com/spine-transform-constraints
- Spine runtime license and SDK terms must be separately reviewed before any source-code reuse; **no Spine source was imported here**.

## Mechanically relevant translation

| Observed practice in Spine | RealSaS research adaptation | Difference / authority |
|---|---|---|
| Ordered constraints, world transform recomputation | Evaluate typed CanonicalMechanics schedule; record primary pose before secondary and Graphics | Our schedule is source-program declared and must prove operator execution, not blindly imitate 2D order |
| Secondary physics with inertia, strength, damping and controlled FPS | Deterministic bounded critically-damped position lag on explicitly declared independent secondary chains | Frequency/angle cap/fixed substep are **runtime policy**, not measured physical constants |
| Physics reset and deterministic timeline scrubbing | Explicit secondary state reset by new state sequence and re-evaluation; fixed substep count and simulation tick receipt | Arbitrary-time seek with history replay is **not implemented** |
| Transform constraints to follow another bone or attachment | Declared `CanonicalAttachmentAnchorIRV2` world position and 3D world basis evaluator | A source-proven grip orientation requires a declared local 3×3 basis; parent basis alone does not mint grip truth |
| IK targeting and constraint mixing | Multigoal rotational FK, world position pins, analytic Jacobian and explicit active-DOF rank selection | Our declared world-space DOFs and source/capability/policy separations are different; full force, contact, mix/priority and oriented IK are **not completed** |
| Runtime cost and stability | Prepared sparse skin, preserved segment length, deterministic warm-start and rank-basis hysteresis | No full renderer/Graphics FPS evidence, no unseen teacher binding |

## Strict research execution

1. Sample editable Motion and transient user control overrides.
2. Solve declared primary IK/control constraints with goal residuals and optional world-position contact pins. Invalid source/program/authority → FAIL.
3. Evaluate read-only position+orientation for declared attachments from primary world pose. Unknown local grip basis cannot satisfy an authored orientation target. Attachment orientation evaluation is **not** a substitute for active orientation IK.
4. Apply secondary deformation exclusively to explicitly declared non-primary and non-attachment-owned joints. Unqualified primary pose or grip target cannot silently drive secondary. Fixed-step internal dynamics integrate with bounded substeps, preserve segment length and angular cone constraints.
5. Deform immutable prepared sparse skin, run full per-vertex and segment metrics, and hand mesh/visibility consequences to Graphics later. Mechanics does not own draw order.
6. Separate continuous-rank active-set *hysteresis* from source rig truth; a previous rank-complete basis may persist, but all original drag and pin goals must still pass. If not, full-declared DOF fallback is a runtime policy, never evidence that the input has fewer anatomical/physical DOFs.
7. Deterministic bounded restarts may address some singular poses; all attempts retain declared bounds/constraint residuals. Finite starts cannot guarantee global convergence.

## Unfinished research and product barriers

- Exact source-authored physical mass, stiffness, wind, gravity, damping and collision behavior is not supplied by the current source truth. No retroactive teacher authority.
- Stable response under long arbitrary variable-FPS trajectories, contact switching, joint limit manifolds and edit timeline seeking/reset/replay remains unverified outside the bounded functional tests.
- A true orientation-aware IK/coupled attachment/grip constraint is not delivered by world-basis reporting alone. Multiple attachments sharing conflicting controls require a separate feasibility/priority court.
- No blind unseen generalization, trained AXIS/MIRA behavior, or real IRIS/TESSA output-source correspondence is proven here.
- No source license or product license assertion is inferred from the ability to read public Spine docs.
