# IRIS vNext — Single-Carrier Mechanics + Graphics

Date: 2026-10-16
Status: user-authorized architecture; implementation review branch

## Non-negotiable invariant

RealSaS has exactly one geometric carrier after candidate selection:

```text
candidate M* set
  -> static admission
  -> candidate AXIS / MIRA
  -> bounded mechanical selection probes
  -> Compiler survivor selection
  -> one canonical M / G / W_M
```

Candidate meshes are immutable proposal evidence only. They never become parallel product authorities. Graphics, Motion and Runtime may bind only to the selected canonical carrier. An independent deformable visual mesh is forbidden.

This preserves the lesson from the Knight line: a system in which mechanics move correctly while a separate visual carrier disappears, drifts or tears is architecturally invalid.

## Model

```text
8 x RGBA + cameras
        |
 frozen DINOv2-S
        |
 shared scene core Z
   /        |         \
geometry  graphics   structure
field     evidence    memory
H(x)      relations   mech demand
Flexi     appearance  TESSA grammar
   \        |         /
     candidate M* set
```

There is no named semantic taxonomy. Structural knowledge is exposed only as role-free evidence.

### Mechanical-demand supervision

During dataset construction, authored M/G/W and native animation may produce teacher truth:

```text
authored animation
 -> per-edge strain
 -> per-face deformation/Jacobian
 -> area change
 -> principal stretch direction
 -> bend demand
 -> rigidity
 -> optional contact persistence
 -> MechanicalDemandTruth
```

At inference those authored mechanics are absent. Static visual evidence predicts MechanicalDemandEvidence; the topology decoder uses it to learn what kind of mesh can survive the expected mechanical life of the surface.

References: Animation-Aware Quadrangulation (CGF 2013), NVIDIA FlexiCubes (SIGGRAPH 2023), StdGEN++ (arXiv:2601.07660).

## Geometry / extraction

vNext baseline: scene width 512, 6 scene layers, 8 heads, triplane 3 x 96 x 64 x 64, field width 384, 6 residual blocks, explicit H(x), bounded extraction displacement and 21-D cube extraction configuration. Flexi-style parameters are extraction freedom, never MIRA skin weights.

## Structure

TESSA survives as a topology language/decoder, not a second scene-understanding model. Baseline: 512 width, 8 heads, 8 decoder layers, 256 memory tokens. Structure memory sees triplane evidence plus SurfaceWitness XYZ/normal/H(x)/uncertainty/MechanicalDemandEvidence. Decoder may emit K diverse candidates (default K=4). Compiler does not design a replacement mesh.

## Canonicalization

The old timing `static M -> canonical M -> AXIS/MIRA -> dynamic test` is retired in vNext.

```text
M* candidates
 -> static admission
 -> candidate G/W
 -> topology-selection mechanical probes
 -> deterministic survivor selection
 -> canonical M/G/W
```

A separate post-canonical production motion proof remains. The two dynamic courts must not be conflated.

## Graphics

`CAA` is retired as canonical product vocabulary. Historical CAA artifacts/codecs remain readable for lineage compatibility; active product domain is **Graphics**.

Graphics consumes source-wins observed RGBA/material evidence, learned role-free relation evidence, physical depth/visibility from canonical M, and canonical M/G/W motion/contact evidence.

Learned evidence includes same_drawable_affinity, cut_probability, fragment_embedding, visibility_probability, front_probability, occlusion_probability, rigidity_probability, attachment_affinity, alpha_boundary_probability, appearance_boundary_probability and uncertainty. Compiler globally solves these into GraphicsFragments, per-view partial order, visibility state, attachment anchors, fragment motion fields and bounded silhouette support bands. Model never invents persistent fragment IDs.

Current Stage37 mainly derives boundary cuts/face groups; current runtime can verify a graph binding without behaviorally consuming its contents. vNext forbids that state. Face ownership, per-view order, visibility and attachments must be materialized into runtime data.

## 2D support

Canonical M remains the only 3D carrier. A bounded 2D support sidecar is allowed, not another mesh:

```text
posed canonical M/G/W
 -> fragment projection
 -> FragmentMotionField
 -> bounded SilhouetteSupportBand
 -> final Graphics composition
```

This follows the useful Paperman separation between 3D mechanics and 2D drawing support.

## Compiler rule

> Compiler may enforce truth; Compiler may not compensate for model incapacity.

Allowed: exact validation; IDs/hashes/addressing; bounded numerical sanitation; global constraint solving; candidate selection; exact contact/barycentric solves; fail closed.

Forbidden: large repair rescuing weak learned output; semantic skin transfer hiding carrier drift; any second visual mesh.

## Trainer

Old subject-serial trainer is retired for vNext. One update is a real cohort batch: eight resident/pinned subjects -> B=8 scene encode -> batched queries/courts -> one backward -> fused AdamW. CUDA uses BF16/SDPA/fused kernels where supported. Reproducibility is anchored by immutable inputs and deterministic FP32 evaluation courts.

## Fit8 re-entry gates

The old ~5400-step run is preserved only as a control artifact. Before new scientific Fit8: vNext contracts pass; single-carrier authority contracts pass; Graphics runtime-content contract passes; batched A100 engineering preflight passes; MechanicalDemand and anonymous GraphicsRelation truth exist for all eight; no subject-specific branch exists; exact input/policy/model binding is sealed.
