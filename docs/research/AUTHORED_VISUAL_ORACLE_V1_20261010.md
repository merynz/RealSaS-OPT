# Authored visual oracle V1

User-authorized 2026-10-10. Source implementation; Go execution pending.
Protocol: `canonical/AUTHORED_VISUAL_ORACLE_V1_20261010.json`.
Released research graph: `canonical/AUTHORED_VISUAL_ORACLE_RESEARCH_DAG_V1.json`.
The 46-node product graph and Knight M/G/W/motion are unchanged.

## Question and independent reference

Can the existing native source-owned visual consumer reproduce genuine
authored layered 2D evidence when given explicit owner addresses, source sprite
support, local motion and per-frame authored order? This oracle evaluates one
different 2D character in its own Idle/run/jump/walk/hit back/basic attack1 mainline keys.
These 52 keys cover all four authored owner/order configurations found in the
source, including actual order changes inside jump and walk. It does
not retarget this character onto Quaternius Knight or use Knight 3D material
as appearance truth. One V0 in a research RSS is one actual authored view,
never eight copies presented as multiview evidence.

Stage47 computes local SCML hierarchy, spin interpolation and reflected
component transforms. It never uses cached absolute source poses as its motion
input. Stage48 bakes sprite rectangles and fixed UV into the existing RSS
format; authored z-index becomes explicit triangle order for each sealed key.
The existing C++ `REALSAS_V2_SOURCE_OWNED_VISUAL_2D` consumer renders it. Source
RGBA sampling is its existing straight-to-premultiplied legacy visual branch;
this experiment does not exercise the newer CAA linear-material/depth branch.

Stage49 reads the raw ZIP/SCML again. Independently authored cached `abs_*`
poses and cached asset identities drive an inverse-affine sprite compositor.
Native triangle rasterization, hierarchy interpolation and corner construction
are not its reference algorithm. Pose residuals compare computed hierarchy to
cached source transforms. The cached poses are authoring evidence, not independent
human aesthetic judgment; any inconsistency remains a failing measurement.
Existing baked PNGs have unqualified export timing and are not silently matched
by image index. No continuous inter-key fidelity is claimed.

## Producer / consumer / measurement

| Contract | Producer | Actual consumption / independent measurement |
|---|---|---|
| Owner and source address | entity+authored-object ID, distinct even when PNG is reused | Owner table sealed inside RSS; native per-pixel face IDs mapped and compared to raw-source owner raster |
| Motion | SCML local transforms, hierarchy, linear/instant interpolation | Sealed RSVP vertices rendered; absolute source pose residuals and independent raster comparison |
| Material | Exact native-alpha sprite PNGs and transparent atlas gutters | RSS texture UV sampling; raw RGBA premultiplied error/alpha IoU |
| Intended order | Mainline z-index with duplicate rejection | Baked face order; independently composited source order and wrong-order control |
| Occluded sprite material | Authored full isolated sprite support | Initially opaque texel centers covered by a different robust owner, subsequently visible canonical owner+image+texel addresses; count correct native owner and alpha |
| Visual contact / grip | Required-domain annotations absent | Unknown cardinality, no PASS claimed |
| 3D mechanical bind / CAA completion | Not produced by this 2D source experiment | Not qualified or claimed |

Opaque source texels hidden initially are only admitted if their center is
onscreen and covered by another robust owner. Unseen texels from undersampling,
transparent support or cropping are excluded. This measures consumption of
authored sprite material, not amodal anatomy inference or CAA completion.

Budget values are sealed before real-source execution. A failed pixel/pose
measurement is retained in `measurement.json`; no threshold adjustment closes
it. Negative controls reverse face order, omit cape material, rotate semantic
owner labels while leaving pixels intact, and shift head motion by12pixels.
The ownership control must fail even when image/alpha checks pass. These
variants are declared experimental outputs of one consumer stage, not child
Attempts that purport to prove platform intervention reuse.

Compiler stage PASS means the declared experiment produced exact measurements.
The measurement separately records baseline fit, hidden support coverage and
negative-control detection. Attempt COMPLETED or a green workflow does not
mean visual quality PASS. Full visual qualification always remains false.

## Canonical execution and decision boundary

The one-shot native-host workflow waits for exact main deployment. Its operator
verifies clean live main, hydrates pinned source bytes and imports exact manifest
sections, seals a RESEARCH EngineRelease/SubjectInput and creates one Go Attempt.
It reads agent-context, enters the exact scope, submits the idempotent research
command, inspects its terminal state and exits with the actual artifact binding
handoff. A recovery checkpoint stores returned API IDs; it does not own state.
The checkpoint prevents a timeout from triggering a new Attempt. A source SHA
change requires inspection and an explicit child intervention. No direct stage
execution or ProductRevision is used.

The source/parser, native package/consumer and measurement adapters are separate
implementation closures. Editing the package consumer must leave producer
identity intact. The measurement reads raw source independently as a declared
input in addition to producer/consumer outputs. Product runtime behavior is
unchanged; this oracle uses a research-only package writer without fabricated
mechanical or eight-camera qualification hashes.

If this limited oracle passes, it supports a layered appearance representation
for this authored source and sampled motion. It does not prove that IRIS can
predict it, that a 3D-to-2D binder is correct or that current product Stage37/42
consumes all five missing visual relations. Those require separate typed
evidence, a genuine same-character reference/bind and independent contact/grip
domains. If it fails, inspect pose, pixels, ownership and hidden-address counts
before changing model architecture. A missing required domain stays missing.
