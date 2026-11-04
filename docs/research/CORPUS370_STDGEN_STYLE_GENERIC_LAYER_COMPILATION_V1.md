# Corpus370 / RealSaS: StdGEN-style decomposition without an ontology

**Decision:** Adopt a generic, **operational drawable compilation** objective.
Transfer StdGEN's *layer-field / selective peel / independent extraction*
mechanism but **not** VRM semantic labels, names, fixed class count, or
claim that an arbitrary source artist-defined drawable partition is unique.

**Scope:** architecture / qualification contract. No Graphics owner training
authority, release/merge, or finalized positive/negative supervision from
this design document.

## What is actually transferable from StdGEN

Public StdGEN (CVPR 2025, He et al.):
- S-LRM reconstructs 3D geometry/density, color and **semantic probabilities**
  from multi-view inputs, with multi-layer semantic image supervision.
- Semantic-equivalent NeRF replaces opacity at a sampled 3D point by the
  probability-weighted per-semantic opacity, so hidden semantic layers may
  be rendered separately.
- Semantic-equivalent SDF uses `max(f(x), max_{r!=s} p_r(x)-p_s(x))`
  to extract per-semantic surfaces from an underlying shared implicit field.
- Its public Blender teacher uses material/object names to assign fixed
  Hair/Cloth/Face/Body groups, rendering FULL, WITHOUT_HAIR, WITHOUT_CLOTH
  and semantic masks, as well as depth/normal and alpha.

References:
- https://openaccess.thecvf.com/content/CVPR2025/papers/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.pdf
- https://github.com/hyz317/StdGEN/blob/main/blender/blender_lrm_script.py

The **transferable principle** is to create and supervise multiple *surfaces
under selective suppression* from a **shared** scene. The predefined
semantics are their method for selecting which layers to suppress, not a
mathematical requirement of the compositing mechanism.

## RealSaS generic replacement: anonymous editable-layer field

Let `Z` be IRIS shared Dense3D memory with TESSA surface carrier.
For each source-canonical surface locus `x`, predict:
- `f(x)`: source-equivalent geometry/density/surface carrier (IRIS authority);
- `c(x)`: graphics appearance with CAA source address (independent graphics);
- `p_k(x)`: probability assigned to anonymous *graphics layer* slot `k`;
- `q_j(x)`: **separate** mechanical control/attachment/skin response slot;
- `d_{k,t,v}`: per-layer draw order / depth-visibility in time and view.

`k` and `j` are **not semantic categories** and need not have equal
cardinalities. A wing membrane may be one graphic layer with multiple
mechanical influences. A single solid torso may be split into distinct
presentation layers because a weapon/hand must occlude correctly.

The renderer may use StdGEN-style per-slot alpha compositing:
`alpha_{i,k} = alpha_i * p_k(x_i)` along the full ordered ray
and extract candidate layered surfaces from compatible shared surface
carriers, without changing IRIS Dense3D or turning source faces into
inference-time IDs. Do not naively apply global-solid signed-distance
extraction to an open source surface: retain explicit solid/source surface
separation and all existing geometry courts.

The number of active `k` slots is variable. Match slots by their
source-locus/effect support across views and frames (permutation-invariant
matching and stable quotient identity), **never their index or semantic
name**. Preserve source identity as separate authority. A slot split or
merge is a compiler decision, not a ground-truth part name.

## The key conceptual shift: compiler-defined editability, not original
## author-part recovery

Many partitions of the same geometry render identically. Even complete
motion and rig truth does not uniquely recover the artist's intended
2D drawable boundaries. This is an identifiability fact; holding all
models hostage for unknowable same/different *author* labels is unsound.

**The product's useful definition** is a *minimal operational partition*
`P*` that reproduces the original character and supports independent
graphics edits and mechanical controls at acceptable granularity:

`P* = argmin_P (L_source_render + lambda1*L_edit_leak +
lambda2*L_motion_replay + lambda3*L_occlusion_seam +
lambda4*L_complexity)`

subject to hard constraints:
- source geometry and appearance conservation (CAA);
- full-ray occlusion, hidden-face reappearance, cross-view draw order;
- no-op parity and independent replay of controlled interventions;
- lineage/persistent source identity, valid skin/deform/canonical mapping;
- no removal of legitimate decorative or open surface components;
- abstain on ambiguous edit granularity, don't assert author identity.

`L_complexity` prevents "one drawable per triangle"; held-out edit
independence prevents "everything one drawable". The choices of
regularization/thresholds are **product policy** and must be evaluated,
not relabeled as observed source truth.

**Important:** a teacher can recolor any arbitrary face, so a recolor
difference only demonstrates **source contributor existence**.
It cannot alone prove a distinct *editable puppet layer*. Likewise,
co-motion is not an equivalence constraint: a single skinned drawable
may bend while independent objects may move under one root.

## Evidence and training route

1. **Evidence collection (existing, measured):** immutable source-rest,
   complete geometric ray stack, FULL/NOOP/REMOVE(A)/REMOVE(B)/REMOVE(AB),
   reciprocal recolor, alpha/depth and motion, 8-view canonical frames;
   existing 3-source + 8-new-source pilots prove bounded visibility,
   not anonymous owner slots.
2. **Generic proposal producer:** from source surface graph, seams,
   occlusion constraints, source-barycentric loci, independently measured
   bone-control effects and CAA, form *anonymous atomic patches*. Generate
   multiple candidate partitions via cut/merge, **without** names,
   material identifiers as part labels, fixed humanoid topology, or
   outcome-based retrospective subject selection.
3. **Held-out consumer court:** for each candidate partition, instantiate
   editable layer composites in 2D/3D hybrid renderer. Check source
   appearance, occlusion, hidden edit recovery, seam fidelity,
   controlled independent edits, action replay and topology/attachment
   under *held-out* views, poses and commands. Compare one-layer,
   per-face, connected-component, rig-only and anonymous hypotheses.
4. **Qualification:** if multiple partitions meet the hard constraints,
   retain an operational minimal-complexity winner plus equivalence or
   ambiguity information. The winner is **COMPILER_OPERATIONAL_LAYER**
   authority, NOT "the source author drew these as one part".
   A failed functional edit is a measured negative *consumer consequence*;
   missing edit intent remains UNKNOWN.
5. **Model training:** derive permutation-invariant slot masks from
   independently courted compiler-operational outputs only. Train a
   source-image-to-anonymous-layer probability head (IRIS/TESSA consumer
   design choice, not automatically a new model). Held-out proof must
   show slot predictions preserve live puppet editing, not only IoU.
   Privileged source triangle, bone, material, semantic/object names and
   intervention commands may be teacher controls and **must not** be
   visible to student inference.
6. **Global admission:** separate family authority matrix, lineage-safe
   unseen across all derived views/animations, optional family masks.
   Geometry-only IRIS eligibility can be sealed independently; do not
   encode unknown owner or open-solid sign as zero.

## Hard prerequisites vs parallelism

- The proposed anonymous-slot *architecture*, teacher generator, baseline
  scorer and source-lineage gating can be built **now in parallel** with
  real bone interventions, CAA and TESSA work.
- Actual product owner supervision needs renderer/consumer held-out
  consequences and an explicit operational granularity policy. Existing
  6- and 16-pixel visibility trials are not a complete segmentation
  dataset, and successful source rig-control perturbations alone do not
  satisfy this requirement.
- 370 full-source render/patch extraction should run on hosted CPUs after
  small *precommitted* oracle/negative tests, with independent small
  per-source failures retained; then admit only source-verified masks.

**Verdict:** A morphology-generic StdGEN-like layered reconstruction is
technically viable and better aligned with the actual **editable puppet**
product than trying to infer human-authored semantics or await impossible
unique ground-truth drawable owner partitions. Its major research risk is
**stable, useful operational layer granularity**, not semantic naming.
