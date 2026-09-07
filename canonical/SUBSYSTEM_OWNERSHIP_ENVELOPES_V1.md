# RealSaS — Subsystem Ownership Envelopes V1

**Status:** `BINDING_CONTINUITY_GUARD`

This document is the canonical shorthand-expansion guard for reasoning about RealSaS subsystems. A learned model is never the whole subsystem. When a continuation says **IRIS**, **Geppetto**, or **Arachne** in a system-level discussion, expand the name to the complete ownership envelope below unless the discussion is explicitly restricted to model internals.

Primary historical ownership audit: `canonical/DETERMINISTIC_DOWNSTREAM_LAYER_OWNERSHIP_OVERLAP_AUDIT_20260831.md`.

## Global rule

> **One semantic owner per question. Qualification may constrain; it may not secretly re-infer proposal semantics. Repair routes to the owner.**

Do not move a learned responsibility into deterministic code merely because a deterministic heuristic can make one witness pass. Do not move exact legality/canonicalization into a learner merely because the learner can predict a likely answer.

## 1. IRIS envelope

### System shorthand

`IRIS := observations/cameras -> learned IRIS evidence -> deterministic GSA/RiggingSurfaceIR assembly/validation/provenance -> learned consumers`

### Learned IRIS may own

- observation-conditioned signed geometry/evidence;
- support/existence/confidence/uncertainty evidence;
- multimodal observation evidence that is not analytically identifiable;
- learned appearance/spatial descriptors within the sealed observation-only input contract.

### Deterministic upstream / surrounding authority

- observation and camera contracts;
- allowed analytic camera relations;
- production-input firewall and provenance/hash validation;
- fail-closed materialization/resource checks.

### GSA / RiggingSurfaceIR owns deterministically

- analytic point construction from admitted signed/depth evidence and exact camera geometry;
- surface-evidence assembly, support admission, provenance and view/raster binding;
- lossless packing/normalization contracts;
- deterministic compaction and qualified local geometric operators when explicitly assigned there;
- preservation of UNKNOWN/unobserved state rather than hidden completion.

### IRIS must not own

- final canonical skeleton/control IDs;
- legal root/parent/tree selection;
- canonical skin weights;
- hidden/full character completion authority;
- final product graph or proof authority.

### Memory guard

If someone says “IRIS outputs the surface,” first ask internally whether they mean learned signed evidence or the **GSA-qualified RiggingSurfaceIR**. The latter is a deterministic surrounding layer, not raw model authority.

## 2. Geppetto envelope

### System shorthand

`Geppetto := GSA/RiggingSurfaceIR -> Geppetto learned SkeletonProposalIR/evidence -> Compiler exact graph qualification -> QualifiedSkeletonIR + canonical IDs`

### Learned Geppetto may own

- whether a mechanically relevant control is proposed / needed;
- control locus and uncertainty;
- root evidence;
- directed parent/attachment evidence;
- learned duplicate/sameness evidence where part of the proposal contract;
- **mechanical salience / functional simplification evidence**: whether an artist/source control is mechanically necessary or can be omitted without losing required deformation behavior.

### Deterministic conditioning seam may own

- normalization, padding, hashing and typed packing;
- preservation of lossless per-view evidence;
- deterministic serialization required by a preregistered experiment, provided it does not infer missing semantics.

### Compiler graph qualification owns deterministically

- admissible final root/parent/tree selection from supplied evidence;
- exact graph legality, connectivity/cycle/tree invariants;
- canonicalization and canonical ID minting **after exact solve**;
- provenance and fail-close;
- explicit rejection when proposal evidence is insufficient.

### Compiler must not do for Geppetto

- geometry-only semantic deduplication;
- hidden cardinality repair;
- missing deform-node synthesis/completion on the product route;
- infer mechanical salience from legality/geometry heuristics;
- silently convert an invalid proposal into a plausible rig and call it a model success.

### Memory guard

> **Geppetto is proposal, not canonical rig authority.**

If a future chat says “Geppetto made the skeleton,” expand that statement to: “Geppetto proposed controls/relations; the Compiler qualified the legal canonical skeleton.” If a proposed Geppetto repair actually adds semantic work to the Compiler, flag the ownership drift before accepting it.

## 3. Arachne envelope

### System shorthand

`Arachne := qualified IRIS/GSA surface + Compiler-qualified skeleton + learned Arachne skin/deformation proposal -> Compiler skin/mesh qualification -> qualified editable deformation state`

### Learned Arachne may own

- semantic per-control influence / dense weight proposal;
- learned deformation/skin field conditioned on the qualified skeleton and observation evidence;
- uncertainty/evidence needed to judge skinning quality;
- future mechanisms adopted after clean-room SkinTokens analysis when explicitly preregistered.

### Compiler skin/mesh qualification owns deterministically

- canonical joint-reference validity;
- finite/non-negative/simplex constraints;
- bounded top-k projection where specified;
- mesh/layout/reference integrity;
- deterministic editable discretization/projection;
- fail-close when a learned proposal cannot be legally represented.

### Deterministic optimizers may own only bounded numerical roles

BBW/QP/KKT or similar solvers may project/regularize **under Arachne-provided semantics**, or exist as an explicitly named alternative proposal arm. They must not become a hidden second semantic skinning owner.

### Arachne/Compiler must not do

- Compiler independently inventing semantic weights and treating them as Arachne success;
- Arachne minting legal/canonical skeleton IDs;
- hidden surface completion changing observation truth;
- shipping-time teacher/source-mesh authority unless explicitly part of a non-product diagnostic.

### Memory guard

“Arachne passed” is incomplete unless the statement says whether it means learned proposal quality, deterministic qualification, verified deformation, or the complete qualified deformation route.

## 4. Canonical product / proof / runtime envelope

`Qualified IRIS/GSA evidence + QualifiedSkeletonIR + QualifiedSkin/mesh -> CanonicalPuppetGraph -> exact-state proof -> PASS-only runtime/export projection`

- `CanonicalPuppetGraph` owns product-state lineage/hash; it does not re-solve model semantics.
- deformation/contact solvers (ARAP/XPBD/etc.) are downstream numerical behavior, not rest-rig semantic owners.
- proof measures the exact candidate product state and returns evidence/pass/fail; proof does not mutate the candidate.
- owner attribution diagnoses which owner should change; it is not silent repair.
- repair is bounded, owner-routed, creates a new candidate state, and requires re-proof.
- runtime/export is a projection of a qualified/proven product state; it is never a second canonical truth.

## 5. Shorthand expansion rules for agents/chats

When the user says only a subsystem name in a system/scientific discussion:

- **“IRIS”** -> recall `observation contract -> learned evidence -> GSA/RiggingSurfaceIR`.
- **“Geppetto”** -> recall `lossless RiggingSurfaceIR -> learned proposal -> exact Compiler qualification/canonical IDs`.
- **“Arachne”** -> recall `qualified surface+skeleton -> learned skin/deformation proposal -> deterministic skin/mesh qualification`.
- **“Compiler”** -> recall it is a family of deterministic qualification/proof/routing responsibilities, not permission to repair missing learned semantics.

Before moving a responsibility, answer:

1. Is the question semantic/uncertain and therefore learned-owner evidence?
2. Is it analytic/contract/provenance/legality and therefore deterministic?
3. Would this change create a second semantic owner?
4. Would it make a model appear successful by repairing its output downstream?
5. Does the resulting state still fail closed when evidence is insufficient?

If any answer is unclear, the responsibility remains unresolved; do not silently assign it.
