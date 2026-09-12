# RealSaS — Generic component / attachment contract — 2026-09-12

**Status:** `DRAFT_AUTHORITY_FOR_MAGE_RECLOSURE__GENERIC_NOT_MAGE_SPECIFIC`

## Motivation

A product observation may contain multiple visible components with different mechanical semantics. A full character cannot be reduced to one anonymous deformable surface if source/product evidence supports rigid joint-bound accessories or other distinct component behavior.

Mage is the first hard witness used to close this generic contract because its source contains body, book variants, wand/staff variants, hat and cape in one asset.

## Required classes

Every admitted visible component must resolve to exactly one mechanical class before product qualification:

- `DEFORMABLE_COMPONENT` — deformation is governed by multi-joint skin/deformer state;
- `RIGID_SKINNED_COMPONENT` — all admitted component vertices are rigidly carried by one canonical joint through skin authority;
- `RIGID_BONE_ATTACHMENT` — source/provenance explicitly binds the component rigidly to a bone/socket without deformable skin semantics;
- `NON_MECHANICAL_VISUAL_COMPONENT` — visible in product observations but deliberately excluded from mechanical deformation authority; requires explicit visual-only qualification;
- `EXCLUDED_SOURCE_COMPONENT` — source provenance exists but component is not admitted to product observations/mechanics; exclusion reason is mandatory.

`UNKNOWN` / `AMBIGUOUS` is not a qualifying product class. It must block final qualification for a required visible component.

## Rigidity is not detachability

`rigid == detachable` is forbidden.

Detachability / swappability is an independent property:

- `FIXED_COMPONENT`
- `DETACHABLE_COMPONENT`
- `SWAPPABLE_SLOT_COMPONENT`
- `DETACHABILITY_UNKNOWN`

No detachable/swappable behavior may be inferred from a component merely because its weights are one-hot.

## Required typed authority

A qualified component record must include:

- stable `component_id`;
- source component/provenance refs;
- product observation membership by view;
- mechanical class;
- canonical parent joint or socket when applicable;
- bind-state authority hash;
- surface membership / geometry lineage;
- skin/deformer lineage when applicable;
- directional render membership;
- detachability class;
- qualification report;
- component lineage hash.

A one-hot row in `QualifiedSkinIR` is evidence for rigid carry but **is not sufficient product-level assembly semantics by itself**.

## Compiler ownership

The Compiler owns qualification of component class and canonical binding. Learned systems may propose evidence but may not assign canonical component IDs, joint/socket authority or detachable semantics.

## Full-subject invariant

For every visible source/product component admitted by the observation contract, one of the following must be true:

1. it is represented by qualified mechanical geometry and typed component semantics; or
2. it is explicitly qualified as visual-only; or
3. it is explicitly excluded with a product-valid reason.

Silent disappearance is forbidden.

## Mage witness obligations

The reclosure must account for every Mage rendered component, including at least the body set plus book, wand/staff, hat and cape families present in source provenance. Exact component classification must follow source authority, not filenames alone.

Known source-side evidence already recovered during audit includes one-hot rigid carry for book -> left hand slot, wand/staff -> right hand slot, hat -> head and cape -> chest. Those facts are evidence to validate the generic contract; they must not be hard-coded into generic implementation.

## Product gates

- visible required component accounting = 100%;
- no required component with `UNKNOWN` / `AMBIGUOUS` final class;
- rigid component has exactly one qualified canonical carry binding;
- deformable component has qualified skin/deformer authority;
- visual-only component has explicit visual qualification and is forbidden from masquerading as mechanical geometry;
- component lineage participates in product-state hashing;
- directional renderable membership cannot silently merge or drop components.
