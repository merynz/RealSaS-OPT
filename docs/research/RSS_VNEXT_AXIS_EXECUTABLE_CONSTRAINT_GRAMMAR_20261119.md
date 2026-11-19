# RSS.vNext AXIS — executable causal control/constraint grammar

**2026-11-19.** The AXIS V6 product already learns causal joint parents, one
unique local frame per joint, auxiliary chains, variable-arity controls and
MIRA FSQ skin codes on one persistent seeded-TESSA M*. An executable
`CONSTRAINT` is not merely a syntactically valid token. It must contain
enough controls, joint outputs and (where required) a declared chain for
the **same** Mechanics operator that will execute the program.

## Single authority

`compiler/realsas_compiler_core/rig_constraint_schema_v1.py` defines the
actual production vocabulary: IK, AIM, FOLLOW, COPY_TRANSFORM, STRETCH.
Each operator has MIN/MAX control inputs, MIN/MAX joint outputs, and
REQUIRED/OPTIONAL/FORBIDDEN chain address. `PATH` and unimplemented
LIMIT operators have no shipping executor; they cannot be selected just
because the base neural token vocabulary contains them.

AXIS now applies the same operator schema in THREE places:

1. **Generation**: masks impossible constraint-kind transitions before token
   sampling; cannot leave a constraint or begin a new one until its minimum
   dependencies are emitted. Also masks over-max input/output branches.
   Operator logits exclude missing-chain or missing-control choices.
2. **Training**: independently qualified teacher events must contain valid
   executable arity and chain pointers, not an invented empty IK program.
3. **Compiler input**: serialized product event consumers independently
   reject fake, incomplete or unsupported constraints. Compiler does NOT
   synthesize targets or repair parent/control/chain programs.

This is a deterministic structural contract: it does not prove the proposed
controller identity, pole direction, joint limit, attachment, physics,
artist-authored animation, or character semantics are *learned correctly*.
Those require their separate source/native intervention teachers.

## Needed next

- Source-native controls, constraints and retarget clips must be separately
  admitted from licensed authoring data; a valid source *deform rig* alone
  does not prove IK effectors, poles or source motion intent.
- TailorRig-style category-conditioned auxiliary chain training requires
  explicit, source-qualified semantic classes; names such as `Wing_L01`
  are evidence candidates, never automatic license or physical truth.
- Qualified 190+ joint/garment inference, stage-wise timing/memory, LBS,
  native 2D appearance, physics contact and .rss edit roundtrip still need
  real held-out evaluation.
