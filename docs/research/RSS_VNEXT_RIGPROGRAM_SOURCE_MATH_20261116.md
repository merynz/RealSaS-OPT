# RSS.vNext — Source-grounded RigProgram mathematics (2026-11-16)

## Boundary

One RSS.vNext model and the existing AXIS V6 RigProgram grammar remain authoritative.
This change adds learned, independently qualified mathematical consequences to
the AXIS–MIRA trainer. It does not make source segmentation a physics/material
teacher, does not import TailorRig's model/code/dataset, and does not change the
canonical runtime or silently activate XPBD. It is an original implementation
of standard geometric, probability and LBS losses informed by research.

## Field-by-field status

| RigProgram field | Source mathematics / learned contract | Existing downstream execution | Still requires independent evidence |
| --- | --- | --- | --- |
| JOINT | Discrete xyz expectation vs source position; point-to-ground-truth bone segment projection | Parent graph, joint frames and FK/IK | True joint-to-source index mapping |
| FRAME | 6D basis cosine loss; masked BCE on independently known DOF axes | Canonical joint frames; DOF | Authored limits or independently observed interventions; unobserved = UNKNOWN |
| CHAIN | Disjoint maximal source unbranched bone chains; chain-weight KL | Parent-contiguous chain validation | True source parent graph and full skin matrix |
| CONTROL | Existing teacher-forced causal kind/target CE; added smooth L1 source rest position | Existing runtime ROOT/FK/IK/pole/socket controls | Source-authored control object, real intervention to prove target response |
| CONSTRAINT | Existing operator/connection CE and four-parameter smooth L1 | IK, AIM, FOLLOW, COPY_TRANSFORM, STRETCH are implemented | Independent control-consequence truth; PATH/LIMIT operators are not promoted |
| ANCHOR | Existing parent + coordinate grammar and independent source face correspondence; added native local xyz loss | Socket/attachment runtime | Source-native anchor identity and held/holder evidence, never proximity alone |
| SKIN_CODE | Frozen per-joint FSQ codec + bone KL + source-derived chain KL + optional native posed-LBS | MIRA hard code decode and canonical LBS | Full source weights, exact mesh reindex, source native nonrigid poses |

## Maths

For source joint p_j and source parent p_parent, predicted point p_hat is
penalized against the closest point on the ground-truth segment [p_parent,p_j].
Compute t = clamp(dot(p_hat-p_parent, p_j-p_parent) /
norm2(p_j-p_parent),0,1);
L_bone_projection = mean_j norm2(p_hat-(p_parent+t*(p_j-p_parent))).

The exact source parent hierarchy is partitioned into maximal unbranched bone
chains, with root singleton chains preserving direct root skin weights. For
membership matrix M[c,j], source bone weight W_gt[v,j] and decoded student
weight W_pred[v,j], compute chain mass by summation:
W_gt_chain[v,c] = sum_j M[c,j]*W_gt[v,j].
The chain loss is KL(W_gt_chain || W_pred_chain) averaged per vertex.
The bone loss is KL(W_gt || W_pred) averaged per vertex.
**This is a chain-supervised MIRA decoder, not TailorRig's separate
chain-first neural predictor.** It does not invent semantic classes.

For independently observed source pose matrices T_gt[p,j] and exact-index
rest vertices v_gt, L_posed_lbs is the vertex-mean absolute difference
between source posed vertices and the LBS result from **predicted** skin and
the source bone transforms. Global-rigid-only pose batches are ineligible.

CONTROLS, CONSTRAINTS and ANCHORS remain separate proof classes. An authored
controller rest coordinate is not an IK response; skeleton proximity is not
a weapon-grip source; a 6D frame alone does not establish locked DOF.

## Trainer binding and source eligibility

- New trainer weight: axis_rig_grounding, default 0. Existing checkpoint
  objective fingerprints are unchanged when disabled.
- New optional batch tensors: axis_math_joint_xyz, axis_math_parent_indices,
  axis_math_skin_weights and source-qualified optional frame, DOF, control,
  anchor, native posed LBS evidence.
- Requires PRODUCT_MESH_MATCHED_REQUIRED, existing source-native AXIS teacher,
  pinned frozen MIRA FSQ and same source-reindexed TESSA vertex/joint domain.
- Trusted launcher-owned axis_math registry must bind source tensor digest,
  dataset manifest, AXIS teacher digest, source qualification key, and TESSA
  source mesh-address digest. A batch payload cannot self-approve itself.
- DOF labels additionally require native authored-limit or independent
  intervention witness; authored control-rest and anchor-local labels need
  explicit source authoring authority. Pose matrices need independent LBS truth.
- Registry digest is recorded in checkpoint save/resume and the objective is
  included in training phase-owned loss groups.
- No data were removed, rewritten or downloaded in this commit.

## Honest remaining work

1. Inventory the licensed 21k+ source corpus for exact control object, DOF,
   constraint, named parent, nondeforming bones, native anchor, and pose truth;
   preserve all such fields.
2. Extract and qualify native source matrices, source skin, source control
   interventions, and source attachment labels. No absent field becomes zero.
3. Measure held-out control residual, cross-chain weight leakage and posed
   deformation under unseen character geometry.
4. Connect source-backed selected surfaces / pinned attachments to XPBD
   with explicit physics policy. This feature does not authorize automatic
   cloth parameters or make a SAM3 label an XPBD command.

## Sources

- Model: models/rss_vnext/experts/axis/rig_program_model_v6.py
- Decoder: models/rss_vnext/experts/axis/rig_program_decode_v6.py
- Skin: models/rss_vnext/experts/mira/skin_fsq_codec_v6.py
- Added losses: models/rss_vnext/training_rig_program_grounding_v1.py
- Trainer: models/rss_vnext/training_engine.py
- Phase contract: models/rss_vnext/training_phase_contract_v1.py
- Tests: tests/models/test_rig_program_source_grounding_v1.py
- CI: .github/workflows/rss_vnext_rig_program_math.yml
