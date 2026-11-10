# RSS.vNext — animation-ready topology **must be learned at construction time**

**State: implemented gradient pathways + synthetic CPU verification; NOT an architectural scientific freeze, not actual source-qualified training.**

## Research conclusion from inspected source code

- StdGEN's S-LRM: image → triplane → shared geometry/semantic readouts and semantic-conditioned FlexiCubes extraction. Nothing in published inference source makes emitted triangles intrinsically deformation-suitable.
- SkinTokens/UniRig: native mesh → causal skeleton/skin or bone/point skin attention; UniRig includes LBS posed-vertex training loss. Inference is conditioned on **an existing mesh**, so neither source proves a topology-choice learning objective.
- AniGen: geometry/skeleton/skin coupled latent flows. It can emit all together, but \`configs/slat_dae.json\` disables optional inner-VAE cross attentions; released reconstruction losses teach posed geometry/skin fidelity, NOT a verified counterfactual *same-static-shape, alternative-triangulation* deformation preference.
- RigMo code [\`step1x3d_geometry/models/autoencoders/mesh_motion_vae.py\`](https://github.com/haoz19/RigMo/blob/main/step1x3d_geometry/models/autoencoders/mesh_motion_vae.py) contains \`TopologyAwareAttention\` around lines 506–596, spatial motion encoder and temporal Transformer (around 438–505), Gaussian Skinning LBS around 875–1073, and VAE loss around 1409+. Its data loader consumes motion vertex sequences with **given topology**. It supports joint representation of motion and geometry **once faces exist**; it is not evidence that discrete face generation learns to prefer deformation-ready connectivity.

**Consequent RSS requirement:** future motion and articulation must supervise *the structural token policy itself*, not be used only as post-generated mesh repair. Runtime Compiler may verify/reject a poor mesh, but has no license to invent different triangles or repair unsupported anatomy to make a demo PASS.

## Required two-gradient formulation

Let Z be the 2-source-view shared carrier; let T be grammar-valid discrete component-aware face tokens; let V(T,Z) be corresponding rest vertex XYZ; let W(T,Z) be predicted skin and A(Z) mechanical affodances. Let P be qualified source-relative pose operations and S_P source reference surface samples.

1. **Continuous, exact-indexed path**:
  \`L_cont = static_surface(V,T,S_rest) + E_p static_pose_surface(LBS(V,W,P),T,S_P) + G3_relative_Jacobian_and_strain(...)\`.
  With fixed T, gradients update V, carrier, learned local mechanics and W. Triangle quadrature points are sampled from actual faces to avoid pretending arbitrary grid queries are mesh vertices. The same index IDs are reused for LBS, cut/attachment and potential graphics. This by itself does NOT update a different discrete T.
2. **Discrete topology policy path**:
  Produce structurally distinct, statically comparable candidates \`T_i\` for the **same source**; evaluate each under the **same qualified teacher skin-transfer protocol and identical pose suite P**. \`c_i\` is a source-relative motion cost, lower is better. Score each candidate by causal length-normalized \`l_i = log p_theta(T_i|Z)\`. For qualified pairs with comparable static fidelity and significant motion gap:
  \`L_pref = mean_(w,l) softplus(-(l_w - l_l)/tau)\`.
  This loss has an **actual derivative with respect to structural token logits and all upstream shared memory** without pretending G3's non-differentiable triangle selection is differentiable.
3. **Motion-before-topology information**:
  Add a trainable articulation-prior residual and a typed affordance projection to the STRUCTURAL memory *before* T is predicted; gradients from structural CE and L_pref propagate through these into the same Z/H. No teacher rig or hidden view enters this inference conditioning.
4. **Optional bounded residual refinement**:
  A component-safe neighbor attention block may polish local XYZ and skin while preserving vertex/face IDs, but dynamic outcome after that refinement **must not** be presented as the reward for the initial structural policy. \`RSSvNextMotionCourt.evaluation_stage\` requires \`PRE_DYNAMIC_REFINEMENT_STRUCTURAL_MESH\`, preventing repair-conditioned preference labels from hiding poor T.

Source anchors for implemented paths:
- \`models/rss_vnext/motion_aware_topology_learning.py\`
- \`models/rss_vnext/animation_ready_mesh.py\`
- \`models/rss_vnext/unified_character_model.py\`
- \`models/rss_vnext/dynamic_refinement.py\`
- \`models/tessa/v1/mechanical_objective_v1.py\`
- \`models/tessa/v1/sequence_codec_v1.py\`
- \`models/tessa/v2/court_v2.py\`
- \`models/tessa/v31/model_v31.py\`

## Critical truth-qualification prerequisites (NOT the final notebook schema)

- Native source mesh + original source rig/skin/pose must have verified SAME identity, coordinate bind, and source rights. Wrong joint matrix frame or unqualified vertex correspondence makes L_pref a wrong teacher.
- Source pose itself may violate absolute G3 gates. Keep absolute hard court and source-relative deformation error separate. Neither a perfectly plausible T nor a low synthetic pose loss implies source-true appearance or fully controllable mechanics.
- Candidate comparisons require **static-quality equivalence**, a source-relative mechanical response, identical motion suite, and identical teacher skin-transfer method. Otherwise the preference loss may reward a worse geometry, a favorable control rig, or a degenerate rigid blob.
- Source relative posed surface comparison must use surface sampling/area weighting, NOT only nearest source vertex correspondence across alternative topology; full ray-stack/appearance consequences remain their own owner.
- All candidate hashes are SHA256-bound to exact padded topology token content. All source/canonical/pose/skin protocol identities must match. Explicit no-pair -> abstain (never invent label).
- Cross-morphology dataset sampling and learning curves must not be contaminated by many views/clips of the same character counted as independent characters. Pose suite coverage should include flexion, twist, wings/tails, non-rigid cloth/rigid attachments when supported by verified source.

## What has actually been tested

The hosted CPU tests establish:
- A geometric counterexample where two triangulations cover the same rest square but produce distinct **posed triangle surfaces** under the same source joint translation.
- Corresponding source-relative consequence can be converted to a motion preference that yields the correct **sign of gradient** on causal topology log probabilities.
- The topology preference gradient reaches the V3.1 token decoder and shared 2-view scene encoder as well as the pre-topology articulation prior.
- Continuous posed-surface loss gradients reach actual rest mesh features and predicted skin weights, not an unrelated surface grid.
- Bad candidate hash, missing/mismatched canonical/pose/skin evidence, missing pose mask and post-refinement reward stage fail closed.

These are **mechanism tests on tiny synthetic assets**, not evidence that the untrained network has learned animation-ready mesh generation. No final scientifically validated hyperparameters, G3 reward weights, per-morphology qualification or true-source training acceptance yet.

## Next code-freeze gates, not deferred product repair

- Roll out multiple meaningful free-running candidates with strict V3.1 grammar and independent static G3 qualification; the first graph must not be accepted merely for syntactic validity.
- Derive qualified pose/skin transfer and explicit topology preference costs from actual admitted source bytes and exact candidate mesh addressing, not manually supplied unit-test scalar costs.
- Compare candidate static-equivalent pair metrics on held-out human, quadruped, wing/tail, modular and prop meshes. Prove improved dynamic metrics over static-only CE.
- Ensure realistic token length/VRAM and multi-subject batching under the target deployment resource budget; source config's 24×1024 decoder may be too expensive at 30K faces.
- Demonstrate that edit controls/graphics consume exactly the produced triangle set without silent remesh or source appearance change, and fail closed on unsupported open mesh/unknown sign.

Only after these tests and the rest of the united model source-graph audit can the **architecture code freeze** be declared; the final Truth Foundry supervision/mask notebook then derives from it.
