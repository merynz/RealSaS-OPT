# RSS.vNext neural code-freeze research gates — NOT frozen

**Scope/sequence:** This is a test plan for the upstream-derived uncertainties, **not** the final architecture or Truth Foundry target schema. Keep the product and expert names **RSS.vNext / IRIS / TESSA / AXIS / MIRA**. All user-observed inference input is exactly `FRONT + BACK`, never an external 8-view sequence. Internal 6/8-view construction remains eligible as a modeled hypothesis.

## Gate R1 — 2 observed views, 0 privileged auxiliary observations

**Inputs**: RGB(A) front/back in source camera coordinates, image masks, camera metadata where qualified. All other views, including a sampled side, are `PREDICTED` with lineage to these two source images; a projected source depth/normal estimate is also predicted, not an additional observation.

Compare using a fixed training subset and same eligible targets:
- A: direct front/back → shared scene features, no synthetic views;
- B: front/back → cross-view feature fusion → six constructed auxiliary view hypotheses → same scene backend;
- C: same with eight constructed hypotheses;
- D: constructed-view *latent tokens only* instead of RGB raster/resampling to avoid unnecessary information loss and accumulated hallucination.

**Ablations:** source front/back reprojection agreement, unseen side truth held out *only as supervision/evaluation*, pose-conditioned surface reconstruction and graphics ownership; source leakage via camera target label inputs; catastrophic forgetting on mismatched two-view art styles; uncertainty calibration for hidden surfaces; latency/VRAM. Reject any fusion strategy where constructed RGB is silently promoted to source appearance authority.

*Code inspiration only:* StdGEN `infer_multiview.py`, `multiview/pipeline_multiclass.py`; neither is a validated 2-observation model.

## Gate R2 — one shared feature state, multiple learned experts

Compare:
- S: frozen scene state copied/reduced into disconnected downstream predictors (baseline);
- X: structural queries and control/skin states cross-attend the shared scene carrier at fixed justified intervals;
- Y: shared sparse-latent geo/skeleton/skin stack with bidirectional adapters / repeated fusion;
- Z: mixed cross-attention + persistent source-feature querying for exact-addressed mesh vertices and appearances.

Record actual forward graph, tensor dimensions, token counts, position/camera encodings, query cardinality, all detach/freeze points, gradient norms to source encoder, local/proposal memory aliasing, truncation/compression ratios, activation memory and batch throughput. A single Python `nn.Module` wrapper without a tested gradient/information path is NOT a successful unified model.

**Upstream caution:** AniGen `configs/slat_dae.json` sets optional feature cross-exchange flags false; `slat_flow_auto.json` uses active coupled three-stream flow. Test these separately. SkinTokens uses a pointcloud prefix to causal rig generation; it does not directly solve an RSS persistent-field query API.

## Gate R3 — topology that can survive motion (learned vs deterministic)

Split into:
- `structure_truth`: qualified reference faces, components, chart/boundary/seam/contact/attachment candidates;
- `geometry_surrogate`: differentiable same-face rest/posed Jacobian/strain/area/normal source-relative losses;
- `discrete_decision`: topology token and edge/face choices; structural candidate choice/counterfactual ranking requires a real estimator or teacher targets;
- `qualification`: hard G3 micro-stress, G3B skin-topology, full Stage35 clip/playback and render-consequence checks, exact M/W/rig identity.

Source code to preserve: `models/tessa/v1/mechanical_objective_v1.py` numerical surrogate and `compiler/realsas_compiler_core/mesh/deformation_stress_v2.py` G3 court. Avoid rest-normal sign as foldover detector. Positive source teacher response does not by itself prove optimal triangle placement. On source movement itself violating absolute thresholds, record source-relative quality and abstain on absolute product PASS.

Acceptance: same static accuracy but improved unseen pose behavior on articulations; no judge bypass, teacher weights or source clips in inference path.

## Gate R4 — generic semantics and attachment ownership

Core distinction: observation/pixel class, persistent region, geometric component, independent drawable, mechanical attachment, solver socket and user-edit handle are **not interchangeable**.

Compare variable-cardinality region tokens, learned relation graph and anonymous region slots (with stable correspondence), plus 2D graphics consequence tests. Semantic labels must have `UNKNOWN` capability and no human-only fixed segmentation. Cross-view region matching only for confirmed or uncertainty-represented correspondences. Test rigid prop, cloth, appendage, multi-material single mesh, mono-material multi-mesh, embedded object and fully occluded component cases. Compiler decides whether a drawable split is mandatory / alternative / indeterminate based on control and edit consequences, not raw semantic name.

StdGEN's VRM material-name parser is only a source-specific *weak annotation witness* and must fail closed on unsupported assets.

## Gate R5 — control affordance vs standard solver lowering

Neural output may learn evidence-supported joint graph, rest frames, anatomical DOF/limits, effector potential, chain identity, socket/attachment support and sparse skin codes on the EXACT structural mesh.

Compiler may assemble **generic** FK/IK/constraint primitives, anchor transforms, contacts and skin evaluation from typed, sufficiently supported control capabilities. Compiler **must not** synthesize unsupported anatomical intent, arbitrary parent reassignments, guessed skin weights or an ambiguous attachment ownership purely to pass a demo.

Baseline tests:
- predicted skeleton+weights without learned control affordances;
- learned affordance + deterministic solver lowering;
- fully decoded control DSL event stream;
- hybrid learned DSL + compiler normalizer.

Compare edit affordance completeness, user handle count/semantic consistency, solver reachability, hard/soft constraint satisfaction, tree validity, support-foot locking, sword-tip drag with whole-body compensation and repeat after 45° camera/orientation change; adversarial morphologies, rigid tool changes and underconstrained cases. Record fail/unknown rather than forced PASS. Performance targets must be measured rather than invented.

## Gate R6 — product representation and zero-loss lineage

Verify:
- original observed image tensors are immutable appearance authority;
- predicted materials/views never rewrite source evidence;
- source patch surfaces/open meshes remain separately represented where solid sign cannot be justified;
- current canonical product v3.2 watertight-required carrier conflict is resolved by an explicit versioned policy (or a rejection mode), never undocumented auto-welding;
- every remesh/UV split/simplification publishes vertex/face IDs and correspondence proof;
- changing source/mesh/rig identity invalidates downstream skin, controls, graphics, CAA and product revision or requires qualified remap;
- serialization → reopen → edit → export preserves controllability and visible appearance.

Hard courts must include damage/outlier cases and strict error reasons. 'Compiler can fix it' does not qualify a neural incapacity, and permissive postprocessing must not mask it.

## Gate R7 — commercial and reproducible engineering

- Pin source commits and recursively enumerate **all** transitive code libraries, nested proprietary/non-commercial notices, licenses and copied-file origins. Never paste questionable source into a product runtime.
- Verify architecture implementation smoke tests: genuine 2-view shape/error cases, CPU forward and backward gradients on shared fields, masked loss with missing targets, batch-size>1 inference where required, checkpoint save/reload and optimizer+RNG+sampler resume, versioned feature schema.
- Test VRAM/latency and dynamic memory budgets before adopting large external reference widths. Source configs are examples, not RSS training prescriptions.
- Final freeze requires identified code paths, explicit source-correctness authority, all training data privilege boundaries, output grammar, shape/dtype/layout, training and runtime error behavior.

**Only AFTER passing/neatly versioning these gates** should the second project begin: final normative supervision/truth inventory for the notebook, including per-target mask and provenance. The first project's temporary source-code archaeology evidence lists are not that schema.

## Current status

R1–R7: NOT CLOSED. Publication of this plan alone makes no gate PASS; a model code freeze, training ready state and deployed puppet remain unproven. The existing 2026-11-09 design freeze is preserved as a reference; any incompatible changes require a new versioned architecture decision, not an undocumented edit.
